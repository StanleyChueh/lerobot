#include <franka_example_controllers/joint_impedance_example_controller.h>

#include <cmath>
#include <memory>
#include <vector>
#include <array>
#include <fstream>
#include <algorithm>
#include <iostream>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>
#include <trajectory_msgs/JointTrajectory.h>

namespace {
constexpr double kDeltaTauMax{2.0};

// ------------------- spline 相關變數 -------------------
std::vector<std::vector<double>> spline_path_;
size_t spline_index_ = 0;
// bool new_target_received_ = false;

// ------------------- log -------------------
std::vector<std::array<double, 7>> log_tau_d;
std::vector<std::array<double, 7>> log_q;

}  // namespace

namespace franka_example_controllers {

struct TrajPoint {
  std::array<double,7> q, dq, ddq;
};
std::vector<TrajPoint> traj_path_;
size_t traj_index_ = 0;
std::array<double, 7> cb_positions{};

std::vector<TrajPoint> cubicSplineQdQdd(
    const std::vector<double>& q_start,
    const std::vector<double>& q_goal,
    const std::array<double,7>& qdot_start,
    size_t max_steps = 100,
    double max_joint_velocity = 0.5) {

  size_t dof = q_start.size();
  std::vector<TrajPoint> path;
  std::vector<double> delta(dof);
  double max_delta = 0.0;
  for (size_t j=0;j<dof;++j){ delta[j]=q_goal[j]-q_start[j]; max_delta=std::max(max_delta, std::fabs(delta[j])); }
  size_t steps = max_delta>0 ? std::max(max_steps, (size_t)std::ceil(max_delta/max_joint_velocity)) : max_steps;

  std::vector<double> v_start(dof), v_goal(dof);
  for (size_t j=0;j<dof;++j){ v_start[j]=qdot_start[j]; v_goal[j]=0.0; } // 先用0，後面還有crossfade可平順

  path.reserve(steps);
  for (size_t i=0;i<steps;++i){
    double t = (steps==1)? 1.0 : (double)i/(steps-1);
    TrajPoint p{};
    for (size_t j=0;j<dof;++j){
      double a0=q_start[j];
      double a1=v_start[j];
      double a2= 3*delta[j] - 2*v_start[j] - v_goal[j];
      double a3=-2*delta[j] +   v_start[j] + v_goal[j];
      p.q[j]   = a0 + a1*t + a2*t*t + a3*t*t*t;
      p.dq[j]  =        a1   + 2*a2*t + 3*a3*t*t;
      p.ddq[j] =                2*a2   + 6*a3*t;
    }
    path.push_back(p);
  }
  return path;
}

// ------------------- init -------------------
bool JointImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }

  node_handle.param("radius", radius_, 0.1);
  node_handle.param("vel_max", vel_max_, 0.1);
  node_handle.param("acceleration_time", acceleration_time_, 0.5);

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR("Invalid joint_names parameters!");
    return false;
  }

  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR("Invalid k_gains parameters!");
    return false;
  }
  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR("Invalid d_gains parameters!");
    return false;
  }

  double publish_rate;
  node_handle.param("publish_rate", publish_rate, 30.0);
  rate_trigger_ = franka_hw::TriggerRate(publish_rate);

  node_handle.param("coriolis_factor", coriolis_factor_, 1.0);

  // model interface
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR("Error getting model interface");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(ex.what());
    return false;
  }

  // Cartesian pose interface
  auto* cartesian_pose_interface = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface == nullptr) {
    ROS_ERROR("Error getting Cartesian pose interface");
    return false;
  }
  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(ex.what());
    return false;
  }

  // effort joint interface
  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR("Error getting effort joint interface");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(ex.what());
      return false;
    }
  }

  torques_publisher_.init(node_handle, "torque_comparison", 1);
  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0.0);

  // subscriber
  target_positions_from_topic_.fill(0.0);
  joint_target_sub_ = node_handle.subscribe(
      "/position_joint_trajectory_controller/command",
      1,
      &JointImpedanceExampleController::jointTrajectoryCallback,
      this);

  return true;
}

// ------------------- callback -------------------
void JointImpedanceExampleController::jointTrajectoryCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {

  if (msg->points.empty()) {
    ROS_WARN_THROTTLE(1.0, "JointTrajectory 沒有 points");
    return;
  }

  const auto& positions = msg->points[0].positions;
  if (positions.size() != 7) {
    ROS_WARN_THROTTLE(1.0, "positions 長度不是 7");
    return;
  }

  franka::RobotState robot_state = cartesian_pose_handle_->getRobotState();
  std::vector<double> q_curr(7);
  for (size_t i = 0; i < 7; ++i) {
    q_curr[i] = robot_state.q[i];
  }

  // spline_path_ = cubicSplineInterpolation(q_curr, positions, robot_state.dq, 100, 0.5);
  // spline_index_ = 0;
  traj_path_ = cubicSplineQdQdd(q_curr, positions, robot_state.dq,0, 1.5);
  traj_index_ = 0;

  // new_target_received_ = true;
}

// ------------------- starting -------------------
// ros::Time start_time_;
// bool stop_triggered_ = false;
// std::ofstream tau_log_file_;

void JointImpedanceExampleController::starting(const ros::Time& /*time*/) {
  // start_time_ = ros::Time::now();
  // stop_triggered_ = false;

  // tau_log_file_.open("/home/csl/franka_ws/logs/franka_tau_log_pd_100100.csv");
  // tau_log_file_ << "step";
  // for (int i = 0; i < 7; i++) tau_log_file_ << ",tau" << i;
  // for (int i = 0; i < 7; i++) tau_log_file_ << ",q" << i;
  // for (int i = 0; i < 7; i++) tau_log_file_ << ",desired_q" << i;
  // for (int i = 0; i < 7; i++) tau_log_file_ << ",error_q" << i;
  // tau_log_file_ << "\n";

  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;

  franka::RobotState robot_state = cartesian_pose_handle_->getRobotState();
  for (size_t i = 0; i < 7; ++i) {
    target_positions_from_topic_[i] = robot_state.q[i];
  }

  std::array<double, 16> pose_desired = initial_pose_;
  cartesian_pose_handle_->setCommand(pose_desired);
}

// ------------------- update -------------------
void JointImpedanceExampleController::update(const ros::Time& /*time*/,
                                             const ros::Duration& /*period*/) {
  // if(stop_triggered_) return;  // 已停止就跳過 update

  franka::RobotState robot_state = cartesian_pose_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();
  static double max_max_value = 0.0;

  double alpha = 0.999; 
  for (size_t i = 0; i < 7; ++i) {
    dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * robot_state.dq[i];
  }
  if (!traj_path_.empty() && traj_index_ < traj_path_.size()) {
  for (size_t i = 0; i < 7; ++i) {
    target_positions_from_topic_[i] = traj_path_[traj_index_].q[i];
    
  }
  traj_index_++;

}


  std::array<double, 7> tau_d_calculated;
  for (size_t i = 0; i < 7; ++i) {
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
                          k_gains_[i] * (target_positions_from_topic_[i] - robot_state.q[i]) +
                          d_gains_[i] * (robot_state.dq_d[i] - dq_filtered_[i]);
  }
  std::array<double, 7> tau_d_pre = { 0, 0, 0, 0, 0, 0 ,0 };
  double beta = 0.1;
  for (size_t i =0; i < 7; ++i){
    tau_d_calculated[i] = beta*tau_d_pre[i] + (1 - beta)*tau_d_calculated[i];
  }
  tau_d_pre = tau_d_calculated;


  std::array<double, 7> tau_d_saturated =
      saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d_saturated[i]);
  }

  // log tau/q
  // static size_t step = 0;
  // tau_log_file_ << step;
  // for (int i = 0; i < 7; ++i) tau_log_file_ << "," << tau_d_calculated[i];
  // for (int i = 0; i < 7; ++i) tau_log_file_ << "," << robot_state.q[i];
  // for (int i = 0; i < 7; ++i) tau_log_file_ << "," << target_positions_from_topic_[i];
  // for (int i = 0; i < 7; ++i) tau_log_file_ << "," << target_positions_from_topic_[i]-robot_state.q[i];

  // tau_log_file_ << "\n";
  // step++;

  // ros::Duration elapsed = ros::Time::now() - start_time_;
  // std::cout << elapsed.toSec() <<std::endl;
  // if (elapsed.toSec() >= 2000.0) {
  //   ROS_INFO("70 s elapsed. Stopping controller and saving log...");
  //   tau_d_calculated = {0, 0, 0, 0, 0, 0, 0};
  //   for (size_t i = 0; i < 7; ++i) {
  //     joint_handles_[i].setCommand(tau_d_saturated[i]);
  //   }
  //   stop_triggered_ = true;
  //   if(tau_log_file_.is_open()){
  //     tau_log_file_.close();
  //     ROS_INFO("Log saved to /home/csl/franka_ws/logs/franka_tau_log.csv");
  //   }
  // }
}
void JointImpedanceExampleController::stopping(const ros::Time& /*time*/) {
}
// ------------------- saturateTorqueRate -------------------
std::array<double, 7> JointImpedanceExampleController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double diff = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(diff, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointImpedanceExampleController,
                       controller_interface::ControllerBase)
