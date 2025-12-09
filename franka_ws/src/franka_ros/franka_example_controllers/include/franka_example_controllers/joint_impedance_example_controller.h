// // Copyright (c) 2017 Franka Emika GmbH
// // Use of this source code is governed by the Apache-2.0 license, see LICENSE
// #pragma once

// #include <memory>
// #include <string>
// #include <vector>

// #include <controller_interface/multi_interface_controller.h>
// #include <hardware_interface/joint_command_interface.h>
// #include <hardware_interface/robot_hw.h>
// #include <realtime_tools/realtime_publisher.h>
// #include <ros/node_handle.h>
// #include <ros/time.h>

// #include <franka_example_controllers/JointTorqueComparison.h>
// #include <franka_hw/franka_cartesian_command_interface.h>
// #include <franka_hw/franka_model_interface.h>
// #include <franka_hw/trigger_rate.h>
// #include <std_msgs/Float64MultiArray.h>
// #include <mutex>
// #include <franka_hw/franka_state_interface.h>
// #include <franka_hw/franka_cartesian_command_interface.h>
// namespace franka_example_controllers {
//   class JointImpedanceExampleController : public controller_interface::MultiInterfaceController<
//                                               franka_hw::FrankaModelInterface,
//                                               franka_hw::FrankaStateInterface,
//                                               hardware_interface::EffortJointInterface> {
// // class JointImpedanceExampleController : public controller_interface::MultiInterfaceController<
// //                                             franka_hw::FrankaModelInterface,
// //                                             hardware_interface::EffortJointInterface,
// //                                             franka_hw::FrankaPoseCartesianInterface> {
//  public:
//   bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
//   void starting(const ros::Time&) override;
//   void update(const ros::Time&, const ros::Duration& period) override;

//  private:
//   // Saturation
//   std::array<double, 7> saturateTorqueRate(
//       const std::array<double, 7>& tau_d_calculated,
//       const std::array<double, 7>& tau_J_d);  // NOLINT (readability-identifier-naming)

//   ros::Subscriber sub_target_;
//   std::mutex target_mutex_;
//   std::array<double, 7> q_target_{};
//   std::array<double, 7> last_tau_cmd_{};

//   std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
//   ros::Subscriber joint_target_sub_;
//   std::array<double, 7> target_positions_from_topic_{};
//   void jointTrajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
//   bool has_target_{false};
//   void targetCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);

//   // std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
//   std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
//   std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
//   std::vector<hardware_interface::JointHandle> joint_handles_;

//   static constexpr double kDeltaTauMax{1.0};
//   double radius_{0.1};
//   double acceleration_time_{2.0};
//   double vel_max_{0.05};
//   double angle_{0.0};
//   double vel_current_{0.0};

//   std::vector<double> k_gains_;
//   std::vector<double> d_gains_;
//   double coriolis_factor_{1.0};
//   std::array<double, 7> dq_filtered_;
//   std::array<double, 16> initial_pose_;
//   std::array<double, 7> q_ref_{};

//   franka_hw::TriggerRate rate_trigger_{1.0};
//   std::array<double, 7> last_tau_d_{};
//   realtime_tools::RealtimePublisher<JointTorqueComparison> torques_publisher_;
// };

// }  // namespace franka_example_controllers
// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE



// #pragma once

// #include <memory>
// #include <string>
// #include <vector>
// #include <array>
// #include <mutex>
// #include <cmath> 
// #include <franka_hw/trigger_rate.h>
// #include <realtime_tools/realtime_publisher.h>

// #include <controller_interface/multi_interface_controller.h>
// #include <hardware_interface/joint_command_interface.h>
// #include <hardware_interface/robot_hw.h>
// #include <realtime_tools/realtime_publisher.h>
// #include <ros/node_handle.h>
// #include <ros/time.h>
// #include <ros/subscriber.h>

// #include <franka_example_controllers/JointTorqueComparison.h>
// #include <franka_hw/franka_cartesian_command_interface.h>
// #include <franka_hw/franka_model_interface.h>
// #include <franka_hw/trigger_rate.h>
// #include <trajectory_msgs/JointTrajectory.h>

// namespace franka_example_controllers {

// class JointImpedanceExampleController : public controller_interface::MultiInterfaceController<
//                                             franka_hw::FrankaModelInterface,
//                                             hardware_interface::EffortJointInterface,
//                                             franka_hw::FrankaPoseCartesianInterface> {
//  public:
//   bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
//   void starting(const ros::Time&) override;
//   void update(const ros::Time&, const ros::Duration& period) override;

//  private:

//     // === 接收 Python 送進來的期望關節角 (q_d) ===
//   void jointTrajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
//   ros::Subscriber sub_cmd_;
//   std::mutex cmd_mtx_;
//   std::array<double, 7> q_d_target_{};    // 來自 topic 的目標
//   std::array<double, 7> q_d_filtered_{};  // 指令低通後的目標
//   std::array<double, 7> q_d_prev_{};      // 給 dq_d 估測
//   std::array<double, 7> dq_d_target_{};   // 期望速度（若未提供則由差分估）
//   bool got_cmd_{false};
//   double cmd_alpha_{0.2};                 // 指令濾波係數 (0~1)

//   // Saturation
//   std::array<double, 7> saturateTorqueRate(
//       const std::array<double, 7>& tau_d_calculated,
//       const std::array<double, 7>& tau_J_d);  // NOLINT (readability-identifier-naming)

//   std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
//   std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
//   std::vector<hardware_interface::JointHandle> joint_handles_;

//   static constexpr double kDeltaTauMax{1.0};
//   double radius_{0.1};
//   double acceleration_time_{2.0};
//   double vel_max_{0.05};
//   double angle_{0.0};
//   double vel_current_{0.0};

//   std::vector<double> k_gains_;
//   std::vector<double> d_gains_;
//   double coriolis_factor_{1.0};
//   std::array<double, 7> dq_filtered_;
//   std::array<double, 16> initial_pose_;

//   franka_hw::TriggerRate rate_trigger_{1.0};
//   std::array<double, 7> last_tau_d_{};
//   realtime_tools::RealtimePublisher<JointTorqueComparison> torques_publisher_;
// };

// }  // namespace franka_example_controllers

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>

#include <ros/node_handle.h>
#include <ros/ros.h>
#include <ros/subscriber.h>
#include <ros/time.h>

#include <realtime_tools/realtime_publisher.h>

#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/trigger_rate.h>

#include <trajectory_msgs/JointTrajectory.h>

#include <franka_example_controllers/JointTorqueComparison.h>

#include <fstream>


namespace franka_example_controllers {

/**
 * @brief Joint impedance example controller for Franka Panda.
 *
 * 介面：
 *  - FrankaModelInterface：取得動力學模型量（Coriolis、Gravity 等）
 *  - EffortJointInterface：輸出關節力矩命令
 *  - FrankaPoseCartesianInterface：讀/寫笛卡兒末端位姿（亦可取 robot state）
 */
class JointImpedanceExampleController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaModelInterface,
          hardware_interface::EffortJointInterface,
          franka_hw::FrankaPoseCartesianInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void stopping(const ros::Time& /*time*/) override;

 private:
  // === Topic callback: 接收外部關節指令 (JointTrajectory) ===
  void jointTrajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);

  // === 限制力矩變化率，避免跳變 ===
  std::array<double, 7> saturateTorqueRate(const std::array<double, 7>& tau_d_calculated,
                                           const std::array<double, 7>& tau_J_d);

  // --- ROS 訂閱 / 發佈 ---
  ros::Subscriber sub_cmd_;  // 訂閱關節目標 (JointTrajectory)
  realtime_tools::RealtimePublisher<JointTorqueComparison> torques_publisher_;

  // --- 指令狀態（由 topic 餵入）---
  std::mutex cmd_mtx_;
  std::array<double, 7> q_d_target_{};    // 目標關節角 (rad)
  std::array<double, 7> q_d_filtered_{};  // 低通後目標
  std::array<double, 7> q_d_prev_{};      // 供差分估測 dq_d
  std::array<double, 7> dq_d_target_{};   // 目標關節速度 (若未提供以差分估)
  bool got_cmd_{false};
  double cmd_alpha_{0.2};                 // 指令一階濾波係數 [0,1]

  // --- Franka 介面/Handle ---
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  // 目標（原始/濾波）、上一輪濾波值
  std::array<double, 7> target_pos_raw_{};
  std::array<double, 7> target_pos_filt_{};
  std::array<double, 7> prev_target_pos_filt_{};

  // 每關節最大追隨速度 (rad/s)；依實機調
  std::array<double, 7> max_vel_{{0.7, 0.7, 0.7, 0.8, 0.9, 1.2, 1.2}};

  // 濾波器參數
  double fc_cmd_{4.0};   // 目標位置低通截止頻率(Hz)：2~6 常用
  double alpha_cmd_{0.0};

  // 目標是否已收到、看門狗
  bool have_target_{false};
  ros::Time last_msg_stamp_;
  ros::Duration watchdog_{ros::Duration(0.2)};  // 200ms 沒資料就 hold

  // --- 控制/保護參數 ---
  static constexpr double kDeltaTauMax{1.0};  // 每迭代允許的最大力矩變化 (Nm)
  double radius_{0.1};
  double acceleration_time_{2.0};
  double vel_max_{0.05};
  double angle_{0.0};
  double vel_current_{0.0};

  // --- 阻抗增益與補償 ---
  std::vector<double> k_gains_;  // size = 7
  std::vector<double> d_gains_;  // size = 7
  double coriolis_factor_{1.0};

  // --- 狀態暫存 ---
  std::array<double, 7> dq_filtered_{};
  std::array<double, 16> initial_pose_{};
  std::array<double, 7> last_tau_d_{};
  ros::Subscriber joint_target_sub_;
  std::array<double, 7> target_positions_from_topic_{};

  // --- 發佈頻率限制 ---
  franka_hw::TriggerRate rate_trigger_{1.0};
};

}  // namespace franka_example_controllers
