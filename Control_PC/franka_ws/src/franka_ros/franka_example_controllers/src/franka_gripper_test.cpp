#include <franka_gripper/GraspAction.h>
#include <franka_gripper/HomingAction.h>
#include <franka_gripper/MoveAction.h>

#include <actionlib/client/simple_action_client.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

using GraspClient = actionlib::SimpleActionClient<franka_gripper::GraspAction>;
using HomingClient = actionlib::SimpleActionClient<franka_gripper::HomingAction>;
using MoveClient = actionlib::SimpleActionClient<franka_gripper::MoveAction>;

int main(int argc, char** argv) {
  ros::init(argc, argv, "franka_gripper_test");
  ros::NodeHandle nh;

  HomingClient homing_client("panda/franka_gripper/homing", true);
  GraspClient grasp_client("panda/franka_gripper/grasp", true);
  MoveClient move_client("panda/franka_gripper/move", true);

  ROS_INFO("Waiting for action servers...");
  homing_client.waitForServer();
  grasp_client.waitForServer();
  move_client.waitForServer();
  ROS_INFO("Connected to gripper action servers.");

  // Step 1: Homing once
  ROS_INFO("Sending homing...");
  homing_client.sendGoal(franka_gripper::HomingGoal());
  homing_client.waitForResult(ros::Duration(5.0));

  // Step 2~n: Loop grasp <-> open
  while (ros::ok()) {
    // Grasp
    franka_gripper::GraspGoal grasp_goal;
    grasp_goal.width = 0.0;
    grasp_goal.force = 40.0;
    grasp_goal.speed = 0.1;
    grasp_goal.epsilon.inner = 0.005;
    grasp_goal.epsilon.outer = 0.005;

    ROS_INFO("Grasping...");
    grasp_client.sendGoal(grasp_goal);
    grasp_client.waitForResult(ros::Duration(5.0));
    std::this_thread::sleep_for(1s);

    // Open
    franka_gripper::MoveGoal move_goal;
    move_goal.width = 0.08;  // roughly full open
    move_goal.speed = 0.1;

    ROS_INFO("Opening...");
    move_client.sendGoal(move_goal);
    move_client.waitForResult(ros::Duration(5.0));
    std::this_thread::sleep_for(1s);
  }

  return 0;
}
