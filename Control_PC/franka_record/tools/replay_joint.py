from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from franka_msgs.msg import FrankaState
from franka_gripper.msg import MoveAction, MoveGoal
import actionlib
from dynamic_reconfigure.client import Client

from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import math
import signal
import time
import signal
import sys
import rospy

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True

class FrankaReplayEpisode:
    def __init__(self):
        rospy.init_node("franka_replay_episode", anonymous=False)
        self.file_path = os.path.expanduser('~/.cache/huggingface/lerobot/ethanCSL/pick_n_place_50_joint/data/chunk-000/episode_000002.parquet')
        self.obs_states = self.parquet_reader()

        self.replayed_values = []
        self.dataset_values = []
        # Publisher
        self.joint_pub = rospy.Publisher(
            "/position_joint_trajectory_controller/command",
            JointTrajectory,
            queue_size=10
        )

        # Gripper action client
        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move",
            MoveAction
        )
        rospy.loginfo("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server connected.")

        # Create hard-coded PoseStamped (actually a JointTrajectory here)
        self.pose_msg = JointTrajectory()
        self.pose_msg.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        # Optionally: send gripper width once
        self.send_gripper_command(width=0.08)
        self.pre_width = 0.08

        rospy.loginfo("Publishing hard-coded pose then initializing actual robot before replay.")
        self.run_loop()

    def to_joint_trajectory(self, positions, time_s=0.2):
        msg = JointTrajectory()
        msg.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        pt = JointTrajectoryPoint()
        pt.positions = list(positions[:7])
        pt.time_from_start = rospy.Duration(time_s)
        msg.points = [pt]
        return msg
    
    def initialize_franka_positions(self):
        """
        Move the robot smoothly from its current joint state to the first dataset position
        using a single, time-consistent JointTrajectory.
        """
        # 1) Get initial observation robustly
        try:
            initial_obs = self.obs_states.iloc[0]
        except Exception:
            initial_obs = self.obs_states[0]

        # 2) Parse target and width
        joint_keys = [f"joint{i}_state" for i in range(7)]
        target = None
        width = None
        if isinstance(initial_obs, (dict, pd.Series)):
            try:
                target = [float(initial_obs[k]) for k in joint_keys]
            except Exception:
                target = None
            try:
                width = float(initial_obs["width"]) if "width" in initial_obs else None
            except Exception:
                width = None
        if target is None:
            try:
                target = [float(x) for x in initial_obs[:7]]
            except Exception:
                rospy.logerr("Unable to parse initial observation for joint positions.")
                return
            try:
                width = float(initial_obs[7])
            except Exception:
                width = None

        # 3) Read current joint state; if unavailable, fall back to target
        try:
            js = rospy.wait_for_message("/joint_states", JointState, timeout=2.0)
            if js.name and len(js.position) >= 7:
                start = []
                for jn in [
                    "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
                    "panda_joint5","panda_joint6","panda_joint7",
                ]:
                    if jn in js.name:
                        start.append(js.position[js.name.index(jn)])
                    else:
                        start = list(js.position[:7])
                        break
            else:
                start = list(js.position[:7])
        except Exception:
            rospy.logwarn("Could not read /joint_states; will start from target to avoid abrupt move.")
            start = target[:]

        start_arr = np.array(start, dtype=float)
        target_arr = np.array(target, dtype=float)
        delta = target_arr - start_arr
        max_delta = float(np.max(np.abs(delta)))

        # If already near target, just set gripper and return
        pos_tol = 1e-3
        if max_delta <= pos_tol:
            rospy.loginfo("Robot already at initial pose (within %.4f rad).", pos_tol)
            if width is not None:
                self.send_gripper_command(width=width)
                self.pre_width = width
            return

        # 4) Time parameterization based on a conservative max joint velocity
        v_max = 0.05  # rad/s (more conservative to avoid reflex)
        total_time = max_delta / v_max
        dt = 0.1    # controller-friendly sample period (100 ms for smoother motion)
        steps = int(math.ceil(total_time / dt))
        steps = max(steps, 30)      # ensure enough waypoints for smooth motion
        steps = min(steps, 3000)    # cap for safety

        rospy.loginfo("Init traj: max_delta=%.4f rad, total_time=%.2fs, steps=%d, dt=%.3fs",
                    max_delta, steps * dt, steps, dt)

        # 5) Build one trajectory with a start point at current pose (t=0)
        traj = JointTrajectory()
        traj.joint_names = [
            "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
            "panda_joint5","panda_joint6","panda_joint7",
        ]
        points = []

        # First point: current pose at t=0 (vel=0 helps avoid jerk)
        p0 = JointTrajectoryPoint()
        p0.positions = start_arr.tolist()
        p0.velocities = [0.0] * 7
        p0.accelerations = [0.0] * 7
        p0.time_from_start = rospy.Duration(0.0)
        points.append(p0)

        # Intermediate points to target with velocity and acceleration limits
        all_positions = np.linspace(start_arr, target_arr, steps + 1)[1:]  # exclude start (already added)
        for i, pos in enumerate(all_positions, start=1):
            pt = JointTrajectoryPoint()
            pt.positions = pos.tolist()
            pt.velocities = [0.0] * 7  # Add zero velocities to avoid reflex
            pt.accelerations = [0.0] * 7  # Add zero accelerations to avoid reflex
            pt.time_from_start = rospy.Duration(i * dt)
            points.append(pt)

        traj.points = points

        # Give the controller a small lead time
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.2)

        # 6) Publish once; let the controller execute without replacement
        try:
            self.joint_pub.publish(traj)
        except Exception as e:
            rospy.logwarn("Failed to publish init trajectory: %s", e)
            # Fallback to a single-point command with a sane duration
            final_msg = self.to_joint_trajectory(target_arr, time_s=max(steps * dt, 1.0))
            self.joint_pub.publish(final_msg)

        # 7) Set gripper width while arm is moving or right after
        if width is not None:
            try:
                self.send_gripper_command(width=width)
                self.pre_width = width
            except Exception:
                rospy.logwarn("Failed to send initial gripper width.")

        # Wait roughly for completion
        rospy.sleep(min(steps * dt + 0.5, 15.0))
        rospy.loginfo("Completed smoothed init trajectory to target.")

    def run_loop(self):
        # Ensure we initialize robot positions before starting replay
        self.initialize_franka_positions()
        self.source_replay()
        rospy.spin()

    def send_gripper_command(self, width, wait=False, timeout=None):
        goal = MoveGoal()
        goal.width = width
        goal.speed = 0.05
        self.gripper_client.send_goal(goal)
        rospy.loginfo("Sent gripper width command: %.3f m", width)
        if wait:
            if timeout is None:
                self.gripper_client.wait_for_result()
            else:
                self.gripper_client.wait_for_result(rospy.Duration(timeout))

    def parquet_reader(self):
        df = pd.read_parquet(self.file_path)
        obs_states = df['observation.state']
        print("Load Data Success")
        return obs_states

    def source_replay(self):
        rate = rospy.Rate(10)  # nominal publish rate
        cnt = 0
        force_scale = 0.005

        start_time = rospy.Time.now().to_sec()
        last_time = start_time
        window_start = start_time
        window_count = 0
        total_count = 0

        for obs in self.obs_states:
            if rospy.is_shutdown() or shutdown_requested:
                rospy.loginfo("Shutdown requested, stopping replay...")
                break

            now = rospy.Time.now().to_sec()

            self.joint_msg = JointTrajectory()
            self.joint_msg.joint_names = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ]
            point = JointTrajectoryPoint()
            point.positions = [
                obs[0],
                obs[1],
                obs[2],
                obs[3],
                obs[4],
                obs[5],
                obs[6],
            ]

            if cnt % 1 == 0: # publish every step
                try: 
                    self.joint_msg.points = [point]  # Single point trajectory
                    self.joint_pub.publish(self.joint_msg)
                    width = float(obs[7])
                except Exception:
                    width = self.pre_width
                    rospy.logwarn("Failed to parse gripper width, using previous value: %.3f m", self.pre_width)
                # send gripper command
                self.send_gripper_command(width=width)

                # === Update frequency counters ===
                total_count += 1
                window_count += 1
                dt = now - last_time
                inst_freq = 1.0 / dt if dt > 1e-6 else 0.0
                last_time = now

                # log rolling window frequency once per second
                if (now - window_start) >= 1.0:
                    elapsed = now - window_start
                    window_freq = window_count / elapsed if elapsed > 0 else 0.0
                    total_elapsed = now - start_time
                    total_avg = total_count / total_elapsed if total_elapsed > 0 else 0.0
                    rospy.loginfo(
                        "Replay frequency: window %.2f Hz | instant %.2f Hz | total avg %.2f Hz",
                        window_freq, inst_freq, total_avg
                    )
                    window_start = now
                    window_count = 0

            # store replayed values (safe fallback for width)
            try:
                gripper_val = float(obs[7]) if len(obs) > 7 else self.pre_width
            except Exception:
                gripper_val = self.pre_width

            self.replayed_values.append([
                obs[0],  # joint 1
                obs[1],  # joint 2
                obs[2],  # joint 3
                obs[3],  # joint 4
                obs[4],  # joint 5
                obs[5],  # joint 6
                obs[6],  # joint 7
                gripper_val,
            ])

            cnt += 1
            rate.sleep()

        # final overall frequency report
        end_time = rospy.Time.now().to_sec()
        total_elapsed = end_time - start_time
        if total_elapsed > 0:
            overall_freq = total_count / total_elapsed
            rospy.loginfo("Replay finished. Overall average frequency: %.2f Hz (sent %d commands in %.2f s)",
                          overall_freq, total_count, total_elapsed)
        else:
            rospy.loginfo("Replay finished. No time elapsed to compute frequency.")

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        franka_replay_episode = FrankaReplayEpisode()
    except rospy.ROSInterruptException:
        pass
