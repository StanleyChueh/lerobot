#!/usr/bin/env python3
# client.py

import os
import sys
import math
import time
import signal
import json
import struct
import socket

import rospy
import actionlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, WrenchStamped

from franka_gripper.msg import MoveAction, MoveGoal, GraspGoal, GraspAction
from franka_gripper.msg import GraspEpsilon, GraspResult
from actionlib_msgs.msg import GoalStatus
from collections import deque
import threading

from datetime import datetime

shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)

class FrankaLeRobotEvaluation:
    def __init__(self):
        rospy.init_node("franka_le_robot_evaluation", anonymous=False)

        # --- Publishers / Action clients
        self.pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=10,
        )

        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move", MoveAction
        )
        self.grasp_client = actionlib.SimpleActionClient(
            "/franka_gripper/grasp", GraspAction
        )
        self.move_client = actionlib.SimpleActionClient(
            "/franka_gripper/move", MoveAction
        )

        rospy.loginfo("Waiting for gripper action servers...")
        self.gripper_client.wait_for_server()
        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()
        rospy.loginfo("Gripper action servers connected.")

        # --- Subscribers
        self.pose_sub = rospy.Subscriber("/robot0_eef_pose", PoseStamped, self.pose_cb)
        self.gripper_sub = rospy.Subscriber(
            "/robot0_gripper_width", Float32, self.gripper_cb
        )

        # Image subscribers (raw Image topics, no cv_bridge)
        self.eye_front_sub = rospy.Subscriber(
            "/robot0_eye_in_hand_image_front", Image, self.eye_front_cb, queue_size=1
        )
        self.eye_rear_sub = rospy.Subscriber(
            "/robot0_eye_in_hand_image_rear", Image, self.eye_rear_cb, queue_size=1
        )
        self.agent_sub = rospy.Subscriber(
            "/robot0_agentview_image", Image, self.agent_cb, queue_size=1
        )

        # Optional: gripper effort
        rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self.gripper_effort_callback
        )

        # --- State
        self.gripper_state = None
        self.gripper = 0.08  # initial open 8 cm
        self.gripper_delta_accum = 0.0
        self.last_gripper_command_time = rospy.Time.now()
        self.gripper_effort = None
        self.gripper_now = 0.08

        self.pose = None

        self.eye_image_front = None
        self.eye_image_rear = None
        self.agent_image = None
        self.ext_force = 0.0
        self.action = None

        # Helper: common encodings -> dtype/channels
        self._enc2dt = {
            "mono8": ("uint8", 1),
            "8UC1": ("uint8", 1),
            "bgr8": ("uint8", 3),
            "rgb8": ("uint8", 3),
            "bgra8": ("uint8", 4),
            "rgba8": ("uint8", 4),
        }

        # --- Socket connection
        # SERVER_IP = '192.168.1.162' # FET
        #SERVER_IP = "10.100.4.141"  # TT AGX
        SERVER_IP = "10.100.4.42"  # TT PC
        # SERVER_IP = '192.168.123.128' # ASUS
        PORT = 5001
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((SERVER_IP, PORT))
        self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


        # --- Initial pose
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "panda_link0"
        rospy.loginfo("Waiting for first /robot0_eef_pose...")
        while self.pose is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        rospy.loginfo(f"Initial pose received: {self.pose}")

        self.pose_msg.header.stamp = rospy.Time.now() 

        self.pose_msg.pose.position.x = self.pose[0]
        self.pose_msg.pose.position.y = self.pose[1]
        self.pose_msg.pose.position.z = self.pose[2]
        self.pose_msg.pose.orientation.x = self.pose[3]
        self.pose_msg.pose.orientation.y = self.pose[4]
        self.pose_msg.pose.orientation.z = self.pose[5]
        self.pose_msg.pose.orientation.w = self.pose[6]
        rospy.loginfo("Using current pose as starting pose.") 

        # --- Initialize gripper opening
        self.send_gripper_command(width=self.gripper)
        rospy.loginfo("Publishing hard-coded pose at ~30Hz.")

        self.commanded_log = []
        self.actual_log = []
        self.time_log = []
        self.start_time = time.time()

        # ---------- Asynchronous action queue ----------
        # Must match the policy's n_action_steps on the server (e.g. 50 for smolvla_base).
        self.chunk_size = 50           # TODO: set this to your SmolVLA n_action_steps
        self.queue_threshold = 0.1     # "g" in the algorithm (trigger new chunk when < 30% left)

        self.action_queue = deque()    # holds [x,y,z,ox,oy,oz,ow,gripper] actions
        self._queue_lock = threading.Lock()
        self._async_inflight = False   # True while a background inference is running

        # Fill initial queue synchronously so we start with some actions
        rospy.loginfo("Requesting initial SmolVLA action chunk...")
        init_chunk = self._request_action_chunk_blocking(wait_images=True)
        if init_chunk is None or len(init_chunk) == 0:
            rospy.logwarn("Initial action chunk is empty – robot will hold pose until new chunk arrives.")
        else:
            with self._queue_lock:
                self.action_queue.extend(init_chunk)
            rospy.loginfo(f"Initial action chunk received with {len(init_chunk)} actions.")

    # ----------------- Callbacks -----------------

    def gripper_effort_callback(self, msg: JointState):
        if "panda_finger_joint1" in msg.name:
            idx = msg.name.index("panda_finger_joint1")
            self.gripper_effort = msg.effort[idx]

    def pose_cb(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.pose = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

    def gripper_cb(self, msg: Float32):
        self.gripper_now = msg.data

    def _image_msg_to_numpy(self, msg: Image):
        """
        Convert sensor_msgs/Image to BGR np.array without cv_bridge.
        Supports: mono8, 8UC1, bgr8, rgb8, bgra8, rgba8.
        """
        enc = msg.encoding.lower()
        if enc not in self._enc2dt:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        dt, ch = self._enc2dt[enc]
        arr = np.frombuffer(msg.data, dtype=np.dtype(dt))
        if ch == 1:
            img = arr.reshape((msg.height, msg.width))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = arr.reshape((msg.height, msg.width, ch))
        if enc in ("rgb8", "rgba8"):
            code = cv2.COLOR_RGB2BGR if ch == 3 else cv2.COLOR_RGBA2BGR
            img = cv2.cvtColor(img, code)
        elif enc == "bgra8":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # bgr8 is already BGR
        return img

    def eye_front_cb(self, msg: Image):
        try:
            self.eye_image_front = self._image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"eye_front_cb decode error: {e}")

    def eye_rear_cb(self, msg: Image):
        try:
            self.eye_image_rear = self._image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"eye_rear_cb decode error: {e}")

    def agent_cb(self, msg: Image):
        try:
            self.agent_image = self._image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"agent_cb decode error: {e}")

    def ext_force_callback(self, msg: WrenchStamped):
        try:
            self.ext_force = msg.wrench.force.z
        except Exception as e:
            rospy.logerr(f"Error in force callback: {e}")

    # ---------- Observation snapshot & network helpers ----------

    def _snapshot_obs(self, wait_images: bool = False, timeout: float = 5.0):
        """
        Capture a *consistent* snapshot of (state, three images).

        If wait_images=True, block up to `timeout` seconds for all images.
        """
        if self.pose is None:
            rospy.logwarn_throttle(2.0, "Pose not ready yet, cannot build observation.")
            return None

        # State = current pose + gripper width
        state = list(self.pose) + [self.gripper_now]

        def imgs_ready():
            return (
                self.eye_image_front is not None
                and self.eye_image_front.size > 0
                and self.eye_image_rear is not None
                and self.eye_image_rear.size > 0
                and self.agent_image is not None
                and self.agent_image.size > 0
            )

        if wait_images:
            t0 = time.time()
            while not imgs_ready() and not rospy.is_shutdown():
                if time.time() - t0 > timeout:
                    rospy.logwarn("Timeout waiting for images in _snapshot_obs.")
                    break
                rospy.sleep(0.05)

        if not imgs_ready():
            rospy.logwarn_throttle(2.0, "Images not ready, skipping observation snapshot.")
            return None

        # Make copies to avoid races with ROS callbacks
        eye_front = self.eye_image_front.copy()
        eye_rear = self.eye_image_rear.copy()
        agent = self.agent_image.copy()
        return state, eye_front, eye_rear, agent

    # ----------------- Gripper helpers -----------------

    def send_gripper_command(self, width: float):
        goal = MoveGoal()
        goal.width = float(width)
        goal.speed = 0.15
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result()
        result = self.gripper_client.get_result()
        rospy.loginfo("Sent gripper width command: %.5f m", width)
        rospy.loginfo(f"Gripper command result: {result}")

    # ----------------- Socket helpers -----------------

    def send_data(self, sock, data_type: str, data_bytes: bytes):
        data_type = data_type.ljust(10).encode()
        data_len = struct.pack(">I", len(data_bytes))
        sock.sendall(data_len + data_type + data_bytes)

    def recv_data(self, sock):
        raw_len = sock.recv(4)
        if not raw_len:
            return None, None
        data_len = struct.unpack(">I", raw_len)[0]
        data_type = sock.recv(10).decode().strip()
        data = b""
        while len(data) < data_len:
            packet = sock.recv(data_len - len(data))
            if not packet:
                break
            data += packet
        return data_type, data

    # ----------------- Main data path -----------------

    def save_logs(self):
        df = pd.DataFrame({
            "time": self.time_log,
            "cmd_x": [c[0] for c in self.commanded_log],
            "act_x": [a[0] for a in self.actual_log],
            "cmd_y": [c[1] for c in self.commanded_log],
            "act_y": [a[1] for a in self.actual_log],
            "cmd_z": [c[2] for c in self.commanded_log],
            "act_z": [a[2] for a in self.actual_log],
            "cmd_ox": [c[3] for c in self.commanded_log],
            "act_ox": [a[3] for a in self.actual_log],
            "cmd_oy": [c[4] for c in self.commanded_log],
            "act_oy": [a[4] for a in self.actual_log],
            "cmd_oz": [c[5] for c in self.commanded_log],
            "act_oz": [a[5] for a in self.actual_log],
            "cmd_ow": [c[6] for c in self.commanded_log],
            "act_ow": [a[6] for a in self.actual_log],
            "cmd_gripper": [c[7] for c in self.commanded_log],
            "act_gripper": [a[7] for a in self.actual_log],
        })
        df.to_csv("franka_logs.csv", index=False)

    def _run_policy_on_server(self, state, eye_front, eye_rear, agent):
        """
        Send (state, 3 images) to the SmolVLA server and get back an *action chunk*:
            [[a0...], [a1...], ...]
        This is a blocking call and should be run in a background thread during normal operation.
        """
        # 1) send state
        try:
            self.send_data(self.client_socket, "list", json.dumps(state).encode())
        except Exception as e:
            rospy.logerr(f"Failed to send state to server: {e}")
            return None

        # 2) encode & send images (we already have copies)
        ok1, img1_bytes = cv2.imencode(".jpg", eye_front)
        ok2, img2_bytes = cv2.imencode(".jpg", eye_rear)
        ok3, img3_bytes = cv2.imencode(".jpg", agent)
        if not (ok1 and ok2 and ok3):
            rospy.logerr("JPEG encoding failed for one of the images.")
            return None

        try:
            self.send_data(self.client_socket, "img1", img1_bytes.tobytes())
            self.send_data(self.client_socket, "img2", img2_bytes.tobytes())
            self.send_data(self.client_socket, "img3", img3_bytes.tobytes())
        except Exception as e:
            rospy.logerr(f"Failed to send images to server: {e}")
            return None

        # 3) receive chunk
        data_type, data = self.recv_data(self.client_socket)
        if data_type != "list":
            rospy.logerr(f"Unexpected data_type from server: {data_type}")
            return None

        try:
            chunk = json.loads(data.decode())
        except Exception as e:
            rospy.logerr(f"Failed to parse action chunk JSON: {e}")
            return None

        if not isinstance(chunk, list) or len(chunk) == 0 or not isinstance(chunk[0], list):
            rospy.logerr("Server returned malformed action chunk.")
            return None

        return chunk

    def _request_action_chunk_blocking(self, wait_images: bool) -> list | None:
        """Convenience wrapper: snapshot obs, then call server."""
        obs = self._snapshot_obs(wait_images=wait_images)
        if obs is None:
            return None
        state, eye_front, eye_rear, agent = obs
        return self._run_policy_on_server(state, eye_front, eye_rear, agent)

    # ---------- Asynchronous inference worker ----------

    def _async_infer_worker(self, state, eye_front, eye_rear, agent):
        """
        Runs in a *separate thread*:
        - calls the server
        - when the chunk arrives, replaces the action queue
        """
        try:
            new_chunk = self._run_policy_on_server(state, eye_front, eye_rear, agent)
            if new_chunk is None or len(new_chunk) == 0:
                rospy.logwarn("Async inference returned empty chunk.")
                return

            with self._queue_lock:
                # f(A_t, Â_{t+1}) in Algorithm 1.
                # Here we simply *replace* the remaining queue with the new chunk.
                # self.action_queue.clear()
                self.action_queue.extend(new_chunk)

        except Exception as e:
            rospy.logerr(f"Exception in async infer worker: {e}")
        finally:
            with self._queue_lock:
                self._async_inflight = False

    def _maybe_trigger_async_inference(self):
        """
        Implements the 'if |A_t|/n < g then ASYNCINFER(o_{t+1})' logic.
        Non‑blocking: it only *starts* a thread if needed.
        """
        with self._queue_lock:
            q_len = len(self.action_queue)
            if self.chunk_size <= 0:
                return
            frac = q_len / float(self.chunk_size)
            if self._async_inflight or frac >= self.queue_threshold:
                return  # either already running, or queue is sufficiently full

        # Snapshot a fresh observation (non‑blocking for images)
        obs = self._snapshot_obs(wait_images=False)
        if obs is None:
            return
        state, eye_front, eye_rear, agent = obs

        # Launch worker
        with self._queue_lock:
            if self._async_inflight:
                return
            self._async_inflight = True

        thread = threading.Thread(
            target=self._async_infer_worker,
            args=(state, eye_front, eye_rear, agent),
            daemon=True,
        )
        thread.start()

    # ---------- Execute one action ----------

    def _execute_action(self, action):
        """
        action: [x, y, z, ox, oy, oz, ow, gripper]
        """
        if action is None or len(action) < 8:
            return

        # Cartesian pose
        self.pose_msg.pose.position.x = float(action[0])
        self.pose_msg.pose.position.y = float(action[1])
        self.pose_msg.pose.position.z = float(action[2])
        self.pose_msg.pose.orientation.x = float(action[3])
        self.pose_msg.pose.orientation.y = float(action[4])
        self.pose_msg.pose.orientation.z = float(action[5])
        self.pose_msg.pose.orientation.w = float(action[6])
        self.pose_msg.header.stamp = rospy.Time.now()
        self.pose_pub.publish(self.pose_msg)

        # Logging commanded vs actual
        now = time.time() - self.start_time
        cmd = [
            self.pose_msg.pose.position.x,
            self.pose_msg.pose.position.y,
            self.pose_msg.pose.position.z,
            self.pose_msg.pose.orientation.x,
            self.pose_msg.pose.orientation.y,
            self.pose_msg.pose.orientation.z,
            self.pose_msg.pose.orientation.w,
            self.gripper,  # previous gripper command
        ]
        act = self.pose + [self.gripper_now]

        self.commanded_log.append(cmd)
        self.actual_log.append(act)
        self.time_log.append(now)

        # Gripper command comes from current action
        self.gripper = float(action[7])
        try:
            if self.gripper < 0.045 and self.gripper_state != "closed":
                rospy.loginfo(f"gripper state: close, {self.gripper}")
                goal = GraspGoal()
                goal.width = 0.04
                goal.speed = 0.1
                goal.force = 80.0
                goal.epsilon = GraspEpsilon(inner=0.02, outer=0.02)
                self.grasp_client.send_goal(goal)
                self.gripper_state = "closed"

            elif self.gripper >= 0.055 and self.gripper_state != "open":
                rospy.loginfo(f"gripper state: open, {self.gripper}")
                move_goal = MoveGoal()
                move_goal.width = 0.08
                move_goal.speed = 0.1
                self.move_client.send_goal(move_goal)
                self.gripper_state = "open"
        except Exception as e:
            rospy.logerr(f"Exception sending grasp/move goal: {e}")

    def evaluate(self):
        """
        One control step:
          1. Pop next action from the queue and execute it.
          2. If the queue is getting empty, trigger async inference.
        """
        # 1) Pop an action
        with self._queue_lock:
            action = self.action_queue.popleft() if self.action_queue else None

        if action is None:
            # No action available yet – just re-publish the last pose to keep impedance controller happy.
            rospy.logwarn_throttle(2.0, "Action queue empty, holding last pose.")
            self.pose_msg.header.stamp = rospy.Time.now()
            self.pose_pub.publish(self.pose_msg)
            # Try to kick off async inference if not already running
            self._maybe_trigger_async_inference()
            return

        # 2) Execute current action
        self._execute_action(action)

        # 3) Optionally start asynchronous inference if queue low
        self._maybe_trigger_async_inference()

if __name__ == "__main__":
    try:
        fk = FrankaLeRobotEvaluation()
        rate = rospy.Rate(60)  # avoid busy loop
        while not rospy.is_shutdown() and not shutdown_requested:
            fk.evaluate()
            rate.sleep()
        if rospy.is_shutdown() or shutdown_requested:
            fk.save_logs()
            fk.client_socket.close()

            df = pd.read_csv("franka_logs.csv")

            # --- Compute error
            df["err_x"] = df["act_x"] - df["cmd_x"]
            df["err_y"] = df["act_y"] - df["cmd_y"]
            df["err_z"] = df["act_z"] - df["cmd_z"]
            df["err_ox"] = df["act_ox"] - df["cmd_ox"]
            df["err_oy"] = df["act_oy"] - df["cmd_oy"]
            df["err_oz"] = df["act_oz"] - df["cmd_oz"]
            df["err_ow"] = df["act_ow"] - df["cmd_ow"]

            mean_err_x = abs(df["err_x"]).mean()
            mean_err_y = abs(df["err_y"]).mean()
            mean_err_z = abs(df["err_z"]).mean()
            mean_err_ox = abs(df["err_ox"]).mean()
            mean_err_oy = abs(df["err_oy"]).mean()
            mean_err_oz = abs(df["err_oz"]).mean()
            mean_err_ow = abs(df["err_ow"]).mean()

            max_err_x = df["err_x"].abs().max()
            max_err_y = df["err_y"].abs().max()
            max_err_z = df["err_z"].abs().max()
            max_err_ox = df["err_ox"].abs().max()
            max_err_oy = df["err_oy"].abs().max()
            max_err_oz = df["err_oz"].abs().max()
            max_err_ow = df["err_ow"].abs().max()

            print("\n=== Position Tracking Error Report ===")
            print(f"Mean Absolute Error X: {mean_err_x:.6f} m")
            print(f"Mean Absolute Error Y: {mean_err_y:.6f} m")
            print(f"Mean Absolute Error Z: {mean_err_z:.6f} m")
            print(f"Mean Absolute Error oX: {mean_err_ox:.6f} m")
            print(f"Mean Absolute Error oY: {mean_err_oy:.6f} m")
            print(f"Mean Absolute Error oZ: {mean_err_oz:.6f} m")
            print(f"Mean Absolute Error oW: {mean_err_ow:.6f} m")
            print(f"Max Absolute Error  X: {max_err_x:.6f} m")
            print(f"Max Absolute Error  Y: {max_err_y:.6f} m")
            print(f"Max Absolute Error  Z: {max_err_z:.6f} m")
            print(f"Max Absolute Error  oX: {max_err_ox:.6f} m")
            print(f"Max Absolute Error  oY: {max_err_oy:.6f} m")
            print(f"Max Absolute Error  oZ: {max_err_oz:.6f} m")
            print(f"Max Absolute Error  oW: {max_err_ow:.6f} m")
            print("======================================\n")

            # --- Plot each axis individually
            plt.figure(figsize=(10, 10))

            # X-axis
            plt.subplot(3, 1, 1)
            plt.plot(df["time"], df["cmd_x"], label="Cmd X")
            plt.plot(df["time"], df["act_x"], label="Act X")
            plt.ylabel("Position X (m)")
            plt.title("Command vs Actual - X Axis")
            plt.legend()
            plt.grid(True)

            # Y-axis
            plt.subplot(3, 1, 2)
            plt.plot(df["time"], df["cmd_y"], label="Cmd Y")
            plt.plot(df["time"], df["act_y"], label="Act Y")
            plt.ylabel("Position Y (m)")
            plt.title("Command vs Actual - Y Axis")
            plt.legend()
            plt.grid(True)

            # Z-axis
            plt.subplot(3, 1, 3)
            plt.plot(df["time"], df["cmd_z"], label="Cmd Z")
            plt.plot(df["time"], df["act_z"], label="Act Z")
            plt.xlabel("Time (s)")
            plt.ylabel("Position Z (m)")
            plt.title("Command vs Actual - Z Axis")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            filename = "franka_comparison_xyz.png"
            plt.savefig(filename)
            print(f"Saved 3-axis comparison plot to {filename}")

            plt.figure(figsize=(10, 10))
            # oX-axis
            plt.subplot(4, 1, 1)
            plt.plot(df["time"], df["cmd_ox"], label="Cmd oX")
            plt.plot(df["time"], df["act_ox"], label="Act oX")
            plt.ylabel("Orientation oX")
            plt.title("Command vs Actual - oX Axis")
            plt.legend()
            plt.grid(True)

            # oY-axis
            plt.subplot(4, 1, 2)
            plt.plot(df["time"], df["cmd_oy"], label="Cmd oY")
            plt.plot(df["time"], df["act_oy"], label="Act oY")
            plt.ylabel("Orientation oY")
            plt.title("Command vs Actual - oY Axis")
            plt.legend()
            plt.grid(True)

            # oZ-axis
            plt.subplot(4, 1, 3)
            plt.plot(df["time"], df["cmd_oz"], label="Cmd oZ")
            plt.plot(df["time"], df["act_oz"], label="Act oZ")
            plt.ylabel("Orientation oZ")
            plt.title("Command vs Actual - oZ Axis")
            plt.legend()
            plt.grid(True)

            # oW-axis
            plt.subplot(4, 1, 4)
            plt.plot(df["time"], df["cmd_ow"], label="Cmd oW")
            plt.plot(df["time"], df["act_ow"], label="Act oW")
            plt.ylabel("Orientation oW")
            plt.title("Command vs Actual - oW Axis")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            filename = "franka_comparison_ow.png"
            plt.savefig(filename)
            print(f"Saved 4-axis comparison plot to {filename}")


    except rospy.ROSInterruptException:
        pass
