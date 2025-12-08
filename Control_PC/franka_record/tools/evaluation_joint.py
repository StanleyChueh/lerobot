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

from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, WrenchStamped

from franka_gripper.msg import MoveAction, MoveGoal, GraspGoal, GraspAction
from franka_gripper.msg import GraspEpsilon, GraspResult
from actionlib_msgs.msg import GoalStatus
import matplotlib.pyplot as plt

shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)


class FrankaHardcodedPose:
    def __init__(self):
        rospy.init_node("franka_hardcoded_pose", anonymous=False)

        # --- Publishers / Action clients
        self.joint_pub = rospy.Publisher(
            "/position_joint_trajectory_controller/command",
            JointTrajectory,
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
        #self.pose_sub = rospy.Subscriber("/robot0_eef_pose", PoseStamped, self.pose_cb)
        self.joint_sub = rospy.Subscriber(
            "/joint_states", JointState, self.joint_cb
        )
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

        # --- Initialize gripper opening
        self.send_gripper_command(width=self.gripper)
        rospy.loginfo("Publishing hard-coded pose at ~30Hz.")

    # ----------------- Callbacks -----------------

    def gripper_effort_callback(self, msg: JointState):
        if "panda_finger_joint1" in msg.name:
            idx = msg.name.index("panda_finger_joint1")
            self.gripper_effort = msg.effort[idx]

    def joint_cb(self, msg: JointState):
        joint_positions = []
        for joint_name in [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                joint_positions.append(msg.position[idx])
            else:
                rospy.logwarn(f"Joint {joint_name} not found in JointState message.")
                joint_positions.append(0.0)  # default value if joint not found
        self.pose = joint_positions  # list of 7 joint positions

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

    def data_client(self):
        # 1) send state
        state = self.pose + [self.gripper_now]
        self.send_data(self.client_socket, "list", json.dumps(state).encode())

        # 2) wait images
        wait_start = time.time()

        def imgs_ready():
            return (
                self.eye_image_front is not None
                and self.eye_image_front.size > 0
                and self.eye_image_rear is not None
                and self.eye_image_rear.size > 0
                and self.agent_image is not None
                and self.agent_image.size > 0
            )

        while not imgs_ready() and not rospy.is_shutdown():
            if time.time() - wait_start > 5.0:
                if self.eye_image_front is None or self.eye_image_front.size == 0:
                    rospy.logwarn("等待 eye_front 圖像逾時")
                if self.eye_image_rear is None or self.eye_image_rear.size == 0:
                    rospy.logwarn("等待 eye_rear 圖像逾時")
                if self.agent_image is None or self.agent_image.size == 0:
                    rospy.logwarn("等待 agentview 圖像逾時")
                break
            rospy.sleep(0.05)

        if not imgs_ready():
            rospy.logwarn("影像資料不完整，跳過本次推論")
            return

        # Make local copies to avoid races during encode
        eye_front = self.eye_image_front
        eye_rear = self.eye_image_rear
        agent = self.agent_image

        # 3) encode & send
        ok1, img1_bytes = cv2.imencode(".jpg", eye_front)
        if not ok1:
            rospy.logerr("imencode eye_front_image 失敗")
            return
        self.send_data(self.client_socket, "img1", img1_bytes.tobytes())

        ok2, img2_bytes = cv2.imencode(".jpg", agent)
        if not ok2:
            rospy.logerr("imencode agent_image 失敗")
            return
        self.send_data(self.client_socket, "img2", img2_bytes.tobytes())

        ok3, img3_bytes = cv2.imencode(".jpg", eye_rear)
        if not ok3:
            rospy.logerr("imencode eye_rear_image 失敗")
            return
        self.send_data(self.client_socket, "img3", img3_bytes.tobytes())

        # 4) receive action
        data_type, data = self.recv_data(self.client_socket)
        if data_type == "list":
            try:
                self.action = json.loads(data.decode())
                rospy.loginfo(f"收到 server 回傳: {self.action}")
            except Exception as e:
                rospy.logerr(f"解析 server 回傳失敗: {e}")
                   

    def evaluate(self):
        self.data_client()

        # 控制的單一關節（例如第3關節）
        target_joint_index = 6
        joint_name = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        # ✅ 起始角度設定為當前實際角度
        if not hasattr(self, "pose") or self.pose is None:
            rospy.logwarn("尚未接收到 joint state，使用 0.0 作為起始角度")
            start_pos = 0.0
        else:
            start_pos = self.pose[target_joint_index]

        # ✅ 設定目標角度（例如比當前角度高 0.5 rad）
        end_pos = start_pos + 0.5

        # 移動參數
        num_steps = 100
        step_time = 0.05
        delta = (end_pos - start_pos) / num_steps

        rospy.loginfo(f"開始控制 {joint_name[target_joint_index]} 緩慢移動")
        rospy.loginfo(f"起始角度: {start_pos:.3f}, 目標角度: {end_pos:.3f}")

        base_pose = self.pose.copy() if hasattr(self, "pose") else [0.0] * 7
        command_list = []
        action_list = []

        for i in range(num_steps + 1):
            # 計算當前目標角度
            if i < num_steps-20:
                current_pos = start_pos - delta * i
            base_pose[target_joint_index] = current_pos

            # 發送 joint 指令
            joint_msg = JointTrajectory()
            joint_msg.joint_names = joint_name
            point = JointTrajectoryPoint()
            point.positions = base_pose.copy()
            point.time_from_start = rospy.Duration(step_time)
            joint_msg.points = [point]
            self.joint_pub.publish(joint_msg)

            rospy.loginfo(
                f"Step {i+1}/{num_steps}: {joint_name[target_joint_index]} = {current_pos:.3f}"
            )

            # 紀錄 command / action
            command_list.append(current_pos)
            if self.action is not None:
                action_list.append(self.pose[target_joint_index])
            else:
                action_list.append(None)

            rospy.sleep(step_time)

        start_pos = current_pos
        for i in range(num_steps + 1):
            # 計算當前目標角度
            if i < num_steps-20:
                current_pos = start_pos + delta * i
            base_pose[target_joint_index] = current_pos

            # 發送 joint 指令
            joint_msg = JointTrajectory()
            joint_msg.joint_names = joint_name
            point = JointTrajectoryPoint()
            point.positions = base_pose.copy()
            point.time_from_start = rospy.Duration(step_time)
            joint_msg.points = [point]
            self.joint_pub.publish(joint_msg)

            rospy.loginfo(
                f"Step {i+1}/{num_steps}: {joint_name[target_joint_index]} = {current_pos:.3f}"
            )

            # 紀錄 command / action
            command_list.append(current_pos)
            if self.action is not None:
                action_list.append(self.pose[target_joint_index])
            else:
                action_list.append(None)

            rospy.sleep(step_time)

        rospy.loginfo("單關節緩慢移動完成，開始繪圖")

        # 處理 None 值，避免繪圖錯誤
        valid_idx = [i for i, a in enumerate(action_list) if a is not None]
        if len(valid_idx) > 0:
            print("有效的 self.action 資料，開始繪圖")
            action_list = [action_list[i] for i in valid_idx]
            command_list = [command_list[i] for i in valid_idx]
            x = list(range(len(command_list)))

            plt.figure(figsize=(8, 4))
            plt.plot(x, command_list, linewidth=2)
            plt.plot(x, action_list, linestyle="--")
            plt.xlabel("Step index")
            plt.ylabel("Joint angle (radians)")
            plt.title(f"{joint_name[target_joint_index]} Command vs Action")
            plt.legend(["Command", "Real Action"])
            plt.grid(True)
            plt.tight_layout()
            save_dir = "/home/csl/franka_record/plots"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"joint_plot_{timestamp}.png")

            plt.savefig(save_path)
            plt.close()
            rospy.loginfo(f"✅ 圖表已儲存至：{save_path}")
        else:
            rospy.logwarn("無有效的 self.action 資料，跳過繪圖。")



    # def evaluate(self):
    #     self.data_client()
    #     if self.action is not None:
    #         # joint
    #         joint_msg = JointTrajectory()
    #         joint_msg.joint_names = [
    #             "panda_joint1",
    #             "panda_joint2",
    #             "panda_joint3",
    #             "panda_joint4",
    #             "panda_joint5",
    #             "panda_joint6",
    #             "panda_joint7",
    #         ]
    #         point = JointTrajectoryPoint()
    #         point.positions = self.action[0:7]
    #         print(f"Set joint 7 to: {point.positions[6]}")
    #         point.time_from_start = rospy.Duration(0.1)
    #         joint_msg.points = [point]
    #         self.joint_pub.publish(joint_msg)

    #         # gripper
    #         self.gripper = float(self.action[7])
    #         try:
    #             if self.gripper < 0.055 and self.gripper_state != "closed":
    #                 rospy.loginfo(f"gripper state: close, {self.gripper}")
    #                 goal = GraspGoal()
    #                 goal.width = 0.04
    #                 goal.speed = 0.05
    #                 goal.force = 80.0
    #                 goal.epsilon = GraspEpsilon(inner=0.02, outer=0.02)
    #                 self.grasp_client.send_goal(goal)
    #                 self.gripper_state = "closed"

    #             elif self.gripper >= 0.055 and self.gripper_state != "open":
    #                 rospy.loginfo(f"gripper state: open, {self.gripper}")
    #                 move_goal = MoveGoal()
    #                 move_goal.width = 0.08
    #                 move_goal.speed = 0.05
    #                 self.move_client.send_goal(move_goal)
    #                 self.gripper_state = "open"
    #         except Exception as e:
    #             rospy.logerr(f"Exception sending grasp/move goal: {e}")

    def to_joint_trajectory(self, positions):
        jt = JointTrajectory()
        jt.joint_names = [f"panda_joint{i}" for i in range(1,8)]
        pt = JointTrajectoryPoint()
        pt.positions = [round(float(x),3) for x in positions]
        pt.time_from_start = rospy.Duration(0.5)
        jt.points.append(pt)
        return jt

    def initialize_franka_joint(self):
        # 發送一次現有位置，確保控制器持當前姿態
        for j in range(len(self.pose)):
            for i in range(25):
                time.sleep(0.05)
                self.pose[j] = self.pose[j]-0.0001*i
                self.joint_pub.publish(self.to_joint_trajectory(self.pose))
            for i in range(25):
                time.sleep(0.05)
                self.pose[j] = self.pose[j]+0.0001*i
                self.joint_pub.publish(self.to_joint_trajectory(self.pose))
            rospy.loginfo(f"已完成初始化：{j}")

if __name__ == "__main__":
    try:
        fk = FrankaHardcodedPose()
        rate = rospy.Rate(30)  # avoid busy loop
        fk.initialize_franka_joint()
        # while not rospy.is_shutdown() and not shutdown_requested:
        fk.evaluate()
        rate.sleep()
    except rospy.ROSInterruptException:
        pass
