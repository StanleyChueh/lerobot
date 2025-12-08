#!/usr/bin/env python3
import os
import sys
import time
import json
import socket
import struct
import signal
import argparse
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal, GraspGoal, GraspAction, GraspEpsilon

import pandas as pd
import matplotlib.pyplot as plt
import datetime
def ts(): return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

USE_TURBO = False
try:
    from turbojpeg import TurboJPEG
    _JPEG = TurboJPEG()
    USE_TURBO = True
except Exception:
    _JPEG = None
    USE_TURBO = False

shutdown_requested = False
def _sigint(_sig, _frm):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, _sigint)

def nsec():
    return time.monotonic_ns()

def now():
    return time.monotonic()

def sleep_until(t_deadline):
    # 精準節流：睡到目標時間點（避免累積漂移）
    rem = t_deadline - now()
    if rem > 0:
        time.sleep(rem)

def downscale(img, max_w):
    if max_w <= 0:
        return img
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = float(max_w) / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def encode_jpg(img_bgr, quality=65):
    if USE_TURBO:
        return _JPEG.encode(img_bgr, quality=quality)
    else:
        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return enc.tobytes()

class StableClient:
    def __init__(self,
                 server_ip, port,
                 target_hz=30.0,
                 jpeg_quality=65,
                 max_width=640, #320
                 img_stale_sec=0.2,
                 connect_timeout=5.0,
                 recv_timeout=5.0):
        self.server_ip = server_ip
        self.port = port
        self.target_hz = float(target_hz)
        self.period = 1.0 / self.target_hz if self.target_hz > 0 else 0.0
        self.jpeg_quality = int(jpeg_quality)
        self.max_width = int(max_width)
        self.img_stale_sec = float(img_stale_sec)
        self.connect_timeout = float(connect_timeout)
        self.recv_timeout = float(recv_timeout)

        # ROS
        rospy.init_node("franka_client_stable", anonymous=False)
        self.pose_pub = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose",
                                        PoseStamped, queue_size=10)
        self.gripper_move = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self.gripper_grasp = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        rospy.loginfo("Waiting for gripper action servers...")
        self.gripper_move.wait_for_server()
        self.gripper_grasp.wait_for_server()
        rospy.loginfo("Gripper action servers connected.")

        # Latest state
        self.pose = None
        self.gripper_now = 0.08   # 8cm
        self._img_front = None; self._ts_front = 0.0
        self._img_rear  = None; self._ts_rear  = 0.0
        self._img_agent = None; self._ts_agent = 0.0
        self.grip_state = "open"

        # Subscribers
        rospy.Subscriber("/robot0_eef_pose", PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/robot0_gripper_width", Float32, self._grip_cb, queue_size=1)
        rospy.Subscriber("/robot0_eye_in_hand_image_front", Image, self._front_cb, queue_size=1)
        rospy.Subscriber("/robot0_eye_in_hand_image_rear",  Image, self._rear_cb,  queue_size=1)
        rospy.Subscriber("/robot0_agentview_image",         Image, self._agent_cb, queue_size=1)

        # Socket
        self.sock = None
        self._connect()

        # Pre-build pose msg
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

    # ---------- ROS callbacks ----------
    def _pose_cb(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.pose = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

    def _grip_cb(self, msg: Float32):
        self.gripper_now = float(msg.data)

    def _decode_img(self, msg: Image):
        enc = msg.encoding.lower()
        # 支援 mono8 / bgr8 / rgb8 / bgra8 / rgba8 / 8UC1
        dt = None; ch = 0
        if enc in ("mono8", "8uc1"): dt="uint8"; ch=1
        elif enc in ("bgr8",): dt="uint8"; ch=3
        elif enc in ("rgb8",): dt="uint8"; ch=3
        elif enc in ("bgra8", "rgba8"): dt="uint8"; ch=4
        else:
            raise ValueError("Unsupported encoding: %s" % msg.encoding)
        arr = np.frombuffer(msg.data, dtype=np.dtype(dt))
        if ch == 1:
            img = arr.reshape((msg.height, msg.width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = arr.reshape((msg.height, msg.width, ch))
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif enc == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif enc == "bgra8":
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # bgr8 不用轉
        return img

    def _front_cb(self, msg: Image):
        try:
            self._img_front = self._decode_img(msg)
            self._ts_front = now()
        except Exception as e:
            rospy.logwarn(f"front decode error: {e}")

    def _rear_cb(self, msg: Image):
        try:
            self._img_rear = self._decode_img(msg)
            self._ts_rear = now()
        except Exception as e:
            rospy.logwarn(f"rear decode error: {e}")

    def _agent_cb(self, msg: Image):
        try:
            self._img_agent = self._decode_img(msg)
            self._ts_agent = now()
        except Exception as e:
            rospy.logwarn(f"agent decode error: {e}")

    # ---------- Socket helpers ----------
    def _apply_sock_opts(self, s):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4*1024*1024)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4*1024*1024)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass
        s.settimeout(self.recv_timeout)

    def _connect(self):
        # 自動重連
        while not shutdown_requested and not rospy.is_shutdown():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._apply_sock_opts(s)
                s.settimeout(self.connect_timeout)
                rospy.loginfo(f"Connecting to {self.server_ip}:{self.port} ...")
                s.connect((self.server_ip, self.port))
                self.sock = s
                self._apply_sock_opts(self.sock)  # 連線後再設一次
                rospy.loginfo("Client connected.")
                return
            except Exception as e:
                rospy.logwarn(f"Connect failed: {e}, retry in 1s")
                time.sleep(1.0)

    def _send_one(self, typ: str, payload: bytes):
        # len(4, big-endian) + type(10, ascii padded) + payload
        typ_b = typ.ljust(10).encode('utf-8')
        header = struct.pack('>I', len(payload)) + typ_b
        self.sock.sendall(header + payload)

    def _recv_one(self):
        def recv_exact(n):
            buf = b''
            while len(buf) < n:
                chunk = self.sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            return buf
        raw_len = recv_exact(4)
        if raw_len is None:
            return None, None
        data_len = struct.unpack('>I', raw_len)[0]
        raw_type = recv_exact(10)
        if raw_type is None:
            return None, None
        typ = raw_type.decode('utf-8').strip()
        data = recv_exact(data_len)
        if data is None:
            return None, None
        return typ, data

    def _close(self):
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        self.sock = None

    # ---------- Control helpers ----------
    def _build_state(self):
        # 末端位姿 + gripper 當前寬度
        return self.pose + [self.gripper_now]

    def _have_fresh_images(self, t_now):
        # 三路影像都要在可接受新鮮度內
        return (self._img_front is not None and t_now - self._ts_front <= self.img_stale_sec and
                self._img_rear  is not None and t_now - self._ts_rear  <= self.img_stale_sec and
                self._img_agent is not None and t_now - self._ts_agent <= self.img_stale_sec)

    def _publish_action(self, action):
        # action: [x,y,z, qx,qy,qz,qw, gripper]
        pm = self.pose_msg
        pm.header.stamp = rospy.Time.now()
        pm.pose.position.x = float(action[0])
        pm.pose.position.y = float(action[1])
        pm.pose.position.z = float(action[2]) 
        pm.pose.orientation.x = float(action[3])
        pm.pose.orientation.y = float(action[4])
        pm.pose.orientation.z = float(action[5])
        pm.pose.orientation.w = float(action[6])
        self.pose_pub.publish(pm)

        # 簡單的開合邏輯（可依任務自行調整）
        g = float(action[7])
        try:
            if g < 0.04 and self.grip_state != "closed":
                goal = GraspGoal()
                goal.width = 0.04
                goal.speed = 0.1
                goal.force = 80.0
                goal.epsilon = GraspEpsilon(inner=0.02, outer=0.02)
                self.gripper_grasp.send_goal(goal)
                self.grip_state = "closed"
                time.sleep(0.5)
            elif g >= 0.055 and self.grip_state != "open":
                m = MoveGoal()
                m.width = 0.08
                m.speed = 0.1
                self.gripper_move.send_goal(m)
                self.grip_state = "open"
        except Exception as e:
            rospy.logwarn(f"gripper cmd error: {e}")

    # ---------- Main loop ----------
    def run(self):
        if self.period <= 0:
            rospy.logwarn("target_hz <= 0 : running as fast as possible (may be bursty).")
        next_deadline = now() + (self.period if self.period > 0 else 0.0)

        while not shutdown_requested and not rospy.is_shutdown():
            # a = input("Press Enter to send data to server...")
            t0 = now()

            # 確保連線有效
            if self.sock is None:
                self._connect()

            try:
                # 1) 取最新快照（不等待）
                t_now = now()
                state = self._build_state()
                fresh = self._have_fresh_images(t_now)
                if fresh:
                    img1 = downscale(self._img_front, self.max_width)
                    img2 = downscale(self._img_rear, self.max_width)
                    img3 = downscale(self._img_agent,  self.max_width)
                    b1 = encode_jpg(img1, self.jpeg_quality)
                    b2 = encode_jpg(img2, self.jpeg_quality)
                    b3 = encode_jpg(img3, self.jpeg_quality)

                    self._send_one("list", json.dumps(state).encode('utf-8'))
                    self._send_one("img1", b1)
                    self._send_one("img2", b2)
                    print(f"[{ts()}] Sending images...")
                    self._send_one("img3", b3)
                    print(f"[{ts()}] Waiting for action...")

                # 2B) If images are stale → ONLY send state (no images)
                else:
                    self._send_one("list", json.dumps(state).encode('utf-8'))

                # 3) Receive action from server (always expects one)
                typ, data = self._recv_one()
                print(f"[{ts()}] Received action.")
                if typ != "list":
                    raise RuntimeError(f"Unexpected type from server: {typ}")

                action = json.loads(data.decode('utf-8'))

                # 4) Publish action to the robot
                self._publish_action(action)
                print(f"Published action (fresh={fresh}): {action}")
            except (socket.timeout, ConnectionResetError, BrokenPipeError) as e:
                rospy.logwarn(f"socket error: {e}, reconnecting...")
                self._close()
                time.sleep(0.2)
                continue
            except Exception as e:
                rospy.logwarn(f"loop error: {e}")
                # 視需要是否重連；先不中斷節奏
                # self._close()

            # 5) 精準節流（保持穩定 Hz）
            if self.period > 0:
                next_deadline += self.period
                sleep_until(next_deadline)
            else:
                # 全速模式：避免 busy spin
                time.sleep(0)  # 讓出 GIL 一下

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", type=str, required=True,default="10.100.4.42")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--target-hz", type=float, default=30.0,
                    help="希望的評率（Hz），會進行精準節流；<=0 表示全速")
    ap.add_argument("--jpeg-quality", type=int, default=65)
    ap.add_argument("--max-width", type=int, default=640,
                    help="影像最大寬度（像素），0 表示不縮放")
    ap.add_argument("--img-stale-sec", type=float, default=0.2,
                    help="影像允許的最長延時（秒），超過就略過本輪")
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--recv-timeout", type=float, default=5.0)
    args = ap.parse_args()

    c = StableClient(server_ip=args.server_ip,
                     port=args.port,
                     target_hz=args.target_hz,
                     jpeg_quality=args.jpeg_quality,
                     max_width=args.max_width,
                     img_stale_sec=args.img_stale_sec,
                     connect_timeout=args.connect_timeout,
                     recv_timeout=args.recv_timeout)
    c.run()

if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--server-ip", type=str, default="10.100.4.42")
        ap.add_argument("--port", type=int, default=5001)
        ap.add_argument("--target-hz", type=float, default=30.0)
        ap.add_argument("--jpeg-quality", type=int, default=65)
        ap.add_argument("--max-width", type=int, default=320)
        ap.add_argument("--img-stale-sec", type=float, default=0.2)
        ap.add_argument("--connect-timeout", type=float, default=5.0)
        ap.add_argument("--recv-timeout", type=float, default=5.0)
        args = ap.parse_args()

        # ---- Run the client ----
        c = StableClient(server_ip=args.server_ip,
                         port=args.port,
                         target_hz=args.target_hz,
                         jpeg_quality=args.jpeg_quality,
                         max_width=args.max_width,
                         img_stale_sec=args.img_stale_sec,
                         connect_timeout=args.connect_timeout,
                         recv_timeout=args.recv_timeout)
        c.run()

        # ---- After shutdown, analyze logs if you saved them ----
        if os.path.exists("franka_logs.csv"):
            df = pd.read_csv("franka_logs.csv")

            # Compute position errors
            df["err_x"] = df["act_x"] - df["cmd_x"]
            df["err_y"] = df["act_y"] - df["cmd_y"]
            df["err_z"] = df["act_z"] - df["cmd_z"]

            mean_err = df[["err_x", "err_y", "err_z"]].abs().mean()
            max_err = df[["err_x", "err_y", "err_z"]].abs().max()

            print("\n=== Position Tracking Error Report ===")
            print(f"Mean Absolute Error X: {mean_err['err_x']:.6f} m")
            print(f"Mean Absolute Error Y: {mean_err['err_y']:.6f} m")
            print(f"Mean Absolute Error Z: {mean_err['err_z']:.6f} m")
            print(f"Max Absolute Error  X: {max_err['err_x']:.6f} m")
            print(f"Max Absolute Error  Y: {max_err['err_y']:.6f} m")
            print(f"Max Absolute Error  Z: {max_err['err_z']:.6f} m")
            print("======================================\n")

            # --- Plot command vs actual for X, Y, Z ---
            plt.figure(figsize=(10, 10))

            plt.subplot(3, 1, 1)
            plt.plot(df["time"], df["cmd_x"], label="Cmd X")
            plt.plot(df["time"], df["act_x"], label="Act X")
            plt.ylabel("Position X (m)")
            plt.title("Command vs Actual - X Axis")
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(df["time"], df["cmd_y"], label="Cmd Y")
            plt.plot(df["time"], df["act_y"], label="Act Y")
            plt.ylabel("Position Y (m)")
            plt.title("Command vs Actual - Y Axis")
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(df["time"], df["cmd_z"], label="Cmd Z")
            plt.plot(df["time"], df["act_z"], label="Act Z")
            plt.xlabel("Time (s)")
            plt.ylabel("Position Z (m)")
            plt.title("Command vs Actual - Z Axis")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("franka_comparison_xyz.png")
            print("Saved 3-axis comparison plot to franka_comparison_xyz.png")
        else:
            print("No franka_logs.csv found — nothing to plot.")

    except rospy.ROSInterruptException:
        pass