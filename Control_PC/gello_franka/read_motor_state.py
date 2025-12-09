#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState

# Dynamixel SDK
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

def clamp_arr(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

class DxlToFrankaTeleop:
    def __init__(self):
        rospy.init_node("dxl_to_franka_teleop", anonymous=False)

        # ======== 基本通訊設定（可在 rosparam/YAML 覆寫）========
        self.device_name = rospy.get_param("~device_name", "/dev/ttyUSB0")
        self.baudrate = rospy.get_param("~baudrate", 57600)                # 你說是 57600
        self.protocol_version = rospy.get_param("~protocol_version", 2.0)  # 1.0 or 2.0

        # 控制表位置暫存器（X系列/Prot 2.0 預設 132，AX/MX Prot1.0 常見 36）
        self.addr_present_position = rospy.get_param("~present_position_addr", 132)
        self.present_position_len  = rospy.get_param("~present_position_len", 4)  # 2 or 4 bytes

        # 7 顆馬達 ID 與解析度
        self.dxl_ids = rospy.get_param("~dxl_ids", [1,2,3,4,5,6,7])
        assert len(self.dxl_ids) == 7, "dxl_ids 需為 7 筆"

        self.ticks_per_rev = np.array(rospy.get_param("~ticks_per_rev", [4096]*7), dtype=float)
        self.center_ticks  = np.array(rospy.get_param("~center_ticks",  [2048]*7), dtype=float)

        # 方向(+1/-1)、比例(scale)、角度偏移(rad)（校正手感與零位）
        self.signs       = np.array(rospy.get_param("~signs",       [ 1, -1, 1, 1, 1, -1, 1 ]), dtype=float)
        self.scales      = np.array(rospy.get_param("~scales",      [1.0]*7), dtype=float)
        self.offsets_rad = np.array(rospy.get_param("~offsets_rad", [0.0]*7), dtype=float)

        # 小手臂 index -> Panda index 映射（長度 7、元素為 0..6 的排列）
        # 例如 [0,2,1,3,4,5,6] 代表小手臂第2軸餵給 Panda 第1軸
        self.mapping = np.array(rospy.get_param("~mapping", [0,1,2,3,4,5,6]), dtype=int)

        # Panda 關節軟限制（rad）
        self.panda_lower = np.array(rospy.get_param("~panda_lower",
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]), dtype=float)
        self.panda_upper = np.array(rospy.get_param("~panda_upper",
            [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]), dtype=float)

        # 輸出 Topic 與頻率、濾波/速度限制
        # self.cmd_topic = rospy.get_param("~franka_cmd_topic",
        #     "/franka/joint_group_position_controller/command")
        self.cmd_topic = rospy.get_param("~franka_cmd_topic",
            "/position_joint_trajectory_controller/command")
        self.rate_hz = float(rospy.get_param("~rate_hz", 50))
        self.alpha   = float(rospy.get_param("~lowpass_alpha", 0.35))     # 指數濾波係數 0~1（越小越平滑）
        self.max_vel = np.array(rospy.get_param("~max_vel_rad_s", [1.0]*7), dtype=float)
        self.deadband = float(rospy.get_param("~deadband_rad", 0.0))

        # 使能/急停
        self.enabled = bool(rospy.get_param("~start_enabled", True))
        self.estop_latched = False

        # Publisher / Subscriber
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Float64MultiArray, queue_size=10)
        self.pub_small_arm_js = rospy.Publisher("~small_arm/joint_states", JointState, queue_size=10)

        rospy.Subscriber("~enable", Bool, self.cb_enable)
        rospy.Subscriber("~estop",  Bool, self.cb_estop)

        # Dynamixel 連線
        self.port = PortHandler(self.device_name)
        self.packet = PacketHandler(self.protocol_version)

        if not self.port.openPort():
            rospy.logfatal("開啟串口失敗: %s", self.device_name)
            raise RuntimeError("Open port failed")
        if not self.port.setBaudRate(self.baudrate):
            rospy.logfatal("設定鮑率失敗: %d", self.baudrate)
            raise RuntimeError("Set baudrate failed")

        rospy.loginfo("Dynamixel 連線成功：%s @ %d, Prot=%.1f",
                      self.device_name, self.baudrate, self.protocol_version)

        self.prev_cmd = np.zeros(7, dtype=float)
        self.spin_loop()

    # --- Callbacks ---
    def cb_enable(self, msg: Bool):
        if not self.estop_latched:
            self.enabled = bool(msg.data)
            rospy.logwarn("Teleop enable = %s", self.enabled)

    def cb_estop(self, msg: Bool):
        if msg.data:
            self.estop_latched = True
            self.enabled = False
            rospy.logerr("E-STOP 觸發（鎖定），需重啟節點才會清除。")

    # --- DXL I/O ---
    def read_pos_ticks(self, dxl_id):
        if self.present_position_len == 4:
            val, res, err = self.packet.read4ByteTxRx(self.port, dxl_id, self.addr_present_position)
        elif self.present_position_len == 2:
            val, res, err = self.packet.read2ByteTxRx(self.port, dxl_id, self.addr_present_position)
        else:
            raise ValueError("present_position_len 只能為 2 或 4")
        if res != COMM_SUCCESS or err != 0:
            return None
        # 某些機型可能回傳 32-bit 無號整數（例如你之前看到的 4294967286）
        # 正規化到合理範圍（0..ticks_per_rev-1）附近
        return int(val)

    def ticks_to_rad(self, ticks_arr):
        # rad_base = ((ticks - center) / ticks_per_rev) * 2π
        base = (ticks_arr - self.center_ticks) / self.ticks_per_rev * (2.0 * math.pi)
        rad = base * self.signs * self.scales + self.offsets_rad
        return rad

    def apply_deadband(self, prev, curr, db):
        out = curr.copy()
        small = np.abs(curr - prev) < db
        out[small] = prev[small]
        return out

    def rate_limit(self, prev, curr, dt):
        max_step = self.max_vel * dt
        delta = np.clip(curr - prev, -max_step, +max_step)
        return prev + delta

    def lowpass(self, prev, curr):
        return self.alpha * curr + (1.0 - self.alpha) * prev

    def map_small_to_panda(self, small_rad):
        # small[i] -> panda[mapping[i]]
        panda = np.zeros(7, dtype=float)
        panda[self.mapping] = small_rad
        return clamp_arr(panda, self.panda_lower, self.panda_upper)

    def publish_small_arm_js(self, small_rad):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = [f"dxl_{i}" for i in self.dxl_ids]
        js.position = small_rad.tolist()
        self.pub_small_arm_js.publish(js)

    def spin_loop(self):
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz

        while not rospy.is_shutdown():
            # 讀取 7 軸 ticks
            ticks = []
            for i, dxl_id in enumerate(self.dxl_ids):
                v = self.read_pos_ticks(dxl_id)
                if v is None:
                    # 讀不到就用 center 當暫時值，避免跳動
                    v = int(self.center_ticks[i])
                    rospy.logwarn_throttle(2.0, "讀取 ID=%d 失敗，使用 center_ticks[%d]=%d", dxl_id, i, v)
                # 將 32-bit 異常值（例如 4294967xxx）折回合理範圍
                # 假設有效範圍約在 [0, ticks_per_rev) 附近
                mod = int(self.ticks_per_rev[i])
                if mod > 0:
                    v = v % mod
                ticks.append(v)
            ticks = np.array(ticks, dtype=float)

            # 轉小手臂關節角（rad）
            small_rad = self.ticks_to_rad(ticks)
            self.publish_small_arm_js(small_rad)

            # 映射到 Panda 順序 + 限幅
            panda_cmd = self.map_small_to_panda(small_rad)

            # 死區、速度限制、低通濾波
            panda_cmd = self.apply_deadband(self.prev_cmd, panda_cmd, self.deadband)
            panda_cmd = self.rate_limit(self.prev_cmd, panda_cmd, dt)
            panda_cmd = self.lowpass(self.prev_cmd, panda_cmd)

            # 發布命令
            if self.enabled and not self.estop_latched:
                self.pub_cmd.publish(Float64MultiArray(data=panda_cmd.tolist()))
                self.prev_cmd = panda_cmd  # 只在送出時更新，停用時保持

            rate.sleep()

        try:
            self.port.closePort()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        DxlToFrankaTeleop()
    except rospy.ROSInterruptException:
        pass
