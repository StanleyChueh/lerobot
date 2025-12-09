import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import JointTrajectoryControllerState
from dynamixel_sdk import PortHandler, PacketHandler, GroupBulkRead, GroupSyncWrite, DXL_LOBYTE, DXL_LOWORD, DXL_HIBYTE, DXL_HIWORD
from franka_msgs.msg import FrankaState
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, HomingAction, HomingGoal
from controller_manager_msgs.srv import SwitchController
import numpy as np
import math
import argparse
import yaml
from pathlib import Path
import time
import actionlib
from threading import Thread

# ======== Dynamixel 設定 ======== Gripper range(3269 -> 2695)
DXL_IDS = [1,2,3,4,5,6,7,8]
prev_ticks = [None] * len(DXL_IDS)
BAUDRATE = 2000000
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0
ADDR_PRESENT_POSITION = 132
PRESENT_POS_LEN = 4

PANDA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=float)
PANDA_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=float)

def clamp_cmd(cmd):
    return np.minimum(np.maximum(cmd, PANDA_LOWER), PANDA_UPPER)

# ======== 角度轉 rad ========
DXL_RESOLUTION = 4096
DXL_CENTER = 2048
def dxl_ticks_to_rad(ticks):
    return (ticks - DXL_CENTER) * 2.0 * math.pi / DXL_RESOLUTION

# ---- DXL control table (X/XL series) ----
ADDR_OPERATING_MODE = 11
OPERATING_MODE_CURRENT_POSITION = 5    # current-based position mode
ADDR_TORQUE_ENABLE = 64
TORQUE_ENABLE = 1
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_GOAL_CURRENT = 102                # optional (2 bytes) if supported by your model
ADDR_PROFILE_ACCELERATION = 108        # optional (4 bytes)
ADDR_PROFILE_VELOCITY = 112            # optional (4 bytes)

DXL_GRIPPER_IDX = 7  # <-- index in DXL_IDS for your gripper; VERIFY it's the right one!

# ======== 映射設定（Franka idx -> DXL idx）========
MAPPING   = [0, 1, 2, 3, 4, 5, 6, 7]
DIRECTION = [-1, 1,  -1, -1,  -1, 1,  -1]
FRANKA_ZERO = [0.0]*7

# ======== 偏移檔案 ========
CFG_DIR  = Path.home() / ".config" / "dxl2franka"
CFG_FILE = CFG_DIR / "gello_offsets.yaml"

def disable_gripper_torque():
    grip_id = DXL_IDS[DXL_GRIPPER_IDX]
    ADDR_TORQUE_ENABLE = 64
    TORQUE_DISABLE = 0
    packet.write1ByteTxRx(port, grip_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    rospy.loginfo(f"Torque disabled on DXL {grip_id}")

def save_offsets(offsets):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG_FILE, "w") as f:
        yaml.safe_dump({"HOMING_OFFSET": [float(x) for x in offsets]}, f)

def load_offsets(default=None):
    if CFG_FILE.exists():
        with open(CFG_FILE, "r") as f:
            data = yaml.safe_load(f)
        return data.get("HOMING_OFFSET", default)
    return default

# ---------- Franka 狀態 ----------
def read_franka_from_joint_states(topic, timeout=0.2):
    msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
    name_to_idx = {n:i for i, n in enumerate(msg.name)}
    out = []
    for j in [f"panda_joint{i}" for i in range(1,8)]:
        out.append(msg.position[name_to_idx[j]])
    return np.array(out, dtype=float)

def read_franka_from_controller_state(topic, timeout=0.2):
    st = rospy.wait_for_message(topic, JointTrajectoryControllerState, timeout=timeout)
    return np.array(st.actual.positions[:7], dtype=float)

def read_franka_positions(topic, timeout=0.2):
    if topic.endswith("/state"):
        return read_franka_from_controller_state(topic, timeout)
    else:
        return read_franka_from_joint_states(topic, timeout)

# ---------- 讀 DXL (Bulk Read) ----------

def unwrap_ticks(prev_tick, curr_tick, resolution=4096):
    delta = curr_tick - prev_tick
    if delta > resolution // 2:
        curr_tick -= resolution
    elif delta < -resolution // 2:
        curr_tick += resolution
    return curr_tick

def read_all_dxl_positions_rad(groupBulkRead):
    vals = []
    global prev_ticks
    groupBulkRead.txRxPacket()
    for i, dxl_id in enumerate(DXL_IDS):
        if groupBulkRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, PRESENT_POS_LEN):
            pos = groupBulkRead.getData(dxl_id, ADDR_PRESENT_POSITION, PRESENT_POS_LEN)
            pos = int(pos) % DXL_RESOLUTION
            if prev_ticks[i] is not None:
                pos = unwrap_ticks(prev_ticks[i], pos)
            prev_ticks[i] = pos
            vals.append(dxl_ticks_to_rad(pos))
        else:
            rospy.logwarn(f"DXL ID {dxl_id} 無法讀取位置")
            vals.append(0.0)
    return np.array(vals, dtype=float)

# ---------- 發送 JointTrajectory ----------
def to_joint_trajectory(positions):
    jt = JointTrajectory()
    jt.joint_names = [f"panda_joint{i}" for i in range(1,8)]
    pt = JointTrajectoryPoint()
    pt.positions = [round(float(x),3) for x in positions]
    pt.time_from_start = rospy.Duration(0.5)
    jt.points.append(pt)
    return jt

def limit_protection(initial_jointState, dxl_rad):
    delta = initial_jointState[:7] - dxl_rad[:7]
    for i in range(len(delta)):
        if abs(delta[i])> np.pi/2:
            delta[i] = delta[i]/abs(delta[i])
    return delta


def open_gripper(grasp_client):
    # === Do not use MoveGoal ===#
    # Operation success if d is in this range->
    # width - epsilon.inner < d < width + epsilon.outer (d is the actual grasp width)

    goal = GraspGoal()
    goal.width = 0.079              # Fully close 0.079
    goal.speed = 0.1                   # Moderate closing speed
    goal.force = 80.0                  # Strong grip
    goal.epsilon.inner = 0.079          # Accept small final gap
    goal.epsilon.outer = 0.0          # Accept small overshoot

    grasp_client.send_goal(goal)

    rospy.loginfo(f"[Gripper] Sending open goal...{goal.width}")
    
    #self.grasp_client.send_goal(goal)

    finished = grasp_client.wait_for_result(rospy.Duration(5.0))
    if not finished:
        rospy.logwarn("[Gripper] Grasp action timeout.")
        return False

    result = grasp_client.get_result()
    if result:
        rospy.loginfo(f"[Gripper] Open success: {result.success}")
        return result.success
    else:
        rospy.logwarn("[Gripper] No result returned.")
        return False

def clamp_gripper(grasp_client):
    # === Do not use MoveGoal ===#
    # Operation success if d is in this range->
    # width - epsilon.inner < d < width + epsilon.outer (d is the actual grasp width)

    goal = GraspGoal()
    goal.width = 0.0                   # Fully close
    goal.speed = 0.1                   # Moderate closing speed
    goal.force = 80.0                  # Strong grip
    goal.epsilon.inner = 0.04          # Accept small final gap
    goal.epsilon.outer = 0.08          # Accept small overshoot

    grasp_client.send_goal(goal)

    rospy.loginfo(f"[Gripper] Sending grasp goal...{goal.width}")
    #self.grasp_client.send_goal(goal)

    finished = grasp_client.wait_for_result(rospy.Duration(5.0))
    if not finished:
        rospy.logwarn("[Gripper] Grasp action timeout.")
        return False

    result = grasp_client.get_result()
    if result:
        rospy.loginfo(f"[Gripper] Clamp success: {result.success}")
        return result.success
    else:
        rospy.logwarn("[Gripper] No result returned.")
        return False
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=50.0, help="頻率 Hz")
    parser.add_argument("--joint-states-topic", type=str,
                        default="/joint_states",
                        help="Franka 狀態來源（/franka_state_controller/joint_states 或 /.../state")
    parser.add_argument("--control-joint", type=int, default=7,
                        help="只控制哪一軸1~7預設 7")
    
    args = parser.parse_args()

    ctrl_idx = int(np.clip(args.control_joint, 1, 7)) - 1  # 0-based

    rospy.init_node('dxl_to_franka_only_j7', anonymous=True)
    pub = rospy.Publisher('/position_joint_trajectory_controller/command',
                          JointTrajectory, queue_size=10)
    
    grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    last_state = None

    # DXL
    port = PortHandler(DEVICENAME)
    packet = PacketHandler(PROTOCOL_VERSION)
    if not port.openPort():
        rospy.logerr("無法開啟 Dynamixel Port")
        return
    if not port.setBaudRate(BAUDRATE):
        rospy.logerr("無法設定鮑率")
        return

    grip_id = DXL_IDS[DXL_GRIPPER_IDX]

    # Put gripper in current-based position mode (LeRobot uses this)
    packet.write1ByteTxRx(port, grip_id, ADDR_OPERATING_MODE, OPERATING_MODE_CURRENT_POSITION)

    # (Optional) make the return gentle and safe
    packet.write2ByteTxRx(port, grip_id, ADDR_GOAL_CURRENT, 100)     # tune: ~0.5A on many XL/XM 200
    packet.write4ByteTxRx(port, grip_id, ADDR_PROFILE_VELOCITY, 80)   # tune to taste
    packet.write4ByteTxRx(port, grip_id, ADDR_PROFILE_ACCELERATION, 50)

    # Torque on
    packet.write1ByteTxRx(port, grip_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    # 建立 BulkRead group
    groupBulkRead = GroupBulkRead(port, packet)
    for dxl_id in DXL_IDS:
        groupBulkRead.addParam(dxl_id, ADDR_PRESENT_POSITION, PRESENT_POS_LEN)

    # 載入 offset（若不存在就用你剛校準的那組）
    HOMING_OFFSET = load_offsets(default=None)
    if HOMING_OFFSET is None:
        HOMING_OFFSET = [0.3343503912442118, -2.91, -0.03493154589597779,
                         -1.5142587100028266, 0.13616160753046874, 0.0,
                          1.48]
    HOMING_OFFSET = np.array(HOMING_OFFSET, dtype=float)

    rate = rospy.Rate(args.rate)
    last_actual = np.zeros(7, dtype=float)

    # rospy.loginfo("只控制 panda_joint%d 其餘關節保持當前位置。", ctrl_idx+1)
    d_idx = MAPPING[ctrl_idx]

    initial_jointState = read_all_dxl_positions_rad(groupBulkRead)

    def dxl_rad_to_ticks(rad):
        return int(rad * DXL_RESOLUTION / (2.0 * math.pi) + DXL_CENTER) % DXL_RESOLUTION

    initial_ticks_gripper = dxl_rad_to_ticks(initial_jointState[DXL_GRIPPER_IDX])

    # Set that initial position as the gripper’s goal (LeRobot holds a fixed goal)
    groupSyncWrite = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    param_goal = [
        DXL_LOBYTE(DXL_LOWORD(initial_ticks_gripper)),
        DXL_HIBYTE(DXL_LOWORD(initial_ticks_gripper)),
        DXL_LOBYTE(DXL_HIWORD(initial_ticks_gripper)),
        DXL_HIBYTE(DXL_HIWORD(initial_ticks_gripper)),
    ]
    groupSyncWrite.addParam(grip_id, param_goal)
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

    #print("initial position of dxl:",initial_jointState)

    try:
        franka_actual = read_franka_positions(args.joint_states_topic, timeout=1.0)
    except Exception:
        rospy.logerr("無法讀取 Franka 目前姿態，無法初始化位置")
        return

    # 發送一次現有位置，確保控制器持當前姿態
    for j in range(len(franka_actual)):
        for i in range(25):
            time.sleep(0.05)
            franka_actual[j] = franka_actual[j]-0.0001*i
            pub.publish(to_joint_trajectory(franka_actual))
        for i in range(25):
            time.sleep(0.05)
            franka_actual[j] = franka_actual[j]+0.0001*i
            pub.publish(to_joint_trajectory(franka_actual))
        rospy.loginfo(f"已完成初始化：{j}")

    # rospy.loginfo("已發送當前關節姿態作為初始指令: %s", np.round(franka_actual, 3))
    rospy.loginfo("完成初始化")
    rospy.sleep(0.5)
    HOMING_OFFSET = read_franka_positions(args.joint_states_topic, timeout=0.1)
    #print("HOMING OFFSET:",HOMING_OFFSET)

    grasp_client.wait_for_server()
    move_client.wait_for_server()
    last_j_target = HOMING_OFFSET
    last_sent_width = None
    while not rospy.is_shutdown():
        # 1) 讀 Franka 目前實際角度（做保持）
        try:
            franka_actual = read_franka_positions(args.joint_states_topic, timeout=0.1)
            last_actual = franka_actual
        except Exception:
            franka_actual = last_actual

        cmd = franka_actual.copy()

        # 2) 讀 DXL，換算成對應 Franka 軸的目標
        dxl_rad = read_all_dxl_positions_rad(groupBulkRead)
        #rospy.loginfo(f"dxl: {dxl_rad}")

        j_target = limit_protection(initial_jointState, dxl_rad) * (DIRECTION) +  HOMING_OFFSET + FRANKA_ZERO
        #j_target = 0.5*(j_target-last_j_target) + last_j_target
        # j_target[1:7] = HOMING_OFFSET[1:7]

        cmd[:7] = j_target[:7]

        # 控制 gripper
        if last_sent_width is None or abs(last_sent_width - width) > 0.001:
            move_goal = MoveGoal()
            grasp_goal = GraspGoal()
            width = ((((dxl_rad[-1] / 3.14 * 180) / 360) * 4095) - 640) / (1240-640) * 0.08
            if width <= 0.04:
                if last_state != "clamp":   # only send when switching
                    Thread(target=clamp_gripper, args=(grasp_client,)).start()
                    # clamp_gripper(grasp_client)
                    last_state = "clamp"
            else:
                if last_state != "open":   # only send when switching
                    Thread(target=open_gripper, args=(grasp_client,)).start()
                    #open_gripper(grasp_client)
                    last_state = "open"
        cmd = clamp_cmd(cmd)
        pub.publish(to_joint_trajectory(cmd))
        rate.sleep()
    rospy.on_shutdown(disable_gripper_torque)
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
