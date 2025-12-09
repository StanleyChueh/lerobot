#!/usr/bin/env python3
import rospy
from dynamixel_sdk import *
import numpy as np

# ======== Dynamixel 設定 ========
DXL_IDS = [1,2,3,4,5,6,7]
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0
ADDR_PRESENT_POSITION = 132
DXL_RESOLUTION = 4096

def read_dxl_positions(packetHandler, portHandler):
    positions = []
    for dxl_id in DXL_IDS:
        val, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        val = val % DXL_RESOLUTION
        positions.append(val)
    return np.array(positions, dtype=int)

def main():
    rospy.init_node('measure_dxl_range_interactive', anonymous=True)
    rospy.loginfo("互動式量測 Dynamixel 最小/最大值")

    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    if not portHandler.openPort():
        rospy.logerr("無法開啟串口")
        return
    if not portHandler.setBaudRate(BAUDRATE):
        rospy.logerr("無法設定鮑率")
        return

    min_vals = []
    max_vals = []

    try:
        for i, dxl_id in enumerate(DXL_IDS):
            input(f"請把第 {i+1} 顆馬達（ID {dxl_id}）移到最小位置，按 Enter 繼續...")
            pos = read_dxl_positions(packetHandler, portHandler)[i]
            min_vals.append(pos)
            print(f"讀取到最小值: {pos}")

            input(f"請把第 {i+1} 顆馬達（ID {dxl_id}）移到最大位置，按 Enter 繼續...")
            pos = read_dxl_positions(packetHandler, portHandler)[i]
            max_vals.append(pos)
            print(f"讀取到最大值: {pos}")

    except KeyboardInterrupt:
        pass

    rospy.loginfo("測量完成！")
    print("\nDynamixel 最小值 DXL_MIN = [{}]".format(", ".join(map(str, min_vals))))
    print("Dynamixel 最大值 DXL_MAX = [{}]".format(", ".join(map(str, max_vals))))

    try:
        portHandler.closePort()
    except:
        pass

if __name__ == "__main__":
    main()
