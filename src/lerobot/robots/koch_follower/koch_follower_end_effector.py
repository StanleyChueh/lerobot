# import logging
# import time
# from typing import Any, Dict

# import numpy as np

# from lerobot.cameras import make_cameras_from_configs
# from lerobot.errors import DeviceNotConnectedError
# from lerobot.model.kinematics import RobotKinematics
# from lerobot.motors import Motor, MotorNormMode
# from lerobot.motors.feetech import FeetechMotorsBus

# from .koch_follower import KochFollower
# from .config_koch_follower import KochFollowerEndEffectorConfig

# logger = logging.getLogger(__name__)


# class KochFollowerEndEffector(KochFollower):
#     """
#     Koch follower robot with end-effector space control.
#     """

#     config_class = KochFollowerEndEffectorConfig
#     name = "koch_follower_end_effector"

#     def __init__(self, config: KochFollowerEndEffectorConfig):
#         super().__init__(config)
#         self.bus = FeetechMotorsBus(
#             port=self.config.port,
#             motors={
#                     "shoulder_pan": Motor(1, "xl430-w250", MotorNormMode.DEGREES),
#                     "shoulder_lift": Motor(2, "xl430-w250", MotorNormMode.DEGREES),
#                     "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.DEGREES),
#                     "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.DEGREES),
#                     "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.DEGREES),
#                     "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
#                     },
#             calibration=self.calibration,
#         )

#         # 相機設定
#         self.cameras = make_cameras_from_configs(config.cameras)

#         self.config = config

#         if self.config.urdf_path is None:
#             raise ValueError(
#                 "urdf_path must be provided in KochFollowerEndEffectorConfig."
#             )

#         # 建立運動學模型
#         self.kinematics = RobotKinematics(
#             urdf_path=self.config.urdf_path,
#             target_frame_name=self.config.target_frame_name,
#         )

#         self.end_effector_bounds = self.config.end_effector_bounds
#         self.current_ee_pos = None
#         self.current_joint_pos = None

#     # --------------------------------------------------
#     # Control methods
#     # --------------------------------------------------
#     def send_action(self, action: Dict[str, Any]) -> None:
#         """將 end-effector action 轉換成馬達控制訊號"""
#         if "ee_pos" in action:
#             target_ee_pos = np.array(action["ee_pos"])
#             try:
#                 joint_angles = self.kinematics.inverse_kinematics(target_ee_pos)
#                 if joint_angles is not None:
#                     self.bus.set_positions(joint_angles)
#                 else:
#                     logger.warning("IK failed for target ee_pos: %s", target_ee_pos)
#             except Exception as e:
#                 logger.error("IK computation error: %s", e)

#         if "gripper" in action:
#             gripper_value = float(action["gripper"])
#             self.bus.set_positions({"gripper": gripper_value})

#     def get_observation(self) -> Dict[str, Any]:
#         """回傳機械臂當前觀測值"""
#         try:
#             joint_positions = self.bus.get_positions()
#             ee_pos = self.kinematics.forward_kinematics(joint_positions)

#             obs = {
#                 "joint_pos": joint_positions,
#                 "ee_pos": ee_pos,
#                 "images": {name: cam.get_image() for name, cam in self.cameras.items()},
#             }

#             self.current_joint_pos = joint_positions
#             self.current_ee_pos = ee_pos

#             return obs

#         except DeviceNotConnectedError as e:
#             logger.error("Device not connected: %s", e)
#             return {}

#     def reset(self) -> None:
#         """重置機械臂位置"""
#         logger.info("Resetting Koch follower end effector...")

#         # 這裡可以設置回初始姿勢
#         if self.config.reset_joint_positions is not None:
#             self.bus.set_positions(self.config.reset_joint_positions)

#         time.sleep(1.0)  # 等待馬達到位



import logging
import time
from typing import Any

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

from .koch_follower import KochFollower
from .config_koch_follower import KochFollowerEndEffectorConfig

logger = logging.getLogger(__name__)


class KochFollowerEndEffector(KochFollower):
    """Koch follower robot with end-effector control support."""

    config_class = KochFollowerEndEffectorConfig
    name = "koch_follower_end_effector"

    def __init__(self, config: KochFollowerEndEffectorConfig):
        super().__init__(config)

        if config.urdf_path is None:
            raise ValueError(
                "urdf_path must be provided in the configuration for end-effector control. "
                "Please set urdf_path in your KochFollowerEndEffectorConfig."
            )

        self.config = config
        self.urdf_path = config.urdf_path
        self.target_frame = config.target_frame_name

        # 初始化 URDF 運動學
        self.kinematics = RobotKinematics(
            urdf_path=self.urdf_path,
            target_frame_name=self.target_frame,
        )

        # End-effector bounds
        self.end_effector_bounds = config.end_effector_bounds

        # 初始化當前位置
        self.current_joint_pos = None
        self.current_ee_pos = None

        # 初始化攝影機
        self.cameras = make_cameras_from_configs(config.cameras)

        logger.info(f"[KochFollowerEndEffector] Loaded URDF: {self.urdf_path}, target frame: {self.target_frame}")

    @property
    def action_features(self) -> dict[str, Any]:
        """Define end-effector control action features."""
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def send_action(self, action: dict[str, Any] | list | np.ndarray, leader_pos=None) -> dict[str, float]:
        """Transform EE-space action to joint-space and send to motors."""
        if leader_pos is not None:
            return super().send_action(leader_pos)

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if isinstance(action, dict) and any(key.endswith('.pos') for key in action.keys()):
            return super().send_action(action)

        # 轉換成 numpy array
        if isinstance(action, dict):
            delta_ee = np.array(
                [
                    action.get("delta_x", 0.0) * self.config.end_effector_step_sizes["x"],
                    action.get("delta_y", 0.0) * self.config.end_effector_step_sizes["y"],
                    action.get("delta_z", 0.0) * self.config.end_effector_step_sizes["z"],
                ],
                dtype=np.float32,
            )
            gripper_val = action.get("gripper", 1.0)
            action = np.append(delta_ee, gripper_val)
        else:
            action = np.array(action, dtype=np.float32)

        # 初始化當前位置
        if self.current_joint_pos is None:
            current_joint_pos = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([current_joint_pos[name] for name in self.bus.motors])

        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos)
        # print(self.bus.sync_read("Present_Position"))
        # 設定目標 EE 位置
        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, :3] = self.current_ee_pos[:3, :3]
        desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + action[:3]

        # 限制 EE 範圍
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # 計算逆運動學得到關節角
        target_joint_values = self.kinematics.inverse_kinematics(self.current_joint_pos, desired_ee_pos)

        # 封裝關節動作字典
        joint_action = {f"{name}.pos": target_joint_values[i] for i, name in enumerate(self.bus.motors.keys())}

        # gripper 處理
        joint_action["gripper.pos"] = np.clip(
            self.current_joint_pos[-1] + (action[-1] - 1) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )

        # 更新當前狀態
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = np.array(list(joint_action.values()))

        # 發送關節角到父類
        return super().send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        """讀取關節與攝影機狀態"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # 攝影機影像
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def reset(self):
        """重置 EE 與關節狀態"""
        self.current_joint_pos = None
        self.current_ee_pos = None
