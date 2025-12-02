# from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
# from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
# from lerobot.cameras.configs import ColorMode, Cv2Rotation
# import cv2
# import numpy as np
# # Create a `RealSenseCameraConfig` specifying your camera’s serial number and enabling depth.
# config = RealSenseCameraConfig(
#     serial_number_or_name="243322071995",
#     fps=15,
#     width=640,
#     height=480,
#     color_mode=ColorMode.RGB,
#     use_depth=True,
#     rotation=Cv2Rotation.NO_ROTATION
# )

# # Instantiate and connect a `RealSenseCamera` with warm-up read (default).
# camera = RealSenseCamera(config)
# camera.connect()

# # Capture a color frame via `read()` and a depth map via `read_depth()`.
# while True:
#     color_frame = camera.read()
#     depth_map = camera.read_depth()
#     depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#     depth_gray = depth_norm.astype(np.uint8)
#     depth_rgb = np.zeros((depth_gray.shape[0], depth_gray.shape[1], 3), dtype=np.uint8)
#     depth_rgb[..., 2] = depth_gray
#     color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

#     combined = np.vstack((color_frame, depth_rgb))
#     cv2.imshow("combine", combined)
#     # print("Color frame shape:", color_frame.shape)
#     # print("Depth map shape:", combined.shape)
#     # print(depth_map)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# camera.disconnect()

import cv2
import numpy as np
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# ---------------- Config ----------------
config = RealSenseCameraConfig(
    serial_number_or_name="243322071995",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)

# Depth 顯示範圍 (mm)
min_depth = 0
max_depth = 100
limit_depth = 1500

# ---------------- Connect ----------------
camera = RealSenseCamera(config)
camera.connect()

try:
    while True:
        # 讀取彩色影像與深度影像
        color_frame = camera.read()
        depth_map = camera.read_depth()
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

        # 限制顯示範圍
        # depth_map = np.clip(depth_map, min_depth, max_depth)

        # 映射到 0-255
        depth_map[depth_map>limit_depth] = max_depth
        depth_norm = ((depth_map - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        depth_rgb = np.zeros((depth_norm.shape[0], depth_norm.shape[1], 3), dtype=np.uint8)
        depth_rgb[..., 1] = depth_norm  # 只填紅色通道

        # 合併上下顯示
        combined = np.vstack((color_frame, depth_rgb))

        cv2.imshow("Color + Depth", combined)

        # 按 q 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.disconnect()
    cv2.destroyAllWindows()
