# preview_opencv_camera.py
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
import cv2

cam = OpenCVCamera(OpenCVCameraConfig(index_or_path="/dev/video0"))
cam.connect()

while True:
    frame = cam.read()
    cv2.imshow("front", frame)
    if cv2.waitKey(1) == 27:
        break

cam.disconnect()
cv2.destroyAllWindows()
