import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
import cv2
import numpy as np
import tf 
from franka_msgs.msg import FrankaState
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, HomingAction, HomingGoal
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController

class ImagePublisher:
    def __init__(self):
        rospy.init_node('minimal_image_publisher', anonymous=True)
        self.image_pub = rospy.Publisher('/robot0_agentview_image', Image, queue_size=10)
        self.eyeinhand_pub_rear = rospy.Publisher('/robot0_eye_in_hand_image_rear', Image, queue_size=10)
        self.eyeinhand_pub_front = rospy.Publisher('/robot0_eye_in_hand_image_front', Image, queue_size=10)
        self.eef_pose_pub = rospy.Publisher('/robot0_eef_pose', PoseStamped, queue_size=10)
        self.gripper_state_pub = rospy.Publisher('/robot0_gripper_width', Float32, queue_size=10)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_state_callback)
        rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_state_callback)

        self.cap1 = cv2.VideoCapture(12)
        self.cap2 = cv2.VideoCapture(4)
        self.cap3 = cv2.VideoCapture(10)
        if not self.cap1.isOpened() or not self.cap2.isOpened() or not self.cap3.isOpened():
            rospy.logerr("無法打開攝影機")
            exit(1)
        rospy.loginfo("攝影機已成功打開")

    def gripper_state_callback(self, msg):
        if len(msg.position) >= 2:
            # gripper width = left + right finger joint positions
            width = msg.position[0] + msg.position[1]
            #rospy.loginfo(f"Gripper width: {width:.3f}")
            self.current_gripper_width = width
            self.gripper_state_pub.publish(Float32(width))
        else:
            rospy.logwarn("Gripper joint states message does not contain enough position data.")

    def numpy_to_imgmsg(self, img):
        msg = Image()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = img.shape[1] * 3
        msg.data = img.tobytes()
        return msg

    def publish_agentview_image(self, cv_image):
        resized = cv2.resize(cv_image, (640, 480))
        msg = self.numpy_to_imgmsg(resized)
        self.image_pub.publish(msg)

    def publish_eye_in_hand_image_rear(self, cv_image):
        resized = cv2.resize(cv_image, (640, 480))
        msg = self.numpy_to_imgmsg(resized)
        self.eyeinhand_pub_rear.publish(msg)

    def publish_eye_in_hand_image_front(self, cv_image):
        resized = cv2.resize(cv_image, (640, 480))
        msg = self.numpy_to_imgmsg(resized)
        self.eyeinhand_pub_front.publish(msg)

    def franka_state_callback(self, msg):
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose.position.x = msg.O_T_EE[12]
        pose_msg.pose.position.y = msg.O_T_EE[13]
        pose_msg.pose.position.z = msg.O_T_EE[14]
        # Extract rotation matrix (first 12 elements, 3x4, but only 3x3 is rotation)
        rot = np.array(msg.O_T_EE).reshape(4,4)[:3,:3]
        quat = tf.transformations.quaternion_from_matrix(np.vstack((np.hstack((rot, np.array([[0],[0],[0]]))), np.array([0,0,0,1]))))
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.eef_pose_pub.publish(pose_msg)

    def publish_loop(self, zoom_factor=1.9):#1.3
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            ret3, frame3 = self.cap3.read()
            if zoom_factor >= 1.0:
                h, w = frame1.shape[:2]
                h, w = frame2.shape[:2]
                new_w = int(w / zoom_factor)
                new_h = int(h / zoom_factor)
                start_x = (w - new_w) // 2
                start_y = max((h // 2) - new_h, 0)
                cropped = frame1[start_y:start_y + new_h, start_x:start_x + new_w]
                cropped_2 = frame2[start_y:start_y + new_h, start_x:start_x + new_w]
                frame1 = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                frame2 = cv2.resize(cropped_2, (w, h), interpolation=cv2.INTER_LINEAR)
            if not ret1 or not ret2 or not ret3:
                rospy.logwarn("攝影機讀取失敗")
                continue
            self.publish_agentview_image(frame1)
            self.publish_eye_in_hand_image_rear(frame2)
            self.publish_eye_in_hand_image_front(frame3)
            rate.sleep()


if __name__ == "__main__":
    node = ImagePublisher()
    node.publish_loop()
