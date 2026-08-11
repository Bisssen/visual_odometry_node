#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading

class MonoVideoOdometry:
    def __init__(self,
                 image_fx=736.9205,
                 image_fy=738.6126,
                 image_height=720,
                 image_width=1280,
                 lk_params=dict(winSize=(21, 21), 
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
                 detector=cv2.FastFeatureDetector_create(threshold=25)):
        
        self.detector = detector
        self.lk_params = lk_params
        self.id = 0
        self.n_features = 0
        
        cx = image_width / 2
        cy = image_height / 2
        self.K = np.array([
            [image_fx, 0.0, cx],
            [0.0, image_fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.x = 0.0
        self.y = 0.0
        self.keypoints = None
        self.p0 = None
        self.old_frame = None
        self.current_frame = None
        self.good_old = None
        self.good_new = None
        self.p1 = None
        
    def detect(self, img):
        keypoints = self.detector.detect(img)
        points = np.array(
            [kp.pt for kp in keypoints],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        self.keypoints = keypoints
        return points

    def visual_odometry(self, height):
        # Detect new features if too many are lost
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        # Calculate optical flow
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame,
            self.current_frame,
            self.p0,
            None,
            **self.lk_params
        )
        if self.p1 is None or self.p0 is None:
            return

        # Keep only successful tracks
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        # Need enough points
        if len(self.good_old) < 20:
            return

        # 1. Convert pixels to normalized camera rays
        old_pts = cv2.undistortPoints(self.good_old, self.K, None).reshape(-1, 2)
        new_pts = cv2.undistortPoints(self.good_new, self.K, None).reshape(-1, 2)

        # 2. Calculate Essential Matrix
        E, mask = cv2.findEssentialMat(
            old_pts, new_pts, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None:
            return

        # 3. Recover pose
        _, R, t, mask = cv2.recoverPose(E, old_pts, new_pts, self.K)

        # 4. Get translation in normalized coordinates
        tx = t[0][0]
        ty = t[1][0]

        # 5. Convert to world units using camera height
        scale = height / self.K[0, 0]  # Using fx from camera matrix

        # World displacement (negative because camera moves opposite to flow)
        dx_world = -tx * scale
        dy_world = -ty * scale

        # 6. Integrate position
        self.x += dx_world
        self.y += dy_world

        # Prepare next iteration
        self.old_frame = self.current_frame.copy()
        self.p0 = self.good_new.reshape(-1, 1, 2)
        self.n_features = len(self.p0)

    def process_frame(self, frame, height):
        if self.id < 2:
            self.current_frame = frame
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = frame
            self.visual_odometry(height)
            self.id += 1
            
    def get_position(self):
        return self.x, self.y


class VONode(Node):
    def __init__(self):
        super().__init__('vo_node')
        
        # Declare parameters
        self.declare_parameter('image_fx', 736.9205)
        self.declare_parameter('image_fy', 738.6126)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('image_width', 1280)
        
        # Get parameters
        image_fx = self.get_parameter('image_fx').value
        image_fy = self.get_parameter('image_fy').value
        image_height = self.get_parameter('image_height').value
        image_width = self.get_parameter('image_width').value
        
        # QoS for sensor data
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()
        
        # Initialize visual odometry
        self.vo = MonoVideoOdometry(
            image_fx=image_fx,
            image_fy=image_fy,
            image_height=image_height,
            image_width=image_width
        )
        
        # Data storage
        self.latest_image = None
        self.latest_height = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos
        )
        
        self.height_sub = self.create_subscription(
            Float32,
            '/rangefinder/distance',
            self.height_callback,
            qos
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(
            Point,
            '/visual_odometry/pose',
            10
        )

        threading.Thread(target=self.vo_loop, daemon=True).start()
        
        self.get_logger().info('Visual Odometry Node Started')
    
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Convert to grayscale
        self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    def height_callback(self, msg):
        self.latest_height = msg.data
    
    def vo_loop(self):
        if self.latest_height is None:
            return

        if self.latest_image is None:
            return

        z = self.latest_height

        self.vo.process_frame(self.latest_image, self.latest_height)
        x, y = self.vo.get_position()
        
        # Publish as Pose2D
        pose_msg = Point()
        pose_msg.x = x
        pose_msg.y = y
        pose_msg.z = z
        self.pose_pub.publish(pose_msg)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()