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
import time
import math
from px4_msgs.msg import VehicleOdometry

# from src.fisheye import FisheyeUndistorter
from src.mono_video_odometry import MonoVideoOdometry, quaternion_to_yaw

class VONode(Node):
    def __init__(self):
        super().__init__('vo_node')
        
        # Declare parameters
        self.declare_parameter('image_fx', 736.9205)
        self.declare_parameter('image_fy', 738.6126)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('debug', False)
        
        # Get parameters
        image_fx = self.get_parameter('image_fx').value
        image_fy = self.get_parameter('image_fy').value
        image_height = self.get_parameter('image_height').value
        image_width = self.get_parameter('image_width').value
        self.debug = self.get_parameter('debug').value
        print(self.debug, type(self.debug))

        # self.udistorter = FisheyeUndistorter(2704, 1540, 150, balance=0.0)
        
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
            image_width=image_width,
            debug=self.get_parameter('debug').value
        )
        
        # Data storage
        self.latest_image = None
        self.latest_height = None
        self.latest_yaw = None
        
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

        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos
        )


        threading.Thread(target=self.vo_loop, daemon=True).start()

        self.get_logger().info('Visual Odometry Node Started')
    
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        SCALE = 0.5
        self.latest_image = cv2.resize(gray, (0, 0), fx=SCALE, fy=SCALE)

    def height_callback(self, msg):
        # self.latest_height = msg.data
        pass

    def odom_callback(self, msg):
        self.latest_height = msg.position[2] * -1  # if you're using this for height too
        self.latest_yaw = quaternion_to_yaw(msg.q) + 1.57
 

    
    def vo_loop(self):
        while True:
            if self.latest_height is None:
                time.sleep(0.1)
                self.get_logger().warn('No distance measurement')
                continue

            if self.latest_yaw is None:
                time.sleep(0.1)
                self.get_logger().warn('No odometry measurement')
                continue

            if self.latest_image is None:
                time.sleep(0.001)
                continue


            z = self.latest_height
            yaw = self.latest_yaw
            tmp_image = self.latest_image.copy()
            self.latest_image = None


            self.get_logger().info('processing image')
            self.vo.process_frame(tmp_image, z, yaw)
            x, y = self.vo.get_position()
            
            # Publish as Pose2D
            pose_msg = Point()
            pose_msg.x = x
            pose_msg.y = y
            pose_msg.z = z
            self.get_logger().info(f'{x}, {y}, {z}')
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