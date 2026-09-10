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

def quaternion_to_yaw(q):
    """
    Extract yaw (rotation about Z) from a PX4 VehicleOdometry quaternion.
    q is [w, x, y, z] as published by PX4 (q[0]=w, q[1]=x, q[2]=y, q[3]=z).
    Returns yaw in radians.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
 
 
def rotate_by_yaw(dx_body, dy_body, yaw):
    """
    Rotate a body-frame displacement (forward, right) into the world frame
    using the current yaw angle.
    """
    dx_world = dx_body * math.cos(yaw) - dy_body * math.sin(yaw)
    dy_world = dx_body * math.sin(yaw) + dy_body * math.cos(yaw)
    return dx_world, dy_world
 


class MonoVideoOdometry:
    def __init__(self,
                 image_fx=736.9205,
                 image_fy=738.6126,
                 image_height=720,
                 image_width=1280,
                 debug=False,
                 lk_params=dict(winSize=(21, 21), 
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
                 detector=cv2.FastFeatureDetector_create(threshold=25)):
        
        self.debug = debug
        self.detector = detector
        self.lk_params = lk_params
        self.n_features = 0
        self.yaw = 0
        
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
        # keypoints = self.detector.detect(img)
        # points = np.array(
        #     [kp.pt for kp in keypoints],
        #     dtype=np.float32
        # ).reshape(-1, 1, 2)
        # self.keypoints = keypoints
        # return points
        points = cv2.goodFeaturesToTrack(
            img,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )


        if points is None:
            self.keypoints = []
            return np.empty((0, 1, 2), dtype=np.float32)


        return points

    def visual_odometry(self, height):
        # Detect new features if too many are lost
        if self.n_features < 250:
            self.p0 = self.detect(self.old_frame)

        print(self.n_features)

        # Calculate optical flow
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame,
            self.current_frame,
            self.p0,
            None,
            **self.lk_params
        )
        if self.p1 is None or self.p0 is None:
            print('No points')
            return

        # p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
        #     self.current_frame,
        #     self.old_frame,
        #     self.p1,
        #     None,
        #     **self.lk_params
        # )

        # fb_error = np.linalg.norm(
        #     self.p0 - p0_back,
        #     axis=2
        # ).ravel()

        # mask = (
        #     (st.ravel() == 1) &
        #     (st_back.ravel() == 1) &
        #     (fb_error < 1.5)
        # )

        mask = st.ravel() == 1

        self.good_old = self.p0[mask]
        self.good_new = self.p1[mask]

        if self.debug:
            for old, new in zip(self.good_old, self.good_new):
                x0, y0 = old.ravel()
                x1, y1 = new.ravel()

                cv2.circle(
                    self.old_frame,
                    (int(x0), int(y0)),
                    3,
                    (0, 255, 0),
                    -1
                )
            cv2.imshow("Camera", self.old_frame)

            # ESC to quit
            if cv2.waitKey(1) & 0xFF == 27:
                return
        # Need enough points
        if len(self.good_old) < 20:
            print("Not enough points")
            return

        # ---------------------------------------------------------
        # Estimate robust pixel displacement
        # ---------------------------------------------------------

        old_pts = self.good_old.reshape(-1, 2)
        new_pts = self.good_new.reshape(-1, 2)

        flow = new_pts - old_pts

        # Reject large outliers using median/MAD
        median_flow = np.median(flow, axis=0)

        deviation = np.linalg.norm(
            flow - median_flow,
            axis=1
        )

        threshold = 3.0

        inliers = deviation < threshold

        flow_good = flow[inliers]

        if len(flow_good) < 10:
            print("Not enough good flow")
            return

        # Robust estimate of image displacement
        dx_pixels, dy_pixels = np.median(flow_good, axis=0)

        # ---------------------------------------------------------
        # Convert pixels -> meters on ground
        # ---------------------------------------------------------

        fx = self.K[0, 0]
        fy = self.K[1, 1]

        dx_ground = -dx_pixels * height / fx
        dy_ground = -dy_pixels * height / fy

        # ---------------------------------------------------------
        # Movement magnitude
        # ---------------------------------------------------------

        movement = np.hypot(dx_ground, dy_ground)

        print(
            f"dx={dx_ground:.4f} m, "
            f"dy={dy_ground:.4f} m, "
            f"movement={movement:.4f} m"
        )

        # Integrate
        # self.x += dx_ground
        # self.y += dy_ground
        dx_world, dy_world = rotate_by_yaw(dx_ground, dy_ground, self.yaw)
        self.x += dx_world * 2.2
        self.y += dy_world * 2.2
        


        self.p0 = self.good_new.reshape(-1, 1, 2)  # ← track survivors next frame
        self.n_features = len(self.p0)              # ← add thi
        print(self.n_features)

    def process_frame(self, frame, height, yaw):
        self.yaw = yaw
        if self.current_frame is None:
            self.current_frame = frame
        else:
            self.old_frame = self.current_frame
            self.current_frame = frame
            self.visual_odometry(height)
            
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

        # if self.debug:
        #     self.timer = self.create_timer(0.1, self.timer_callback)
        
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