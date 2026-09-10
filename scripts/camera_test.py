#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class CameraRangePublisher(Node):
    def __init__(self):
        super().__init__('camera_range_publisher')

        self.declare_parameter('camera_index', 4)
        self.declare_parameter('image_fps', 30.0)
        self.declare_parameter('range_fps', 10.0)
        self.declare_parameter('fixed_distance', 1.5)

        camera_index = self.get_parameter('camera_index').value
        image_fps = self.get_parameter('image_fps').value
        range_fps = self.get_parameter('range_fps').value
        self.fixed_distance = self.get_parameter('fixed_distance').value

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.range_pub = self.create_publisher(Float32, '/rangefinder/distance', 10)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at index {camera_index}')
        else:
            self.get_logger().info(f'Opened camera at index {camera_index}')

        self.image_timer = self.create_timer(1.0 / image_fps, self.image_callback)
        self.range_timer = self.create_timer(1.0 / range_fps, self.range_callback)

    def image_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.image_pub.publish(msg)

        cv2.imshow('camera_range_publisher', frame)
        cv2.waitKey(1)

    def range_callback(self):
        msg = Float32()
        msg.data = self.fixed_distance
        self.range_pub.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraRangePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()