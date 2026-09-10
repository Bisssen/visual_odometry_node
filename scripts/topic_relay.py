#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from px4_msgs.msg import VehicleOdometry


class TopicRelay(Node):
    def __init__(self):
        super().__init__('topic_relay')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        ## This is laggy use:
        # ros2 run topic_tools relay /low_light_down_misp_decoded /camera/image_raw
        print('Use: ros2 run topic_tools relay /low_light_down_misp_decoded /camera/image_raw')
        # Image relay
        # self.image_sub = self.create_subscription(
        #     Image,
        #     '/low_light_down_misp_decoded',
        #     self.image_callback,
        #     qos
        # )
        # self.image_pub = self.create_publisher(Image, '/camera/image_raw', qos)

        # Odometry z → rangefinder
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos
        )
        self.height_pub = self.create_publisher(Float32, '/rangefinder/distance', qos)

    def image_callback(self, msg: Image):
        print('new_image')
        self.image_pub.publish(msg)

    def odom_callback(self, msg: VehicleOdometry):
        out = Float32()
        out.data = -float(msg.position[2])  # NED z is negative-down, flip to positive height
        self.height_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TopicRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()