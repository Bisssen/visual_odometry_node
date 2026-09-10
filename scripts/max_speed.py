#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from px4_msgs.msg import VehicleOdometry


class MaxSpeedTracker(Node):
    def __init__(self):
        super().__init__('max_speed_tracker')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos
        )

        self.max_speed = 0.0
        self.max_speed_msg = None

        self.get_logger().info('Tracking max speed... Ctrl+C to stop and print result')

    def odom_callback(self, msg: VehicleOdometry):
        vx, vy, vz = msg.velocity
        speed = math.sqrt(vx**2 + vy**2)

        if speed > self.max_speed:
            self.max_speed = speed

        self.get_logger().info(f'Max speed: {self.max_speed:.3f} m/s')
        self.get_logger().info(f'Current speed: {speed:.3f} m/s')


def main(args=None):
    rclpy.init(args=args)
    node = MaxSpeedTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Final max speed: {node.max_speed:.3f} m/s')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()