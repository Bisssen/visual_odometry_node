#!/usr/bin/env python3
import csv
import threading
import signal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import Point

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')

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

        self.vo_sub = self.create_subscription(
            Point,
            '/visual_odometry/pose',
            self.vo_callback,
            10
        )

        self.lock = threading.Lock()
        self.odom_data = []  # (timestamp, x, y, z)
        self.vo_data = []    # (timestamp, x, y, z)

        self.odom_x, self.odom_y = [], []
        self.vo_x, self.vo_y = [], []

        self.get_logger().info('Recording trajectory, plotting live...')

    def odom_callback(self, msg: VehicleOdometry):
        t = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.odom_data.append((t, msg.position[0], msg.position[1], msg.position[2]))
            self.odom_x.append(msg.position[0])
            self.odom_y.append(msg.position[1])

    def vo_callback(self, msg: Point):
        t = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.vo_data.append((t, msg.x, msg.y, msg.z))
            self.vo_x.append(msg.x)
            self.vo_y.append(msg.y)

    def get_data(self):
        with self.lock:
            return (
                list(self.odom_x), list(self.odom_y),
                list(self.vo_x), list(self.vo_y)
            )

    def save_csv(self, path='trajectory.csv'):
        with self.lock:
            odom_data = list(self.odom_data)
            vo_data = list(self.vo_data)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'timestamp', 'x', 'y', 'z'])

            for (t, x, y, z) in odom_data:
                writer.writerow(['odom', f'{t:.6f}', x, y, z])

            for (t, x, y, z) in vo_data:
                writer.writerow(['vo', f'{t:.6f}', x, y, z])

        self.get_logger().info(f'Saved trajectory to {path}')


def main(args=None):
    import sys
    csv_name = sys.argv[1] if len(sys.argv) > 1 else 'trajectory.csv'

    print('hi------')
    print(csv_name.split('/'))
    print('------------')
    rclpy.init(args=None)
    node = TrajectoryRecorder()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, ax = plt.subplots(figsize=(8, 8))
    odom_line, = ax.plot([], [], label='PX4 odometry', linewidth=1.5)
    vo_line, = ax.plot([], [], label='Visual odometry', linewidth=1.5)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'Live Trajectory — {csv_name.split("/")[-1]}')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    def update(frame):
        odom_x, odom_y, vo_x, vo_y = node.get_data()
        odom_line.set_data(odom_y, odom_x)
        vo_line.set_data(vo_y, vo_x)
        ax.relim()
        ax.autoscale_view()
        return odom_line, vo_line

    def handle_sigint(sig, frame):
        plt.close('all')

    signal.signal(signal.SIGINT, handle_sigint)

    ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)

    try:
        plt.show()
    finally:
        node.save_csv(csv_name)
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()