#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point
from px4_msgs.msg import VehicleOdometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """q = [x, y, z, w] (ROS convention)"""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ])


def yaw_from_quaternion_px4(q: np.ndarray) -> float:
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotation_matrix_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])


class ImuDeadReckoning(Node):
    def __init__(self):
        super().__init__('imu_dead_reckoning')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.imu_sub = self.create_subscription(Imu, '/imu_apps', self.imu_callback, qos)
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.pose_pub = self.create_publisher(Point, '/imu_dead_reckoning/pose', 10)

        # State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.last_t: float | None = None
        self.yaw: float = 0.0
        self.last_yaw = None

        # Gravity calibration
        self.gravity = np.array([-0.22179495, 0.26258331, 9.96988104])#[0.0, 0.0, 9.81])
        self.gravity_samples = []
        self.gravity_ready = True
        self.n_gravity_samples = 200

        # Bias drift correction
        self.acc_bias = np.zeros(3)
        self.bias_alpha = 0.001

        # Acceleration history for plotting
        self.lock = threading.Lock()
        self.t_history = []
        self.ax_history = []
        self.ay_history = []
        self.az_history = []
        self.max_history = 10000  # enough for 30s at ~300 Hz

        self.get_logger().info('IMU dead reckoning node started')

    def odom_callback(self, msg: VehicleOdometry):
        q = np.array(msg.q)
        raw_yaw = yaw_from_quaternion_px4(q)
        if self.last_yaw is not None:
            diff = raw_yaw - self.last_yaw
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            self.yaw += diff
        else:
            self.yaw = raw_yaw
        self.last_yaw = raw_yaw

    def imu_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        a_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])

        if not self.gravity_ready:
            self.gravity_samples.append(a_body.copy())
            if len(self.gravity_samples) >= self.n_gravity_samples:
                self.gravity = np.mean(self.gravity_samples, axis=0)
                self.acc_bias = self.gravity.copy()
                self.gravity_ready = True
                self.last_t = t
                self.get_logger().info(
                    f'Gravity estimated: {self.gravity}, magnitude: {np.linalg.norm(self.gravity):.4f} m/s²'
                )
            return

        if self.last_t is None:
            self.last_t = t
            return

        dt = t - self.last_t
        self.last_t = t

        if dt <= 0 or dt > 0.5:
            return

        q_imu = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        R_imu = quaternion_to_rotation_matrix(q_imu)
        a_world = R_imu @ a_body
        a_world[2] -= np.linalg.norm(self.gravity)

        self.acc_bias = (1 - self.bias_alpha) * self.acc_bias + self.bias_alpha * a_body
        bias_world = R_imu @ self.acc_bias
        bias_world[2] -= np.linalg.norm(self.gravity)
        a_world -= bias_world

        # deadband = 0.05
        # a_world[np.abs(a_world) < deadband] = 0.0

        # Store acceleration history
        # with self.lock:
        #     self.t_history.append(t)
        #     self.ax_history.append(a_world[0])
        #     self.ay_history.append(a_world[1])
        #     self.az_history.append(a_world[2])
        #     if len(self.t_history) > self.max_history:
        #         self.t_history.pop(0)
        #         self.ax_history.pop(0)
        #         self.ay_history.pop(0)
        #         self.az_history.pop(0)

        a_body_corrected = a_body - self.acc_bias
        a_body_corrected[np.abs(a_body_corrected) < 0.05] = 0.0

        
        R_yaw = rotation_matrix_z(self.yaw + 3.14)
        a_world = R_yaw @ a_body_corrected
        a_world[np.abs(a_world) < 0.05] = 0.0

        self.velocity += a_world * dt
        self.position += self.velocity * dt

        msg_out = Point()
        msg_out.x = float(self.position[0])
        msg_out.y = float(-self.position[1])
        msg_out.z = float(self.position[2])
        self.pose_pub.publish(msg_out)

        self.get_logger().info(
            f'yaw={np.degrees(self.yaw):.1f}° '
            f'pos=({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})',
            throttle_duration_sec=1.0
        )

    def get_accel_data(self):
        with self.lock:
            return (
                list(self.t_history),
                list(self.ax_history),
                list(self.ay_history),
                list(self.az_history),
            )


def main(args=None):
    rclpy.init(args=None)
    node = ImuDeadReckoning()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    # ax_x, ax_y, ax_vx, ax_vy = axes

    # line_x,  = ax_x.plot([], [], linewidth=1)
    # line_y,  = ax_y.plot([], [], linewidth=1, color='orange')
    # line_vx, = ax_vx.plot([], [], linewidth=1, color='red')
    # line_vy, = ax_vy.plot([], [], linewidth=1, color='purple')

    # ax_x.set_ylabel('Accel X (m/s²)')
    # ax_y.set_ylabel('Accel Y (m/s²)')
    # ax_vx.set_ylabel('Velocity X (m/s)')
    # ax_vy.set_ylabel('Velocity Y (m/s)')
    # ax_vy.set_xlabel('Time (s)')
    # # ax_x.set_ylim(-5, 5)
    # # ax_y.set_ylim(-5, 5)
    # # ax_vx.set_ylim(-5, 5)
    # # ax_vy.set_ylim(-5, 5)
    # ax_x.set_title('World-frame acceleration and estimated velocity (5s moving average)')
    # for ax in axes:
    #     ax.grid(True)

    # def update(frame):
    #     t, ax_data, ay_data, az_data = node.get_accel_data()
    #     if len(t) < 2:
    #         return line_x, line_y, line_vx, line_vy

    #     t0 = t[0]
    #     t_rel = np.array([x - t0 for x in t])
    #     ax_arr = np.array(ax_data)
    #     ay_arr = np.array(ay_data)

    #     window = 5.0
    #     t_smooth = []
    #     ax_smooth = []
    #     ay_smooth = []

    #     for i in range(len(t_rel)):
    #         mask = (t_rel >= t_rel[i] - window) & (t_rel <= t_rel[i])
    #         t_smooth.append(t_rel[i])
    #         ax_smooth.append(ax_arr[mask].mean())
    #         ay_smooth.append(ay_arr[mask].mean())

    #     t_smooth = np.array(t_smooth)
    #     ax_smooth = np.array(ax_smooth)
    #     ay_smooth = np.array(ay_smooth)

    #     vx = np.zeros(len(t_smooth))
    #     vy = np.zeros(len(t_smooth))
    #     for i in range(1, len(t_smooth)):
    #         dt = t_smooth[i] - t_smooth[i - 1]
    #         vx[i] = vx[i - 1] + ax_smooth[i] * dt
    #         vy[i] = vy[i - 1] + ay_smooth[i] * dt

    #     t_end = t_smooth[-1]
    #     view_mask = t_smooth >= t_end - 30.0

    #     # line_x.set_data(t_smooth[view_mask], ax_smooth[view_mask])
    #     # line_y.set_data(t_smooth[view_mask], ay_smooth[view_mask])
    #     # line_vx.set_data(t_smooth[view_mask], vx[view_mask])
    #     # line_vy.set_data(t_smooth[view_mask], vy[view_mask])
    #     line_x.set_data(t_smooth, ax_smooth)
    #     line_y.set_data(t_smooth, ay_smooth)
    #     line_vx.set_data(t_smooth, vx)
    #     line_vy.set_data(t_smooth, vy)

    #     for ax in axes:
    #         ax.relim()
    #         ax.autoscale_view()

    #     return line_x, line_y, line_vx, line_vy

    # ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy
# from sensor_msgs.msg import Imu
# from geometry_msgs.msg import Point
# from px4_msgs.msg import VehicleOdometry
# import numpy as np


# def quaternion_to_rotation_matrix(q) -> np.ndarray:
#     """q = [x, y, z, w] (ROS convention)"""
#     x, y, z, w = q
#     return np.array([
#         [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
#         [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
#         [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
#     ])


# def yaw_from_quaternion_px4(q: np.ndarray) -> float:
#     """Extract yaw from PX4 quaternion [w, x, y, z]."""
#     w, x, y, z = q
#     return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# def rotation_matrix_z(yaw: float) -> np.ndarray:
#     c, s = np.cos(yaw), np.sin(yaw)
#     return np.array([
#         [ c, -s, 0],
#         [ s,  c, 0],
#         [ 0,  0, 1]
#     ])


# class ImuDeadReckoning(Node):
#     def __init__(self):
#         super().__init__('imu_dead_reckoning')

#         qos = QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             depth=10
#         )

#         self.imu_sub = self.create_subscription(
#             Imu,
#             '/imu_apps',
#             self.imu_callback,
#             qos
#         )

#         self.odom_sub = self.create_subscription(
#             VehicleOdometry,
#             '/fmu/out/vehicle_odometry',
#             self.odom_callback,
#             qos
#         )

#         self.pose_pub = self.create_publisher(
#             Point,
#             '/imu_dead_reckoning/pose',
#             10
#         )

#         # State
#         self.position = np.zeros(3)
#         self.velocity = np.zeros(3)
#         self.last_t: float | None = None
#         self.yaw: float = 0.0  # latest yaw from PX4 odometry

#         # Gravity calibration
#         self.gravity = np.array([0.0, 0.0, 9.81])
#         self.gravity_samples = []
#         self.gravity_ready = False
#         self.n_gravity_samples = 200

#         # Bias drift correction
#         self.acc_bias = np.zeros(3)
#         self.bias_alpha = 0.001

#         self.last_yaw = None

#         self.get_logger().info('IMU dead reckoning node started')

#     def odom_callback(self, msg: VehicleOdometry):
#         q = np.array(msg.q)  # PX4: [w, x, y, z]
#         raw_yaw = yaw_from_quaternion_px4(q)

#         # Unwrap to avoid sudden flips at ±π boundary
#         if self.last_yaw is not None:
#             diff = raw_yaw - self.last_yaw
#             if diff > np.pi:
#                 diff -= 2 * np.pi
#             elif diff < -np.pi:
#                 diff += 2 * np.pi
#             self.yaw += diff
#         else:
#             self.yaw = raw_yaw

#         self.last_yaw = raw_yaw

#     def imu_callback(self, msg: Imu):
#         t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

#         a_body = np.array([
#             msg.linear_acceleration.x,
#             msg.linear_acceleration.y,
#             msg.linear_acceleration.z,
#         ])

#         if not self.gravity_ready:
#             self.gravity_samples.append(a_body.copy())
#             if len(self.gravity_samples) >= self.n_gravity_samples:
#                 self.gravity = np.mean(self.gravity_samples, axis=0)
#                 self.acc_bias = self.gravity.copy()
#                 self.gravity_ready = True
#                 self.last_t = t
#                 self.get_logger().info(
#                     f'Gravity estimated: {self.gravity}, magnitude: {np.linalg.norm(self.gravity):.4f} m/s²'
#                 )
#             return

#         if self.last_t is None:
#             self.last_t = t
#             return

#         dt = t - self.last_t
#         self.last_t = t

#         if dt <= 0 or dt > 0.5:
#             return

#         # Rotate body-frame accel into world frame using IMU orientation
#         q_imu = [
#             msg.orientation.x,
#             msg.orientation.y,
#             msg.orientation.z,
#             msg.orientation.w,
#         ]
#         R_imu = quaternion_to_rotation_matrix(q_imu)
#         a_world = R_imu @ a_body

#         # Remove gravity
#         a_world[2] -= np.linalg.norm(self.gravity)

#         # Bias correction
#         self.acc_bias = (1 - self.bias_alpha) * self.acc_bias + self.bias_alpha * a_body
#         bias_world = R_imu @ self.acc_bias
#         bias_world[2] -= np.linalg.norm(self.gravity)
#         a_world -= bias_world

#         # Deadband
#         deadband = 0.05
#         a_world[np.abs(a_world) < deadband] = 0.0

#         # Integrate velocity in world frame
#         self.velocity += a_body * dt

#         # Rotate velocity into NED/world using drone yaw before integrating position
#         R_yaw = rotation_matrix_z(self.yaw)
#         velocity_world = R_yaw @ self.velocity

#         self.position += velocity_world * dt

#         msg_out = Point()
#         msg_out.x = float(self.position[0])
#         msg_out.y = float(self.position[1])
#         msg_out.z = float(self.position[2])
#         self.pose_pub.publish(msg_out)

#         self.get_logger().info(
#             f'yaw={np.degrees(self.yaw):.1f}° '
#             f'pos=({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}) '
#             f'vel=({self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f})',
#             throttle_duration_sec=1.0
#         )


# def main(args=None):
#     rclpy.init(args=args)
#     node = ImuDeadReckoning()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Shutting down')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()




# #     #!/usr/bin/env python3
# # import rclpy
# # from rclpy.node import Node
# # from rclpy.qos import QoSProfile, QoSReliabilityPolicy
# # from sensor_msgs.msg import Imu
# # from geometry_msgs.msg import Point
# # from px4_msgs.msg import VehicleOdometry
# # import numpy as np


# # def quaternion_to_rotation_matrix(q) -> np.ndarray:
# #     """q = [w, x, y, z] (PX4 convention). Rotates body (FRD) -> world (NED)."""
# #     w, x, y, z = q
# #     return np.array([
# #         [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
# #         [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
# #         [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
# #     ])


# # class ImuDeadReckoning(Node):
# #     def __init__(self):
# #         super().__init__('imu_dead_reckoning')

# #         self.declare_parameter('accel_bias', [-0.22179495, 0.26258331, 9.96988104])
# #         bias = self.get_parameter('accel_bias').value


# #         qos = QoSProfile(
# #             reliability=QoSReliabilityPolicy.BEST_EFFORT,
# #             depth=10
# #         )

# #         self.imu_sub = self.create_subscription(
# #             Imu, '/imu_apps', self.imu_callback, qos
# #         )
# #         self.odom_sub = self.create_subscription(
# #             VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos
# #         )
# #         self.pose_pub = self.create_publisher(
# #             Point, '/imu_dead_reckoning/pose', 10
# #         )

# #         # Attitude from PX4 (body FRD -> world NED)
# #         self.R_ned_body = np.eye(3)
# #         self.have_attitude = False

# #         # IMU axes -> FRD. If the result is mirrored, try np.diag([1, -1, -1]) (FLU -> FRD)
# #         self.imu_to_frd = np.eye(3)

# #         self.g_ned = np.array([0.0, 0.0, 9.81])

# #         # State
# #         self.position = np.zeros(3)
# #         self.velocity = np.zeros(3)
# #         self.last_t: float | None = None

# #         # Startup bias calibration (only valid if the drone is stationary)
# #         self.bias_body = np.array(bias)
# #         self.calib_ready = True  # skip calibration phase
# #         self.last_t = None
# #         self.calib_samples = []
# #         self.n_calib_samples = 200

# #         self.get_logger().info('IMU dead reckoning node started')

# #     def odom_callback(self, msg: VehicleOdometry):
# #         self.R_ned_body = quaternion_to_rotation_matrix(msg.q)
# #         self.have_attitude = True

# #     def imu_callback(self, msg: Imu):
# #         if not self.have_attitude:
# #             return

# #         t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

# #         f_body = self.imu_to_frd @ np.array([
# #             msg.linear_acceleration.x,
# #             msg.linear_acceleration.y,
# #             msg.linear_acceleration.z,
# #         ])

# #         if not self.calib_ready:
# #             self.calib_samples.append(f_body.copy())
# #             if len(self.calib_samples) >= self.n_calib_samples:
# #                 samples = np.array(self.calib_samples)
# #                 mean = samples.mean(axis=0)
# #                 std = samples.std(axis=0).max()
# #                 stationary = abs(np.linalg.norm(mean) - 9.81) < 0.5 and std < 0.3

# #                 if stationary:
# #                     expected = self.R_ned_body.T @ (-self.g_ned)
# #                     self.bias_body = mean - expected
# #                     self.get_logger().info(f'Accel bias (body): {self.bias_body}')
# #                 else:
# #                     self.get_logger().warn(
# #                         f'Not stationary during calibration (|f|={np.linalg.norm(mean):.2f}, '
# #                         f'std={std:.2f}), using zero bias'
# #                     )

# #                 self.calib_ready = True
# #                 self.last_t = t
# #                 print(self.bias_body)
# #             return

# #         if self.last_t is None:
# #             self.last_t = t
# #             return

# #         dt = t - self.last_t
# #         self.last_t = t
# #         if dt <= 0 or dt > 0.5:
# #             return

# #         # Specific force -> world acceleration (NED): a = R f + g
# #         a_world = self.R_ned_body @ (f_body - self.bias_body) + self.g_ned
# #         a_world[np.abs(a_world) < 0.05] = 0.0

# #         self.velocity += a_world * dt
# #         self.position += self.velocity * dt

# #         msg_out = Point()
# #         msg_out.x = float(self.position[0])
# #         msg_out.y = float(self.position[1])
# #         msg_out.z = float(self.position[2])
# #         self.pose_pub.publish(msg_out)

# #         self.get_logger().info(
# #             f'pos=({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}) '
# #             f'vel=({self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f})',
# #             throttle_duration_sec=1.0
# #         )


# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = ImuDeadReckoning()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         node.get_logger().info('Shutting down')
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()