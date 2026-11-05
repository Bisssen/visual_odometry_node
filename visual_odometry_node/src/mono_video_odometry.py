import math, cv2
import numpy as np
import time

# def quaternion_to_yaw(q: list[float] | np.ndarray) -> float:
#     """
#     Extract yaw (rotation about Z) from a PX4 VehicleOdometry quaternion.
#     q is [w, x, y, z] as published by PX4 (q[0]=w, q[1]=x, q[2]=y, q[3]=z).
#     Returns yaw in radians.
#     """
#     w, x, y, z = q[0], q[1], q[2], q[3]
#     siny_cosp = 2.0 * (w * z + x * y)
#     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
#     return math.atan2(siny_cosp, cosy_cosp)


def quaternion_to_rotation_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from a PX4 VehicleOdometry quaternion.
    q is [w, x, y, z] as published by PX4 (q[0]=w, q[1]=x, q[2]=y, q[3]=z).
    Transforms body-frame vectors into world-frame.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ])

def rotate_by_yaw(dx_body: float, dy_body: float, yaw: float) -> tuple[float, float]:
    """
    Rotate a body-frame displacement (forward, right) into the world frame
    using the current yaw angle.
    """
    dx_world = dx_body * math.cos(yaw) - dy_body * math.sin(yaw)
    dy_world = dx_body * math.sin(yaw) + dy_body * math.cos(yaw)
    return dx_world, dy_world

class MonoVideoOdometry:
    def __init__(self,
                image_fx: float = 736.9205,
                image_fy: float = 738.6126,
                image_height: int | None = None,
                image_width: int | None = None,
                debug: bool = False,
                lk_params: dict = dict(winSize=(21, 21),
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
                detector: cv2.FastFeatureDetector = cv2.FastFeatureDetector_create(threshold=25)):

        self.debug: bool = debug
        self.detector: cv2.FastFeatureDetector = detector
        self.lk_params: dict = lk_params
        self.n_features: int = 0
        self.attitude: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self.height: float = 0.0

        self.image_fx = image_fx
        self.image_fy = image_fy

        if image_height is None or image_width is None:
            self.K: np.ndarray | None = None
        else:
            self._build_K(image_fx, image_fy, image_width, image_height)

        self.x: float = 0.0
        self.y: float = 0.0
        self.p0: np.ndarray | None = None
        self.old_frame: np.ndarray | None = None
        self.current_frame: np.ndarray | None = None
        self.good_old: np.ndarray | None = None
        self.good_new: np.ndarray | None = None
        self.p1: np.ndarray | None = None
        self.scalar: float = 1
        self.last_time: float | None = None  # The time which the last measurement was made at
        self.speed: float = 0.0
        self.speed_limit: float = 50.0


    def _build_K(self, fx: float, fy: float, width: int, height: int) -> None:
        cx = width / 2
        cy = height / 2
        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def set_image_size(self, width: int, height: int) -> None:
        """Update K for the actual image size being processed."""
        self._build_K(self.image_fx, self.image_fy, width, height)

    def process_frame(self, frame: np.ndarray, height: float, attitude: list[float] | np.ndarray) -> None:
        if self.K is None:
            return  # Can't run without calibration matrix

        if abs(height) < 50:
            return

        self.height = height
        self.attitude = attitude
        if self.current_frame is None:
            self.current_frame = frame
        else:
            self.old_frame = self.current_frame
            self.current_frame = frame
            self.visual_odometry()

    def detect(self, img: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            img,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        return points

    def track_features(self) -> bool:
        """
        Run Lucas-Kanade optical flow and filter to successfully tracked points.
        Returns False if tracking failed or too few points remain.
        """
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame,
            self.current_frame,
            self.p0,
            None,
            **self.lk_params
        )

        # Check if any points are detected
        if self.p1 is None or self.p0 is None:
            return False

        # Get the correctly tracked features
        mask = st.ravel() == 1
        self.good_old = self.p0[mask]
        self.good_new = self.p1[mask]

        # Not enough points
        if len(self.good_old) < 20:
            return False

        return True

    def debug_draw(self) -> bool:
        """
        Draw tracked feature points on the old frame and display it.
        Returns True if ESC was pressed.
        """
        if not self.debug:
            return False
        for old, new in zip(self.good_old, self.good_new):
            x0, y0 = old.ravel()
            cv2.circle(self.old_frame, (int(x0), int(y0)), 3, (0, 255, 0), -1)

        cv2.imshow("Camera", self.old_frame)
        return cv2.waitKey(1) & 0xFF == 27

    def get_pixel_displacement(self) -> tuple[float, float] | tuple[None, None]:
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
            return None, None

        # Robust estimate of image displacement
        dx_pixels, dy_pixels = np.median(flow_good, axis=0)
        return dx_pixels, dy_pixels

    def update_position(self, dx_pixels: float, dy_pixels: float) -> None:
        """
        Convert pixel displacement to metric ground displacement, correct for
        the full attitude (roll, pitch, yaw) via the quaternion rotation matrix,
        integrate into world position, and update tracked features.
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]

        # ---------------------------------------------------------
        # Convert pixels -> normalised camera-frame ray
        # ---------------------------------------------------------
        dx_cam = -dx_pixels / fx
        dy_cam = -dy_pixels / fy

        # ---------------------------------------------------------
        # Correct for attitude by rotating into world frame,
        # then scale by height to get ground-plane displacement
        # ---------------------------------------------------------
        R = quaternion_to_rotation_matrix(self.attitude)

        # Camera-frame displacement as a vector (z=0, we only have 2D flow)
        cam_vec = np.array([dx_cam, dy_cam, 0.0])

        # Rotate into world frame
        world_vec = R @ cam_vec

        # Scale by height — divide by the z-component of the nadir direction
        # (how much of the camera's z-axis points at the ground after tilting)
        nadir = R @ np.array([0.0, 0.0, 1.0])
        scale = self.height / nadir[2] if abs(nadir[2]) > 0.01 else self.height

        dx_world = world_vec[0] * scale
        dy_world = world_vec[1] * scale
        dx_world, dy_world = rotate_by_yaw(dx_world, dy_world, 1.57)

        if self.debug:
            movement = np.hypot(dx_world, dy_world)
            print(
                f"dx={dx_world:.4f} m, "
                f"dy={dy_world:.4f} m, "
                f"movement={movement:.4f} m"
            )


        dx_limited, dy_limited = self.limit_speed(
            dx_world * self.scalar,
            dy_world * self.scalar
        )

        self.x += dx_limited
        self.y += dy_limited

        self.p0 = self.good_new.reshape(-1, 1, 2)  # ← track survivors next frame
        self.n_features = len(self.p0)              # ← add thi


    def limit_speed(self, dx: float, dy: float) -> tuple[float, float]:
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return dx, dy

        dt = now - self.last_time
        self.last_time = now

        if dt <= 0:
            return dx, dy

        speed = np.hypot(dx, dy) / dt
        self.speed = float(speed)
        if speed > self.speed_limit:
            scale = (self.speed_limit * dt) / np.hypot(dx, dy)
            dx *= scale
            dy *= scale

        return dx, dy


    def visual_odometry(self) -> None:
        # Detect new features if too many are lost
        if self.n_features < 250:
            self.p0 = self.detect(self.old_frame)

        if not self.track_features():
            return

        if self.debug_draw():
            return

        dx_pixels, dy_pixels = self.get_pixel_displacement()
        if dx_pixels is None:
            return

        self.update_position(dx_pixels, dy_pixels)


    def get_position(self) -> tuple[float, float]:
        return self.x, self.y