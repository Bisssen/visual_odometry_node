#!/usr/bin/env python3
"""
Approximate fisheye undistortion for the ModalAI M0186 (IMX664, 150 deg FOV M12 lens).

NO REAL CALIBRATION DATA IS AVAILABLE for this specific lens, so this assumes
an IDEAL EQUIDISTANT (f-theta) fisheye projection with zero higher-order
distortion (k1=k2=k3=k4=0). This is the standard first-order approximation
used when no calibration exists, but it will NOT be as accurate as a real
factory or checkerboard calibration -- expect residual distortion,
especially near the frame edges.

Equidistant model: r = f * theta
  r     = pixel distance from the image center
  theta = angle (radians) between the incoming ray and the optical axis
  f     = focal length in pixels

f is derived from the DIAGONAL FOV, since that spans the full sensor extent
and is the most robust single number to anchor an equidistant model to.
"""

import math

import cv2
import numpy as np

# ── Camera parameters (from datasheet) ─────────────────────────
IMAGE_WIDTH = 2704
IMAGE_HEIGHT = 1540
DIAGONAL_FOV_DEG = 150  # from "FOV (DxHxV): 150 x 130 x 75"


def build_fisheye_intrinsics(width, height, diagonal_fov_deg):
    """
    Derive an approximate equidistant fisheye camera matrix from resolution
    and diagonal FOV alone. Returns (K, D) for use with cv2.fisheye.
    """
    diag_px = math.hypot(width, height)
    r_max = diag_px / 2.0
    theta_max = math.radians(diagonal_fov_deg / 2.0)
    f = r_max / theta_max  # pixels per radian

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Assume ideal equidistant projection -> no additional distortion terms
    D = np.zeros((4, 1), dtype=np.float64)

    return K, D


class FisheyeUndistorter:
    def __init__(self, width, height, diagonal_fov_deg, balance=0.0):
        """
        balance: 0.0 keeps max pixel density (crops more of the FOV),
                 1.0 keeps the full FOV (more black border / stretching).
                 Tune this to taste once you see the output.
        """
        self.K, self.D = build_fisheye_intrinsics(width, height, diagonal_fov_deg)
        self.dim = (width, height)

        # New camera matrix balances FOV retention vs. pixel density
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.dim, np.eye(3), balance=balance
        )

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, self.dim, cv2.CV_16SC2
        )

    def undistort(self, frame):
        return cv2.remap(
            frame, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )


def main():
    undistorter = FisheyeUndistorter(
        IMAGE_WIDTH, IMAGE_HEIGHT, DIAGONAL_FOV_DEG, balance=0.0
    )

    print("K =\n", undistorter.K)
    print("new_K =\n", undistorter.new_K)
    print("D =\n", undistorter.D)

    # ── Example: undistort a single test image ──────────────────
    # img = cv2.imread('frame.png')
    # undistorted = undistorter.undistort(img)
    # cv2.imwrite('frame_undistorted.png', undistorted)

    # ── Example: undistort a video file ──────────────────────────
    # cap = cv2.VideoCapture('input.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output_undistorted.mp4', fourcc, 30,
    #                        (IMAGE_WIDTH, IMAGE_HEIGHT))
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     out.write(undistorter.undistort(frame))
    # cap.release()
    # out.release()


if __name__ == '__main__':
    main()