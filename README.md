# Visual Odometry Node

## Overview

This node performs monocular visual odometry using optical flow to estimate position from a downward-facing camera and rangefinder.

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera image stream (BGR) |
| `/rangefinder/distance` | `std_msgs/Float32` | Height from ground in meters |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/visual_odometry/pose` | `geometry_msgs/Point` | Estimated position (x, y, z) where z = height |

## Method

1. **Feature Detection**: FAST corner detector finds keypoints in each frame
2. **Optical Flow**: Lucas-Kanade pyramid tracks features between frames
3. **Pose Estimation**: Essential matrix decomposition recovers camera translation
4. **Scale Recovery**: Uses camera height from rangefinder to convert pixel motion to real-world displacement
5. **Position Integration**: Accumulates displacement to estimate global position

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_fx` | 736.9205 | Focal length X (pixels) |
| `image_fy` | 738.6126 | Focal length Y (pixels) |
| `image_height` | 720 | Image height (pixels) |
| `image_width` | 1280 | Image width (pixels) |

## Usage

```bash
# Run node
ros2 run visual_odometry_node visual_odometry_node

# View position
ros2 topic echo /visual_odometry/pose