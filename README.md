# Visual Odometry Node

## Overview

This node performs monocular visual odometry using optical flow to estimate position from a downward-facing camera. Height and attitude are sourced from PX4 vehicle odometry.

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera image stream 
| `/fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` | Height (z) and attitude quaternion from PX4 |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/visual_odometry/pose` | `geometry_msgs/Point` | Estimated position (x, y, z) where z = height |

## Method

1. **Feature Detection**: `goodFeaturesToTrack` (Shi-Tomasi) detects up to 500 corners; re-runs when tracked features drop below 250
2. **Optical Flow**: Lucas-Kanade pyramid (21×21 window) tracks features between consecutive frames
3. **Outlier Rejection**: Median/MAD filtering removes flow vectors deviating more than 3 px from the median
4. **Pixel → Metric Conversion**: Median pixel displacement is divided by focal length to get normalised camera-frame displacement, then scaled by height
5. **Attitude Correction**: Full quaternion rotation matrix corrects for roll, pitch, and yaw; a fixed 90° yaw offset is applied
6. **Position Integration**: World-frame displacement is accumulated into a global (x, y) estimate

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_fx` | 736.9205 | Focal length X (pixels) |
| `image_fy` | 738.6126 | Focal length Y (pixels) |
| `image_height` | 720 | Image height (pixels) |
| `image_width` | 1280 | Image width (pixels) |
| `debug` | `false` | Enable debug visualisation and console output |
| `image_down_scale` | `0.5` | Downsizes the input image to improve run speed |

## Usage

```bash
# Run node
ros2 run visual_odometry_node visual_odometry_node

# View position
ros2 topic echo /visual_odometry/pose
```

> **Note:** `/rangefinder/distance` is subscribed but currently unused — height is read from `/fmu/out/vehicle_odometry` instead.