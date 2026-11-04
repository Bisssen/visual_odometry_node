#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'trajectory_1_scale.csv'

df = pd.read_csv(path)
odom = df[df['source'] == 'odom'].sort_values('timestamp').reset_index(drop=True)
vo = df[df['source'] == 'vo'].sort_values('timestamp').reset_index(drop=True)

def compute_speed(df, window=200):
    dt = df['timestamp'].diff()
    dx = df['x'].diff()
    dy = df['y'].diff()
    speed = np.sqrt(dx**2 + dy**2) / dt
    speed_smooth = speed.rolling(window=window, center=True).mean()
    t = df['timestamp'] - df['timestamp'].iloc[0]
    return t.to_numpy(), speed_smooth.to_numpy()

odom_t, odom_speed = compute_speed(odom)
vo_t, vo_speed = compute_speed(vo)

odom_mask = (odom_t >= 100) & (odom_t <= 600)
vo_mask = (vo_t >= 100) & (vo_t <= 600)
print(f"Average PX4 speed: {np.nanmean(odom_speed[odom_mask]):.2f} m/s")
print(f"Average VO speed:  {np.nanmean(vo_speed[vo_mask]):.2f} m/s")
print(np.nanmean(vo_speed[vo_mask])/np.nanmean(odom_speed[odom_mask]) * 738)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(odom_t, odom_speed, linewidth=1, label='PX4 odometry', color='tab:blue')
axes[0].set_ylabel('Speed (m/s)')
axes[0].set_xlabel('Time (s)')
axes[0].set_title('PX4 speed estimate (moving average)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(vo_t, vo_speed, linewidth=1, label='Visual odometry', color='tab:orange')
axes[1].set_ylabel('Speed (m/s)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('VO speed estimate (moving average)')
axes[1].grid(True)
axes[1].legend()

plt.suptitle(f'Speed estimates — {path}')
plt.tight_layout()
plt.show()