#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'trajectory.csv'

df = pd.read_csv(path)
odom = df[df['source'] == 'odom'].sort_values('timestamp').reset_index(drop=True)
vo = df[df['source'] == 'vo'].sort_values('timestamp').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 5))

if not odom.empty:
    t = (odom['timestamp'] - odom['timestamp'].iloc[0]).to_numpy()
    ax.plot(t, odom['z'].to_numpy(), linewidth=1, label='PX4 odometry', color='tab:blue')

if not vo.empty:
    t = (vo['timestamp'] - vo['timestamp'].iloc[0]).to_numpy()
    ax.plot(t, vo['z'].to_numpy(), linewidth=1, label='Visual odometry', color='tab:orange')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Z (m)')
ax.set_title(f'Z over time — {path}')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()