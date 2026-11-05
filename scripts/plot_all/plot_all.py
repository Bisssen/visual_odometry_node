#!/usr/bin/env python3
import sys
import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    odom_t, odom_x, odom_y, odom_z = [], [], [], []
    vo_t, vo_x, vo_y, vo_z = [], [], [], []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['source'] == 'odom':
                odom_t.append(float(row['timestamp']))
                odom_x.append(float(row['x']))
                odom_y.append(float(row['y']))
                odom_z.append(float(row['z']) if 'z' in row else 0.0)
            elif row['source'] == 'vo':
                vo_t.append(float(row['timestamp']))
                vo_x.append(float(row['x']))
                vo_y.append(float(row['y']))
                vo_z.append(float(row['z']) if 'z' in row else 0.0)

    return odom_t, odom_x, odom_y, odom_z, vo_t, vo_x, vo_y, vo_z


def interpolate_odom(odom_t, odom_vals, vo_t):
    return np.interp(vo_t, odom_t, odom_vals)


def plot_csv(path):
    odom_t, odom_x, odom_y, odom_z, vo_t, vo_x, vo_y, vo_z = load_csv(path)

    if not odom_x and not vo_x:
        print(f'  No data in {path}, skipping')
        return

    if vo_x and odom_x:
        vo_x = [x + (odom_x[0] - vo_x[0]) for x in vo_x]
        vo_y = [y + (odom_y[0] - vo_y[0]) for y in vo_y]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Trajectory ---
    ax = axes[0]
    if odom_x:
        ax.plot(odom_y, odom_x, label=f'PX4 odometry ({len(odom_x)} pts)', linewidth=1.5)
        ax.plot(odom_y[0], odom_x[0], 'o', color='tab:blue')
    if vo_x:
        ax.plot(vo_y, vo_x, label=f'Visual odometry ({len(vo_x)} pts)', linewidth=1.5)
        ax.plot(vo_y[0], vo_x[0], 'o', color='tab:orange')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Trajectory')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    if vo_t and odom_t:
        matched_x = interpolate_odom(odom_t, odom_x, vo_t)
        matched_y = interpolate_odom(odom_t, odom_y, vo_t)
        # error_x = np.array(vo_x) - matched_x
        # error_y = np.array(vo_y) - matched_y
        # t_rel = np.array(vo_t) - vo_t[0]

        # # --- X error ---
        # axes[1].plot(t_rel, error_x, color='tab:red', linewidth=1)
        # axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        # axes[1].set_xlabel('Time (s)')
        # axes[1].set_ylabel('Error (m)')
        # axes[1].set_title('X error (VO - GT)')
        # axes[1].grid(True)

        # # --- Y error ---
        # axes[2].plot(t_rel, error_y, color='tab:purple', linewidth=1)
        # axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
        # axes[2].set_xlabel('Time (s)')
        # axes[2].set_ylabel('Error (m)')
        # axes[2].set_title('Y error (VO - GT)')
        # axes[2].grid(True)

        error = np.sqrt((np.array(vo_x) - matched_x)**2 + (np.array(vo_y) - matched_y)**2)
        t_rel = np.array(vo_t) - vo_t[0]

        axes[1].plot(t_rel, error, color='tab:red', linewidth=1)
        axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Error (m)')
        axes[1].set_title('Euclidean error (VO - GT)')
        axes[1].grid(True)

    # # --- Z over time ---
    # ax4 = axes[3]
    # if odom_t and odom_z:
    #     t_odom_rel = np.array(odom_t) - odom_t[0]
    #     ax4.plot(t_odom_rel, odom_z, label='PX4 z', linewidth=1.5)
    # if vo_t and vo_z:
    #     t_vo_rel = np.array(vo_t) - vo_t[0]
    #     ax4.plot(t_vo_rel, vo_z, label='VO z', linewidth=1.5)
    # ax4.set_xlabel('Time (s)')
    # ax4.set_ylabel('Z (m)')
    # ax4.set_title('Z over time')
    # ax4.grid(True)
    # ax4.legend()

    plt.suptitle(os.path.basename(path))
    plt.tight_layout()

    out = path.replace('.csv', '.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved {out}')


def main():
    search_dir = os.path.dirname(os.path.abspath(__file__))
    csvs = sorted(glob.glob(os.path.join(search_dir, '*.csv')))

    if not csvs:
        print('No CSV files found.')
        return

    print(f'Found {len(csvs)} CSV files:')
    for path in csvs:
        print(f'  {os.path.basename(path)}')
        plot_csv(path)

    print('Done.')


if __name__ == '__main__':
    main()