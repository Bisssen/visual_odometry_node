#!/usr/bin/env python3
import sys
import csv
import matplotlib.pyplot as plt

def load_csv(path):
    odom_x, odom_y = [], []
    vo_x, vo_y = [], []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['source'] == 'odom':
                odom_x.append(float(row['x']))
                odom_y.append(float(row['y']))
            elif row['source'] == 'vo':
                vo_x.append(float(row['x']))
                vo_y.append(float(row['y']))

    return odom_x, odom_y, vo_x, vo_y


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'trajectory.csv'

    odom_x, odom_y, vo_x, vo_y = load_csv(path)

    fig, ax = plt.subplots(figsize=(8, 8))

    if odom_x:
        ax.plot(odom_y, odom_x, label=f'PX4 odometry ({len(odom_x)} pts)', linewidth=1.5)
        ax.plot(odom_y[0], odom_x[0], 'o', color='tab:blue')   # start marker

    if vo_x:
        ax.plot(vo_y, vo_x, label=f'Visual odometry ({len(vo_x)} pts)', linewidth=1.5)
        ax.plot(vo_y[0], vo_x[0], 'o', color='tab:orange')     # start marker

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'Trajectory — {path}')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    out = path.replace('.csv', '.png')
    plt.savefig(out, dpi=150)
    print(f'Saved to {out}')
    plt.show()


if __name__ == '__main__':
    main()