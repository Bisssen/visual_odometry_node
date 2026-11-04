# #!/usr/bin/env python3
# import sys
# import csv
# import matplotlib.pyplot as plt

# def load_csv(path):
#     odom_x, odom_y = [], []
#     vo_x, vo_y = [], []

#     with open(path, newline='') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if row['source'] == 'odom':
#                 odom_x.append(float(row['x']))
#                 odom_y.append(float(row['y']))
#             elif row['source'] == 'vo':
#                 vo_x.append(float(row['x']))
#                 vo_y.append(float(row['y']))

#     return odom_x, odom_y, vo_x, vo_y


# def main():
#     path = sys.argv[1] if len(sys.argv) > 1 else 'trajectory.csv'

#     odom_x, odom_y, vo_x, vo_y = load_csv(path)

#     fig, ax = plt.subplots(figsize=(8, 8))

#     if odom_x:
#         ax.plot(odom_y, odom_x, label=f'PX4 odometry ({len(odom_x)} pts)', linewidth=1.5)
#         ax.plot(odom_y[0], odom_x[0], 'o', color='tab:blue')   # start marker

#     if vo_x:
#         ax.plot(vo_y, vo_x, label=f'Visual odometry ({len(vo_x)} pts)', linewidth=1.5)
#         ax.plot(vo_y[0], vo_x[0], 'o', color='tab:orange')     # start marker

#     ax.set_xlabel('East (m)')
#     ax.set_ylabel('North (m)')
#     ax.set_title(f'Trajectory — {path}')
#     ax.axis('equal')
#     ax.grid(True)
#     ax.legend()

#     out = path.replace('.csv', '.png')
#     plt.savefig(out, dpi=150)
#     print(f'Saved to {out}')
#     plt.show()


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    odom_t, odom_x, odom_y = [], [], []
    vo_t, vo_x, vo_y = [], [], []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['source'] == 'odom':
                odom_t.append(float(row['timestamp']))
                odom_x.append(float(row['x']))
                odom_y.append(float(row['y']))
            elif row['source'] == 'vo':
                vo_t.append(float(row['timestamp']))
                vo_x.append(float(row['x']))
                vo_y.append(float(row['y']))

    return odom_t, odom_x, odom_y, vo_t, vo_x, vo_y


def interpolate_odom(odom_t, odom_x, odom_y, vo_t):
    """Interpolate odom x/y at each VO timestamp."""
    matched_x = np.interp(vo_t, odom_t, odom_x)
    matched_y = np.interp(vo_t, odom_t, odom_y)
    return matched_x, matched_y


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'trajectory.csv'

    odom_t, odom_x, odom_y, vo_t, vo_x, vo_y = load_csv(path)

    # Align VO start to odom start
    if vo_x and odom_x:
        vo_x = [x + (odom_x[0] - vo_x[0]) for x in vo_x]
        vo_y = [y + (odom_y[0] - vo_y[0]) for y in vo_y]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Trajectory plot ---
    ax = axes[0]
    if odom_x:
        ax.plot(odom_y, odom_x, label=f'PX4 odometry ({len(odom_x)} pts)', linewidth=1.5)
        ax.plot(odom_y[0], odom_x[0], 'o', color='tab:blue')
    if vo_x:
        ax.plot(vo_y, vo_x, label=f'Visual odometry ({len(vo_x)} pts)', linewidth=1.5)
        ax.plot(vo_y[0], vo_x[0], 'o', color='tab:orange')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'Trajectory — {path}')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    # --- Error plots ---
    if vo_t and odom_t:
        matched_odom_x, matched_odom_y = interpolate_odom(odom_t, odom_x, odom_y, vo_t)
        error_x = np.array(vo_x) - matched_odom_x
        error_y = np.array(vo_y) - matched_odom_y

        # Relative time for readability
        t_rel = np.array(vo_t) - vo_t[0]

        ax2 = axes[1]
        ax2.plot(t_rel, error_x, color='tab:red', linewidth=1)
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m)')
        ax2.set_title('X error (VO - GT)')
        ax2.grid(True)

        ax3 = axes[2]
        ax3.plot(t_rel, error_y, color='tab:purple', linewidth=1)
        ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error (m)')
        ax3.set_title('Y error (VO - GT)')
        ax3.grid(True)

    plt.suptitle(f'Trajectory analysis — {path}')
    plt.tight_layout()

    out = path.replace('.csv', '.png')
    plt.savefig(out, dpi=150)
    print(f'Saved to {out}')
    plt.show()


if __name__ == '__main__':
    main()