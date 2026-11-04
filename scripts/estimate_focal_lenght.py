import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('trajectory_narrow.csv')
odom = df[df['source'] == 'odom'].sort_values('timestamp').reset_index(drop=True)
vo = df[df['source'] == 'vo'].sort_values('timestamp').reset_index(drop=True)


# Time ranges where drone moves primarily in x and y
t_x_end = 1793783293.336736
t_y_start = t_x_end
t_y_end = 1793783425.630468

# # X segment
# odom_x = odom[odom['timestamp'] <= t_x_end]
# vo_x = vo[vo['timestamp'] <= t_x_end]

# # Y segment
# odom_y = odom[(odom['timestamp'] >= t_y_start) & (odom['timestamp'] <= t_y_end)]
# vo_y = vo[(vo['timestamp'] >= t_y_start) & (vo['timestamp'] <= t_y_end)]


# # X segment — compare displacement in x
# odom_x_disp = odom_x['x'].iloc[-1] - odom_x['x'].iloc[0]
# vo_x_disp = vo_x['x'].iloc[-1] - vo_x['x'].iloc[0]
# scale_x = odom_x_disp / vo_x_disp
# fx_for_vo = (736.9205 / abs(scale_x)) / 2 # Due to half image size

# # Y segment — compare displacement in y
# odom_y_disp = odom_y['y'].iloc[-1] - odom_y['y'].iloc[0]
# vo_y_disp = vo_y['y'].iloc[-1] - vo_y['y'].iloc[0]
# scale_y = odom_y_disp / vo_y_disp
# fy_for_vo = (738.6126 / abs(scale_y)) / 2 # Due to half image size

# print(f"X segment — odom: {odom_x_disp:.2f} m, VO: {vo_x_disp:.2f} m, scale: {scale_x:.4f}")
# print(f"Y segment — odom: {odom_y_disp:.2f} m, VO: {vo_y_disp:.2f} m, scale: {scale_y:.4f}")
# print(f"\nFocal lengths to pass to VO node:")
# print(f"  fx: {fx_for_vo:.4f}")
# print(f"  fy: {fy_for_vo:.4f}")


# odom_x_disp = abs(odom['x'].iloc[-1] - odom['x'].iloc[0])
# odom_y_disp = abs(odom['y'].iloc[-1] - odom['y'].iloc[0])

# vo_x_disp = abs(vo['x'].iloc[-1] - vo['x'].iloc[0])
# vo_y_disp = abs(vo['y'].iloc[-1] - vo['y'].iloc[0])

# scale_x = odom_x_disp / vo_x_disp
# scale_y = odom_y_disp / vo_y_disp

# fx_for_vo = (736.9205 / 2) / scale_x
# fy_for_vo = (738.6126 / 2) / scale_y

# print(f"X displacement — odom: {odom_x_disp:.2f} m, VO: {vo_x_disp:.2f} m, scale: {scale_x:.4f}")
# print(f"Y displacement — odom: {odom_y_disp:.2f} m, VO: {vo_y_disp:.2f} m, scale: {scale_y:.4f}")
# print(f"\nFocal lengths to pass to VO node:")
# print(f"  fx: {fx_for_vo:.4f}")
# print(f"  fy: {fy_for_vo:.4f}")

# # --- Plot ---
# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# for ax, label, source in zip(axes, ['VO', 'Odom'], [vo, odom]):

#     # Full run (grey)
#     ax.plot(source['x'].to_numpy(), source['y'].to_numpy(),
#             color='lightgrey', linewidth=1, label='Full run', zorder=1)

#     # X segment (blue)
#     seg_x = source[source['timestamp'] <= t_x_end]
#     ax.plot(seg_x['x'].to_numpy(), seg_x['y'].to_numpy(),
#             color='royalblue', linewidth=2, label='X segment', zorder=2)

#     # Y segment (orange)
#     seg_y = source[(source['timestamp'] >= t_y_start) & (source['timestamp'] <= t_y_end)]
#     ax.plot(seg_y['x'].to_numpy(), seg_y['y'].to_numpy(),
#             color='darkorange', linewidth=2, label='Y segment', zorder=3)

#     ax.set_title(label)
#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.axis('equal')
#     ax.grid(True)
#     ax.legend()

# plt.suptitle('Trajectory with X/Y calibration segments marked')
# plt.tight_layout()
# plt.show()

# Sample every second
t_start = max(odom['timestamp'].min(), vo['timestamp'].min())
t_end = min(odom['timestamp'].max(), vo['timestamp'].max())
sample_times = np.arange(t_start, t_end, 5)

# Interpolate both at sample times
odom_x_sampled = np.interp(sample_times, odom['timestamp'], odom['x'])
odom_y_sampled = np.interp(sample_times, odom['timestamp'], odom['y'])
vo_x_sampled = np.interp(sample_times, vo['timestamp'], vo['x'])
vo_y_sampled = np.interp(sample_times, vo['timestamp'], vo['y'])

# Displacement per second
odom_dx = np.diff(odom_x_sampled)
odom_dy = np.diff(odom_y_sampled)
vo_dx = np.diff(vo_x_sampled)
vo_dy = np.diff(vo_y_sampled)

# Only use samples where there is meaningful motion to avoid division by near zero
min_motion = 0.5  # metres per second minimum

mask_x = np.abs(vo_dx) > min_motion
mask_y = np.abs(vo_dy) > min_motion

scale_x = np.median(odom_dx[mask_x] / vo_dx[mask_x])
scale_y = np.median(odom_dy[mask_y] / vo_dy[mask_y])

fx_default = 740#736.9205
fy_default = 704#738.6126

fx_for_vo = fx_default / scale_x
fy_for_vo = fy_default / scale_y

print(f"Samples used — x: {mask_x.sum()}, y: {mask_y.sum()}")
print(f"Median scale x: {scale_x:.4f}")
print(f"Median scale y: {scale_y:.4f}")
print(f"\nFocal lengths to pass to VO node:")
print(f"  fx: {fx_for_vo:.4f}")
print(f"  fy: {fy_for_vo:.4f}")