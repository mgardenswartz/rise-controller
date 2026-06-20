import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Locate the latest CSV ---
csv_files = glob.glob("trial_data_cost_*.csv")
if not csv_files:
    print("[!] No CSV files found in the current directory.")
    exit()

# Get the most recently created CSV
latest_csv = max(csv_files, key=os.path.getctime)
print(f"[*] Analyzing: {latest_csv}")

df = pd.read_csv(latest_csv)

# --- 2. Calculate Decoupled Errors ---
# XY Tracking Error Norm: sqrt((x - xd)^2 + (y - yd)^2)
df['Error_XY'] = np.sqrt((df['x'] - df['xd'])**2 + (df['y'] - df['yd'])**2)

# Z Tracking Error Norm: |z - zd|
df['Error_Z'] = np.abs(df['z'] - df['zd'])

# --- 3. Setup Output Directory ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
base_filename = latest_csv.replace('.csv', '')

# --- 4. Generate Static Summary Plot ---
fig_static = plt.figure(figsize=(14, 8))
plt.suptitle(f"Flight Analysis: {base_filename}", fontsize=16, fontweight='bold')

# Subplot 1: Top-Down XY View
ax1 = plt.subplot(1, 2, 1)
ax1.plot(df['xd'], df['yd'], 'k--', label='Desired Trajectory', alpha=0.7)
ax1.plot(df['x'], df['y'], 'b-', label='Actual Flight Path', linewidth=2)
ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', marker='o', s=100, label='Start')
ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', marker='X', s=100, label='End')
ax1.set_title("Top-Down Trajectory (X-Y Plane)")
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Subplot 2: XY Error
ax2 = plt.subplot(2, 2, 2)
ax2.plot(df['Time_s'], df['Error_XY'], 'm-', linewidth=2)
ax2.set_title("Horizontal Tracking Error Norm ($e_{xy}$)")
ax2.set_ylabel("Error (m)")
ax2.grid(True)

# Subplot 3: Z Error
ax3 = plt.subplot(2, 2, 4, sharex=ax2)
ax3.plot(df['Time_s'], df['Error_Z'], 'c-', linewidth=2)
ax3.set_title("Vertical Tracking Error Norm ($e_z$)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error (m)")
ax3.grid(True)

plt.tight_layout()
static_filepath = os.path.join(output_dir, f"{base_filename}_summary.png")
plt.savefig(static_filepath, dpi=300)
print(f"[*] Saved static summary to: {static_filepath}")

# --- 5. Generate Animated XY Plot (Real-Time Synced) ---
print("[*] Generating animation...")
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.set_title("Top-Down Trajectory Animation (Real-Time)")
ax_anim.set_xlabel("X Position (m)")
ax_anim.set_ylabel("Y Position (m)")
ax_anim.grid(True)
ax_anim.axis('equal')

# Set static limits based on data bounds with a 10% margin
x_min, x_max = min(df['x'].min(), df['xd'].min()), max(df['x'].max(), df['xd'].max())
y_min, y_max = min(df['y'].min(), df['yd'].min()), max(df['y'].max(), df['yd'].max())
margin_x = (x_max - x_min) * 0.1
margin_y = (y_max - y_min) * 0.1
ax_anim.set_xlim(x_min - margin_x, x_max + margin_x)
ax_anim.set_ylim(y_min - margin_y, y_max + margin_y)

# Plot static desired trajectory in background
ax_anim.plot(df['xd'], df['yd'], 'k--', label='Desired Path', alpha=0.4)

# Initialize moving elements
line_actual, = ax_anim.plot([], [], 'b-', linewidth=2, label='Flight Path (5s trail)')
point_actual, = ax_anim.plot([], [], 'ro', markersize=8, label='Quadcopter')
point_desired, = ax_anim.plot([], [], 'go', markerfacecolor='none', markersize=10, markeredgewidth=2, label='Target Position')

# Initialize Stopwatch (Anchored to top-left of the axes)
time_text = ax_anim.text(0.03, 0.95, '', transform=ax_anim.transAxes, fontsize=12, 
                         fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

ax_anim.legend(loc="upper right")

# --- Real-Time Sync & Interpolation ---
fps = 30  # Change this to 60 if you want a smoother video, time will still be 1x real-time
duration = df['Time_s'].max()
video_times = np.arange(0, duration, 1.0 / fps)

# Interpolate data to match exact video frames to prevent speed-up/slow-down
x_vals = np.interp(video_times, df['Time_s'], df['x'])
y_vals = np.interp(video_times, df['Time_s'], df['y'])
xd_vals = np.interp(video_times, df['Time_s'], df['xd'])
yd_vals = np.interp(video_times, df['Time_s'], df['yd'])

# Trail mathematics
trail_length_seconds = 5.0
trail_frames = int(trail_length_seconds * fps)

def init():
    line_actual.set_data([], [])
    point_actual.set_data([], [])
    point_desired.set_data([], [])
    time_text.set_text('')
    return line_actual, point_actual, point_desired, time_text

def update(frame):
    # Calculate start index for the 5-second disappearing trail
    start_idx = max(0, frame - trail_frames)
    
    # Update trail
    line_actual.set_data(x_vals[start_idx:frame+1], y_vals[start_idx:frame+1])
    
    # Update leading points
    point_actual.set_data([x_vals[frame]], [y_vals[frame]])
    point_desired.set_data([xd_vals[frame]], [yd_vals[frame]])
    
    # Update stopwatch
    time_text.set_text(f"Elapsed: {video_times[frame]:.2f} s")
    
    return line_actual, point_actual, point_desired, time_text

anim = animation.FuncAnimation(
    fig_anim, update, frames=len(video_times), 
    init_func=init, blit=True
)

anim_filepath = os.path.join(output_dir, f"{base_filename}_animation.mp4")
writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Aviary'), bitrate=2500)

print(f"[*] Saving MP4 animation at 1x real-time (this takes a few seconds)...")
anim.save(anim_filepath, writer=writer)

print(f"[*] Saved animation to: {anim_filepath}")
print("[*] Analysis Complete.")