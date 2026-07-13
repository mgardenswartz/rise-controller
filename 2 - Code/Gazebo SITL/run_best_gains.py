#!/usr/bin/env python3
import os
import argparse
import glob
import jax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Prevent JAX from gobbling up GPU if any, stay on CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from unified_orchestrator import run_trial, get_base_param_dict, FIXED_K_1, FIXED_K_2, FIXED_K_3, FIXED_K_RISE, SEED, FIXED_K_ST_1, FIXED_K_ST_2, FIXED_K_ST_3

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ros2_ws", "src", "aviary_rise_controller", "aviary_rise_controller")))
from jax_resnet import init_resnet_weights

def build_param_dict(controller_type: str, desired_trajectory: int):
    key = jax.random.PRNGKey(SEED)

    param_dict = get_base_param_dict(controller_type, desired_trajectory)
    param_dict['save_data'] = True
    param_dict['h_act_func'] = 'swish'
    param_dict['o_act_func'] = 'tanh'
    param_dict['shortcut_act_func'] = 'swish'
    param_dict['theta_bar'] = 1e6
    
    param_dict['k_1'] = FIXED_K_1
    param_dict['k_2'] = FIXED_K_2
    param_dict['k_3'] = FIXED_K_3
    param_dict['k_rise'] = FIXED_K_RISE

    initial_weight_scale_factor = float('nan')

    if controller_type == "baseline":        
        param_dict['d_in'] = 12

        param_dict['num_blocks'] = 6
        param_dict['k_0'] = 2
        param_dict['k_i'] = 2
        param_dict['hidden_width'] = 12
        param_dict['gamma'] = 10.0
        param_dict['sigma_mod'] = 3.0
        initial_weight_scale_factor = 0.1
    elif controller_type == "developed":        
        param_dict['d_in'] = 15

        param_dict['num_blocks'] = 6
        param_dict['k_0'] = 2
        param_dict['k_i'] = 2
        param_dict['hidden_width'] = 12
        param_dict['gamma'] = 10.0
        param_dict['sigma_mod'] = 3.0
        initial_weight_scale_factor = 0.1
    elif controller_type == "supertwisting":
        param_dict['k_1'] = FIXED_K_ST_1
        param_dict['k_2'] = FIXED_K_ST_2
        param_dict['k_3'] = FIXED_K_ST_3
   
    if controller_type in ["developed", "baseline"]:
        initial_weights_jax = initial_weight_scale_factor * init_resnet_weights(
            key, param_dict['d_in'], param_dict['hidden_width'],  
            param_dict['d_out'], param_dict['num_blocks'], 
            param_dict['k_0'], param_dict['k_i'], 'xavier', 'he'
        )
        param_dict['initial_weights'] = [float(w) for w in initial_weights_jax]

    return param_dict

def main():
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Pipeline")
    parser.add_argument("--desired_trajectory", type=int, choices=[1, 2], required=True)
    parser.add_argument("--wind", action="store_true")
    args = parser.parse_args()

    controllers = ["noresnet", "baseline", "developed", "supertwisting"]
    
    for controller in controllers:
        print(f"\n{'='*60}")
        print(f"[*] EVALUATING: {controller.upper()} | Trajectory: {args.desired_trajectory} | Wind: {args.wind}")
        print(f"{'='*60}")
        
        param_dict = build_param_dict(controller, args.desired_trajectory)
        
        # Run trial to generate telemetry CSVs
        result = run_trial(param_dict, args.desired_trajectory, args.wind)
        print(f"[*] {controller.upper()} Evaluation Finished.")
        if result[0] is not None:
            print(f"    - ITAE Cost: {result[0]}")
            print(f"    - RMS Error: {result[1]}")
            print(f"    - RMS Control: {result[2]}")

            # Find the newly generated CSV in plot_data/controller/traj
            traj_name = "figure_eight" if args.desired_trajectory == 1 else "rose"
            
            # Since we run on the host Mac, we check the local plot_data directory
            local_csv_dir = f"plot_data/{controller}/{traj_name}"
            
            if os.path.exists(local_csv_dir):
                csv_files = glob.glob(f"{local_csv_dir}/*.csv")
                if csv_files:
                    # Find the most recently created CSV file
                    latest_csv = max(csv_files, key=os.path.getctime)
                    print(f"[*] Triggering post-flight analysis on: {latest_csv}")
                    run_post_flight_analysis(latest_csv)
                else:
                    print(f"[!] No CSV found in {local_csv_dir}")
            else:
                print(f"[!] Directory {local_csv_dir} does not exist.")
        else:
            print(f"[!] Trial failed for {controller.upper()}, skipping analysis.")

def run_post_flight_analysis(latest_csv: str):
    print(f"[*] Analyzing: {latest_csv}")
    df = pd.read_csv(latest_csv)

    # --- 2. Calculate Decoupled Errors ---
    # XY Tracking Error Norm: sqrt((x - xd)^2 + (y - yd)^2)
    df['Error_XY'] = np.sqrt((df['x'] - df['xd'])**2 + (df['y'] - df['yd'])**2)

    # Z Tracking Error Norm: |z - zd|
    df['Error_Z'] = np.abs(df['z'] - df['zd'])

    # --- 3. Setup Output Directory ---
    output_dir = os.path.dirname(latest_csv)
    base_filename = os.path.basename(latest_csv).replace('.csv', '')

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
    plt.close(fig_static)

    # --- 4.5 Generate Weights Plot ---
    weight_cols = [col for col in df.columns if col.startswith('W')]
    if weight_cols:
        fig_weights = plt.figure(figsize=(12, 6))
        plt.title(f"ResNet Weights Evolution: {base_filename}", fontsize=14, fontweight='bold')
        for w_col in weight_cols:
            plt.plot(df['Time_s'], df[w_col], alpha=0.6, linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Weight Value")
        plt.grid(True)
        plt.tight_layout()
        weights_filepath = os.path.join(output_dir, f"{base_filename}_weights.png")
        plt.savefig(weights_filepath, dpi=300)
        print(f"[*] Saved weights plot to: {weights_filepath}")
        plt.close(fig_weights)

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
    plt.close(fig_anim)

if __name__ == "__main__":
    main()