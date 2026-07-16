import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ---------------------------------------------------------
    # 1. Argument Parsing
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Analyze drone flight CSV data and generate diagnostic plots.")
    parser.add_argument("csv_path", type=str, help="Path to the flight data CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"The specified file does not exist: {args.csv_path}")

    # ---------------------------------------------------------
    # 2. Data Loading and Preparation
    # ---------------------------------------------------------
    df = pd.read_csv(args.csv_path)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()

    # Create the output directory: {csv_name}_figures
    csv_dir = os.path.dirname(os.path.abspath(args.csv_path))
    csv_filename = os.path.splitext(os.path.basename(args.csv_path))[0]
    output_dir = os.path.join(csv_dir, f"{csv_filename}_figures")
    os.makedirs(output_dir, exist_ok=True)

    # Calculate Tracking Error Metrics
    if 'x' in df.columns and 'xd' in df.columns:
        ex = df['x'] - df['xd']
        ey = df['y'] - df['yd']
        ez = df['z'] - df['zd']
        exy = np.sqrt(ex**2 + ey**2)
        ez_abs = np.abs(ez)
        e_3d = np.sqrt(ex**2 + ey**2 + ez**2)
    else:
        e_3d = df['Error_Norm_m'] if 'Error_Norm_m' in df.columns else np.zeros(len(df))
        exy = df['Error_Norm_m'] if 'Error_Norm_m' in df.columns else np.zeros(len(df))
        ez_abs = df['Error_Norm_m'] if 'Error_Norm_m' in df.columns else np.zeros(len(df))

    # ---------------------------------------------------------
    # RMS Calculation via Trapezoidal Integration
    # ---------------------------------------------------------
    time_s = df['Time_s'].values
    error_sq = e_3d.values**2 if hasattr(e_3d, 'values') else e_3d**2
    
    # Integrate E(t) over time
    error_sq_integral = np.trapezoid(error_sq, x=time_s)
    
    # Divide by total elapsed time to get the mean, then square root
    total_time = time_s[-1] - time_s[0] 
    rms_error = np.sqrt(error_sq_integral / total_time) if total_time > 0 else 0.0

    # Identify weight and control signals
    weight_cols = [col for col in df.columns if col.startswith('W')]
    has_weights = len(weight_cols) > 0
    has_control = 'Control_Output_Norm_mps2' in df.columns

    # ---------------------------------------------------------
    # FIGURE 1: Trajectory & Tracking Error (Always Generated)
    # ---------------------------------------------------------
    fig1 = plt.figure(figsize=(14, 7))
    gs = fig1.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.2)
    ax_traj = fig1.add_subplot(gs[:, 0])      
    ax_eh = fig1.add_subplot(gs[0, 1])        
    ax_ev = fig1.add_subplot(gs[1, 1])        

    # Subplot A: Top-Down Trajectory
    ax_traj.plot(df['xd'], df['yd'], 'k--', alpha=0.7, label='Desired Trajectory')
    ax_traj.plot(df['x'], df['y'], 'b-', linewidth=2, label='Actual Flight Path')
    ax_traj.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=100, zorder=5, label='Start')
    ax_traj.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', marker='X', s=120, zorder=5, label='End')
    
    # Invert Y-axis per requirements (Positive at top, Negative at bottom)
    ax_traj.set_xlim(ax_traj.get_xlim()) 
    ax_traj.set_ylim(max(df['y'].max(), df['yd'].max()) + 0.5, min(df['y'].min(), df['yd'].min()) - 0.5) 
    
    ax_traj.set_title('Top-Down Trajectory (X-Y Plane)', fontsize=12)
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.grid(True, linestyle='-', alpha=0.7)
    ax_traj.legend(loc='center right')

    # RMS Error Text Stamp
    ax_traj.text(0.05, 0.05, f'Total 3D Tracking Error RMS: {rms_error:.4f} m', 
                 transform=ax_traj.transAxes, fontsize=11, weight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Subplot B: Horizontal Error
    ax_eh.plot(df['Time_s'], exy, 'm-', linewidth=1.5)
    ax_eh.set_title(r'Horizontal Tracking Error Norm ($e_{xy}$)', fontsize=11)
    ax_eh.set_ylabel('Error (m)')
    ax_eh.grid(True)

    # Subplot C: Vertical Error
    ax_ev.plot(df['Time_s'], ez_abs, 'c-', linewidth=1.5)
    ax_ev.set_title(r'Vertical Tracking Error Norm ($e_{z}$)', fontsize=11)
    ax_ev.set_xlabel('Time (s)')
    ax_ev.set_ylabel('Error (m)')
    ax_ev.grid(True)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'trajectory_and_errors.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # ---------------------------------------------------------
    # FIGURE 2: NN Weights & Control Outputs (Conditional)
    # ---------------------------------------------------------
    # Count how many of these secondary subplots we need to render
    num_subplots = sum([has_weights, has_control])

    if num_subplots > 0:
        fig2, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharex=True if num_subplots > 1 else False)
        
        # Ensure 'axes' is an iterable array even if it's a single subplot
        if num_subplots == 1:
            axes = [axes]
            
        ax_idx = 0

        # Conditional Plotting: Weights
        if has_weights:
            axes[ax_idx].plot(df['Time_s'], df[weight_cols], alpha=0.6, linewidth=0.8)
            axes[ax_idx].set_title(f'Neural Network Weights Over Time ({len(weight_cols)} Total Signals)', fontsize=12)
            axes[ax_idx].set_ylabel('Weight Values')
            axes[ax_idx].grid(True, alpha=0.5)
            ax_idx += 1

        # Conditional Plotting: Control Output
        if has_control:
            axes[ax_idx].plot(df['Time_s'], df['Control_Output_Norm_mps2'], 'g-', linewidth=1.5)
            axes[ax_idx].set_title('Control Output Norm Over Time', fontsize=12)
            axes[ax_idx].set_ylabel(r'Control Output Norm ($m/s^2$)')
            axes[ax_idx].grid(True, alpha=0.5)
            # Label x-axis on whatever the bottom-most subplot ends up being
            axes[ax_idx].set_xlabel('Time (s)')

        # If weights was the only plot, it needs an x-axis label too
        if has_weights and not has_control:
            axes[0].set_xlabel('Time (s)')

        plt.tight_layout()
        fig2_path = os.path.join(output_dir, 'weights_and_control_outputs.png')
        plt.savefig(fig2_path, dpi=300)
        plt.close(fig2)
        print(f"Success! Figures successfully saved to:\n--> {output_dir}")
    else:
        print(f"Success! Only trajectory plots generated. (No Weight or Control column data found to plot.)\n--> {output_dir}")

if __name__ == "__main__":
    main()