import argparse
import pandas as pd
import numpy as np
from scipy import stats
import yaml
from typing import Any, Tuple, List, Dict
import os
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from src.run_sim import SimRun

def check_normality_and_compare(df: pd.DataFrame, metric: str) -> None:
    ALPHA = 0.05
    print("\n==================================================")
    print(f" 1. NORMALITY CHECKS (Shapiro-Wilk) - {metric}")
    print("==================================================")
    for col in df.columns:
        if metric not in col:
            continue
        stat, p_val = stats.shapiro(df[col])
        is_normal = p_val > ALPHA
        status = "APPROXIMATELY NORMAL" if is_normal else "NOT NORMAL (Skewed)"
        print(f"[{col}] p-value: {p_val:.4e} -> {status}")

    print("\n==================================================")
    print(f" 2. STATISTICAL SIGNIFICANCE - {metric}")
    print("==================================================")
    
    inn_col = f'INN_Integrated_{metric}'
    baselines = [f'NN_Feedforward_{metric}', f'RISE_{metric}', f'SuperTwisting_{metric}', f'PID_{metric}']
    
    for base in baselines:
        if base not in df.columns or inn_col not in df.columns:
            continue
        diffs = df[inn_col] - df[base]
        diff_stat, diff_p = stats.shapiro(diffs)
        diff_is_normal = diff_p > ALPHA
        
        if diff_is_normal:
            test_name = "Paired t-test"
            stat, p_val = stats.ttest_rel(df[inn_col], df[base], alternative='less')
        else:
            test_name = "Wilcoxon Signed-Rank Test"
            stat, p_val = stats.wilcoxon(df[inn_col], df[base], alternative='less')
            
        inn_median = df[inn_col].median()
        base_median = df[base].median()
        
        if inn_median < base_median:
            direction = "better"
        elif inn_median > base_median:
            direction = "worse"
        else:
            direction = "exactly tied with"
            
        if p_val < ALPHA:
            sig_text = "and this WAS statistically significant"
        else:
            sig_text = "but it was NOT statistically significant"
            
        print(f"\n{inn_col} vs {base}:")
        print(f"  Test Used: {test_name} (Pairwise Difference Normality p = {diff_p:.4e})")
        print(f"  Conclusion: {inn_col} was {direction} than {base}, {sig_text} (p = {p_val:.4e}).")

def print_statistics(df: pd.DataFrame, metric: str) -> None:
    print("\n==================================================")
    print(f" DESCRIPTIVE STATISTICS - {metric}")
    print("==================================================")
    
    metric_cols = [col for col in df.columns if metric in col]
    stats_df = pd.DataFrame({
        f'Median {metric}': df[metric_cols].median(),
        'IQR': df[metric_cols].quantile(0.75) - df[metric_cols].quantile(0.25),
        f'Max {metric} (Worst)': df[metric_cols].max(),
        f'Min {metric} (Best)': df[metric_cols].min()
    })
    
    stats_df.index = [idx.replace(f"_{metric}", "") for idx in stats_df.index]
    print(stats_df.to_string())

from quadsim import QuadSim
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file

def run_robustness_sweep(n_trials: int, config_path: str, controllers: List[Tuple[str, Dict[str, Any]]], output_csv: str, runner: LockstepRunner, px4: Px4Link) -> pd.DataFrame:
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        base_config = full_config['aviary_rise_node']['ros__parameters']
        
    base_desired_traj = base_config['desired_trajectory']
    if base_desired_traj == 1:
        base_x = base_config['traj1_init_x_m_ned']
        base_y = base_config['traj1_init_y_m_ned']
    else:
        base_x = base_config['traj2_init_x_m_ned']
        base_y = base_config['traj2_init_y_m_ned']
    base_z = base_config['hover_start_z_m_ned']
    
    xy_range = base_config['xy_rand_range_m']
    z_range = base_config['z_rand_range_m']
    
    results: Dict[str, List[float]] = {}
    for name, _ in controllers:
        results[f"{name}_Cost"] = []
        results[f"{name}_e_RMS"] = []
        results[f"{name}_u_RMS"] = []
    
    traj1_fixed = [(1.5, 5.5), (1.5, 2.5), (-1.5, 5.5), (-1.5, 2.5)]
    traj2_fixed = [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5)]
    
    print(f"[*] Starting Monte Carlo Sweep ({n_trials} trials per controller)...")
    
    for i in range(n_trials):
        np.random.seed(100 + i) 
        
        if i < 4:
            if base_desired_traj == 1:
                trial_x, trial_y = traj1_fixed[i]
            else:
                trial_x, trial_y = traj2_fixed[i]
            trial_z = base_config['init_z_m_ned']
        else:
            trial_x = base_x + np.random.uniform(-xy_range, xy_range)
            trial_y = base_y + np.random.uniform(-xy_range, xy_range)
            trial_z = base_config['init_z_m_ned'] + np.random.uniform(-z_range, z_range)
        
        print(f"\n--- Trial {i+1}/{n_trials} | Spawn: ({trial_x:.2f}, {trial_y:.2f}, {trial_z:.2f}) ---")
        
        for name, params in controllers:
            trial_params = params.copy()
            trial_params['init_x_m_ned'] = trial_x
            trial_params['init_y_m_ned'] = trial_y
            trial_params['hover_start_z_m_ned'] = trial_z
            
            print(f"[reset] PX4 position control -> ({trial_x:.2f}, {trial_y:.2f}, {trial_z:.2f})")
            if not (px4.is_armed() and px4.in_offboard()):
                raise RuntimeError("vehicle left armed OFFBOARD during a trial")
            
            yaw_rad = 0.0 # simplified
            if not runner.fly_to_ned(
                trial_x, trial_y, trial_z, yaw_ned=yaw_rad,
                tol=0.20, vel_tol=0.50, settle_s=1.0, timeout_s=60.0,
            ):
                raise RuntimeError("PX4 could not recover to start position.")
                
            sim = SimRun(trial_params, yaml_config_path=config_path, runner=runner, px4=px4)
            cost, e_rms, u_rms = sim.run()
            results[f"{name}_Cost"].append(cost)
            results[f"{name}_e_RMS"].append(e_rms)
            results[f"{name}_u_RMS"].append(u_rms)
            print(f" > {name}: ITAE = {cost:.2f} | e_RMS = {e_rms:.4f} | u_RMS = {u_rms:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n[*] Sweep complete. Data saved to '{output_csv}'.")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Robustness Sweep & Statistics")
    parser.add_argument("--num_trials", type=int, required=True, help="Number of Monte Carlo trials to run.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--db_dir", type=str, required=True, help="Directory containing best_gains.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config_yaml = yaml.safe_load(f)
        full_config = full_config_yaml['aviary_rise_node']['ros__parameters']

    best_gains_path = os.path.join(args.db_dir, "best_gains.yaml")
    if not os.path.exists(best_gains_path):
        raise FileNotFoundError(f"Best gains file not found at {best_gains_path}. Please run extract_gains.py first.")
        
    with open(best_gains_path, 'r') as f:
        best_gains = yaml.safe_load(f)
        
    # Format into controllers list
    controllers: List[Tuple[str, Dict[str, Any]]] = []
    if 'BEST_RISE' in best_gains:
        controllers.append(("RISE", best_gains['BEST_RISE']))
    if 'BEST_ST' in best_gains:
        controllers.append(("SuperTwisting", best_gains['BEST_ST']))
    if 'BEST_PID' in best_gains:
        controllers.append(("PID", best_gains['BEST_PID']))
    if 'BEST_NN' in best_gains:
        controllers.append(("NN_Feedforward", best_gains['BEST_NN']))
    if 'BEST_INN' in best_gains:
        controllers.append(("INN_Integrated", best_gains['BEST_INN']))
        
    if not controllers:
        raise ValueError("No controllers found in best_gains.yaml")

    robustness_output_path = os.path.join(args.db_dir, f"robustness_sweep_{len(controllers)}_controllers.csv")
    summary_output_path = os.path.join(args.db_dir, f"robustness_sweep_{len(controllers)}_controllers_summary.md")
    
    # Redirect stdout to capture prints
    sys.stdout = Logger(summary_output_path)
    print(f"# Robustness Evaluation Summary\n")

    # Initialize PX4
    full_config = yaml.safe_load(open(args.config, 'r'))
    px4_config = full_config['px4']
    quadsim_config = full_config['quadsim']
    aviary_config = full_config['aviary_rise_node']['ros__parameters']

    sim = QuadSim(
        host=quadsim_config["host"],
        command_port=quadsim_config["port"],
        telemetry_port=quadsim_config["port"] + 1,
    )
    sim.connect()
    bridge = HilBridge(
        px4_host=px4_config["host"],
        instance=px4_config["instance"],
        px4_client=px4_config["client"],
    )
    px4 = Px4Link(stream_hz=0, instance=px4_config["instance"])
    runner = LockstepRunner(
        sim,
        bridge,
        px4,
        control_hz=aviary_config["control_frequency_hz"],
    )
    speed = aviary_config["sim_speed"]
    runner.speed_cap = speed if speed > 0.0 else None

    runner.start()
    runner.wait_px4()
    param_file = os.path.join("conf", px4_config["param_file"])
    load_param_file(px4, param_file)
    px4.configure_offboard_no_rc()
    if not runner.wait_ekf_ready() or not runner.wait_heading():
        raise RuntimeError("PX4 estimator is not ready")

    try:
        ground_ned = np.asarray(runner.ground_ned, dtype=float)
        px4.goto_ned(ground_ned[0], ground_ned[1], ground_ned[2] - 1.0, yaw_ned=0.0)
        px4.emit_setpoint_now()
        if not runner.engage_offboard(arm=True):
            raise RuntimeError("PX4 refused OFFBOARD or arm")

        df_results = run_robustness_sweep(
            n_trials=args.num_trials, 
            config_path=args.config, 
            controllers=controllers, 
            output_csv=robustness_output_path,
            runner=runner,
            px4=px4
        )
        
        for metric in ["e_RMS", "u_RMS", "Cost"]:
            print_statistics(df_results, metric)
            check_normality_and_compare(df_results, metric)
            
    except KeyboardInterrupt:
        print("\n[*] Robustness sweep interrupted.")
    except Exception as e:
        print(f"\n[!] Robustness sweep failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if px4 and px4.is_armed():
                print("\nLanding drone before exit...")
                ground = runner.ground_ned
                runner.fly_to_ned(
                    ground[0], ground[1], ground[2] - 0.08,
                    yaw_ned=0.0, tol=0.12, vel_tol=0.25, settle_s=1.5, timeout_s=45.0,
                )
                runner.disarm()
        except KeyboardInterrupt:
            print("\n[!] Force quitting during landing.")
        except Exception as e:
            print(f"\n[!] Error during landing: {e}")
            
        if runner:
            runner.close()
        if sim:
            sim.resume()
            sim.disconnect()