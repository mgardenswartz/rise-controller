import argparse
import pandas as pd
import numpy as np
from scipy import stats
import yaml
from typing import Any, Tuple, List, Dict
import os

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
    baselines = [f'NN_Feedforward_{metric}', f'RISE_{metric}', f'SuperTwisting_{metric}']
    
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

def run_robustness_sweep(n_trials: int, config_path: str, controllers: List[Tuple[str, Dict[str, Any]]], output_csv: str) -> pd.DataFrame:
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
        
    base_x = base_config.get('init_x_m_ned_aviary', 0.70)
    base_y = base_config.get('init_y_m_ned_aviary', -2.37)
    base_z = base_config.get('hover_start_z_m_ned_aviary', -1.5)
    
    xy_range = base_config.get('xy_rand_range_m', 1.0)
    z_range = base_config.get('z_rand_range_m', 0.0)
    
    results: Dict[str, List[float]] = {}
    for name, _ in controllers:
        results[f"{name}_Cost"] = []
        results[f"{name}_e_RMS"] = []
        results[f"{name}_u_RMS"] = []
    
    print(f"[*] Starting Monte Carlo Sweep ({n_trials} trials per controller)...")
    
    for i in range(n_trials):
        np.random.seed(100 + i) 
        trial_x = base_x + np.random.uniform(-xy_range, xy_range)
        trial_y = base_y + np.random.uniform(-xy_range, xy_range)
        trial_z = base_z + np.random.uniform(-z_range, z_range)
        
        print(f"\n--- Trial {i+1}/{n_trials} | Spawn: ({trial_x:.2f}, {trial_y:.2f}, {trial_z:.2f}) ---")
        
        for name, params in controllers:
            trial_params = params.copy()
            trial_params['init_x_m_ned_aviary'] = trial_x
            trial_params['init_y_m_ned_aviary'] = trial_y
            trial_params['hover_start_z_m_ned_aviary'] = trial_z
            
            sim = SimRun(trial_params, yaml_config_path=config_path)
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
        full_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
        
    best_gains_path = os.path.join(args.db_dir, "best_gains.yaml")
    robustness_output_path = os.path.join(args.db_dir, "robustness_results.csv")
    
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
    if 'BEST_NN' in best_gains:
        controllers.append(("NN_Feedforward", best_gains['BEST_NN']))
    if 'BEST_INN' in best_gains:
        controllers.append(("INN_Integrated", best_gains['BEST_INN']))
        
    if not controllers:
        raise ValueError("No controllers found in best_gains.yaml")

    df_results = run_robustness_sweep(
        n_trials=args.num_trials, 
        config_path=args.config, 
        controllers=controllers, 
        output_csv=robustness_output_path
    )
    
    for metric in ["e_RMS", "u_RMS", "Cost"]:
        print_statistics(df_results, metric)
        check_normality_and_compare(df_results, metric)