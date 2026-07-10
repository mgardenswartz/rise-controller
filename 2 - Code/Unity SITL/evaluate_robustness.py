import pandas as pd
import numpy as np
from scipy import stats
import yaml
from run_sim import SimRun
import argparse

# ==========================================
# 1. PASTE YOUR BEST OPTUNA GAINS HERE
# ==========================================
BEST_RISE = {
    'controller_type': 'noresnet',
    'k_1': 0.125, 'k_2': 0.898, 'k_3': 3.647, 'k_rise': 0.977 # Replace with actual Best 1B
}

BEST_ST = {
    'controller_type': 'supertwisting',
    'k_1': 1.0, 'k_2': 2.0, 'k_3': 1.5 # Replace with actual Best Phase 3
}

BEST_NN = {
    'controller_type': 'baseline',
    # Include the fixed RISE gains from 1B here too, plus NN architecture
    'k_1': 0.125, 'k_2': 0.898, 'k_3': 3.647, 'k_rise': 0.977,
    'num_blocks': 4, 'k_0': 4, 'k_i': 4, 'hidden_width': 8,
    'gamma': 6.74, 'sigma_mod': 1.03
}

BEST_INN = {
    'controller_type': 'developed',
    # Include the fixed RISE gains from 1B here too, plus INN architecture
    'k_1': 0.125, 'k_2': 0.898, 'k_3': 3.647, 'k_rise': 0.977,
    'num_blocks': 4, 'k_0': 4, 'k_i': 4, 'hidden_width': 8,
    'gamma': 6.67, 'sigma_mod': 2.19
}

CONTROLLERS = [
    ("RISE", BEST_RISE),
    ("SuperTwisting", BEST_ST),
    ("NN_Feedforward", BEST_NN),
    ("INN_Integrated", BEST_INN)
]

def run_robustness_sweep(n_trials, config_path="config.yaml"):
    # [Keep the exact same Monte Carlo loop generation logic from the previous script]
    # It will use the n_trials passed from argparse
    pass 

def check_normality_and_compare(df):
    ALPHA = 0.05
    
    print("\n==================================================")
    print(" 1. NORMALITY CHECKS (Shapiro-Wilk, Alpha = 0.05)")
    print("==================================================")
    # Check if each individual controller's distribution is normal
    for col in df.columns:
        stat, p_val = stats.shapiro(df[col])
        is_normal = p_val > ALPHA
        status = "APPROXIMATELY NORMAL" if is_normal else "NOT NORMAL (Skewed)"
        print(f"[{col}] p-value: {p_val:.4e} -> {status}")

    print("\n==================================================")
    print(" 2. STATISTICAL SIGNIFICANCE (Alpha = 0.05)")
    print("==================================================")
    
    inn_col = 'INN_Integrated'
    baselines = ['NN_Feedforward', 'RISE', 'SuperTwisting']
    
    for base in baselines:
        # For paired tests, the assumption of normality applies to the DIFFERENCES between the pairs.
        diffs = df[inn_col] - df[base]
        diff_stat, diff_p = stats.shapiro(diffs)
        diff_is_normal = diff_p > ALPHA
        
        # Dynamically select the correct statistical test
        if diff_is_normal:
            test_name = "Paired t-test"
            stat, p_val = stats.ttest_rel(df[inn_col], df[base], alternative='less')
        else:
            test_name = "Wilcoxon Signed-Rank Test"
            stat, p_val = stats.wilcoxon(df[inn_col], df[base], alternative='less')
            
        # Determine who actually performed better (Lower ITAE Cost is better)
        inn_median = df[inn_col].median()
        base_median = df[base].median()
        
        if inn_median < base_median:
            direction = "better"
        elif inn_median > base_median:
            direction = "worse"
        else:
            direction = "exactly tied with"
            
        # Format the statistical significance string
        if p_val < ALPHA:
            sig_text = "and this WAS statistically significant"
        else:
            sig_text = "but it was NOT statistically significant"
            
        print(f"\n{inn_col} vs {base}:")
        print(f"  Test Used: {test_name} (Pairwise Difference Normality p = {diff_p:.4e})")
        print(f"  Conclusion: {inn_col} was {direction} than {base}, {sig_text} (p = {p_val:.4e}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Robustness Sweep & Statistics")
    parser.add_argument("--trials", type=int, default=50, help="Number of Monte Carlo trials to run.")
    args = parser.parse_args()

    # 1. Run the simulation sweep
    df_results = run_robustness_sweep(n_trials=args.trials)
    
    # Or, if you already have the CSV and just want to run the math:
    # df_results = pd.read_csv("robustness_results.csv")
    
    # 2. Run the math
    check_normality_and_compare(df_results)

def print_statistics(df):
    print("\n==================================================")
    print(" DESCRIPTIVE STATISTICS (Non-Normal Distribution)")
    print("==================================================")
    # ITAE costs are usually skewed, so Median and Interquartile Range (IQR) are standard
    stats_df = pd.DataFrame({
        'Median Cost': df.median(),
        'IQR': df.quantile(0.75) - df.quantile(0.25),
        'Max Cost (Worst)': df.max(),
        'Min Cost (Best)': df.min()
    })
    print(stats_df.to_string())

    print("\n==================================================")
    print(" WILCOXON SIGNED-RANK TEST (Alpha = 0.05)")
    print("==================================================")
    
    # Compare INN against NN
    stat, p_val = stats.wilcoxon(df['INN_Integrated'], df['NN_Feedforward'], alternative='less')
    print(f"INN vs NN Feedforward:")
    print(f"  p-value = {p_val:.4e}")
    if p_val < 0.05:
        print("  RESULT: INN is STATISTICALLY SIGNIFICANTLY better than NN.")
    else:
        print("  RESULT: No significant difference between INN and NN.")

    # Compare INN against RISE
    stat, p_val_rise = stats.wilcoxon(df['INN_Integrated'], df['RISE'], alternative='less')
    print(f"\nINN vs RISE Baseline:")
    print(f"  p-value = {p_val_rise:.4e}")
    if p_val_rise < 0.05:
         print("  RESULT: INN is STATISTICALLY SIGNIFICANTLY better than RISE.")

if __name__ == "__main__":
    df_results = run_robustness_sweep(n_trials=50)
    print_statistics(df_results)