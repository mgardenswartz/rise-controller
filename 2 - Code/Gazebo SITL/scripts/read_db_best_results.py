import argparse
import os
import optuna

# --- Argparse Setup ---
parser = argparse.ArgumentParser(description="Load an Optuna study from a specific .db file.")
parser.add_argument(
    "db_file", 
    type=str,
    help="The path to the Optuna .db file (e.g., resnet.db)"
)
args = parser.parse_args()

# --- Dynamic Configuration ---
# Ensure the file exists (optional, but good practice)
if not os.path.exists(args.db_file):
    parser.error(f"The file '{args.db_file}' does not exist.")

# Construct the DB URL and extract the study name from the filename
db_url = f"sqlite:///{args.db_file}"
study_name = os.path.splitext(os.path.basename(args.db_file))[0]

# --- Verification (Optional) ---
print(f"Loading study '{study_name}' from {db_url}")
try:
    # Inspect the database to see what studies actually exist inside it
    study_summaries = optuna.get_all_study_summaries(storage=db_url)
    
    if not study_summaries:
        print(f"[!] The database at {db_url} contains no studies.")
        exit()
        
    # Warn if there's more than one, but default to the first one found
    if len(study_summaries) > 1:
        print(f"⚠️  [Warning]: Multiple studies found in this DB ({len(study_summaries)} total).")
        print(f"    Loading the first one: '{study_summaries[0].study_name}'")
    else:
        print(f"Loading study: '{study_summaries[0].study_name}'")
        
    # Target the actual name stored in the DB
    study_name = study_summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=db_url)

except Exception as e:
    print(f"[!] Could not load study from {db_url}. Error: {e}")
    exit()

# 1. Get the absolute best trial
print("\n" + "="*40)
print("🏆 ABSOLUTE BEST TRIAL 🏆")
print("="*40)
best_trial = study.best_trial
print(f"Trial Number : {best_trial.number}")
print(f"ITAE Cost    : {best_trial.value:.4f}")
print("Parameters   :")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
print("User Attrs   :")
for key, value in best_trial.user_attrs.items():
    print(f"    {key}: {value}")

# 2. Sort all completed, valid trials by value
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
completed_trials.sort(key=lambda t: t.value)

# --- Determine Table Layout & User Attrs Dynamically ---
has_rms = False
use_nn_layout = False

if completed_trials:
    sample_trial = completed_trials[0]
    
    # Check for RMS attributes
    sample_attrs = sample_trial.user_attrs
    if "error_rms" in sample_attrs or "control_effort_rms" in sample_attrs:
        has_rms = True
        
    # Dynamically determine layout by checking which parameters exist in the database
    if "num_blocks" in sample_trial.params or "hidden_width" in sample_trial.params:
        use_nn_layout = True

# 3. Print Top 15 Trials in a Formatted Table
print("\n" + "="*95)

# Generate table headers dynamically based on detected parameters
if use_nn_layout:
    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'num_blocks':<10} | {'hidden_width':<12} | {'k_0':<6} | {'k_i':<6} | {'gamma':<7} | {'sigma_mod':<9} | {'W_s':<6}"
else:
    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'k_1':<7} | {'k_2':<7} | {'k_3':<7} | {'k_rise':<9}"

# Append extra columns if RMS data is tracked
if has_rms:
    header_str += f" | {'Err RMS':<9} | {'Eff RMS':<9}"

print(header_str)
print("-" * len(header_str))

# Print rows
for i, trial in enumerate(completed_trials[:15]):
    cost = trial.value
    
    # NN Controller Layout (Formerly resnet/integrated_resnet)
    if use_nn_layout:
        nb = trial.params.get("num_blocks", 0)
        hw = trial.params.get("hidden_width", 0)
        k0 = trial.params.get("k_0", 0)
        ki = trial.params.get("k_i", 0)
        g  = trial.params.get("gamma", 0)
        sm = trial.params.get("sigma_mod", 0)
        ws = trial.params.get("initial_weight_scale_factor", 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {cost:<9.3g} | {nb:<10} | " \
                  f"{hw:<12} | {k0:<6.3g} | {ki:<6.3g} | {g:<7.3g} | {sm:<9.3g} | {ws:<6.3g}"
    # Alternate Controller Layout (Formerly baseline/supertwisting)
    else:
        k_1 = trial.params.get('k_1', 0)
        k_2 = trial.params.get('k_2', 0)
        k_3 = trial.params.get('k_3', 0)
        krise = trial.params.get('k_rise', 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {cost:<9.3g} | {k_1:<7.3g} | " \
                  f"{k_2:<7.3g} | {k_3:<7.3g} | {krise:<9.3g}"
    
    # Safely append extra columns if found
    if has_rms:
        err_rms = trial.user_attrs.get("error_rms", 0)
        eff_rms = trial.user_attrs.get("control_effort_rms", 0)
        row_str += f" | {err_rms:<9.3g} | {eff_rms:<9.3g}"
        
    print(row_str)