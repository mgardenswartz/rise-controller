import argparse
import optuna

# --- Argparse Setup ---
parser = argparse.ArgumentParser(description="Load Optuna studies based on controller type.")
parser.add_argument(
    "controller_type", 
    choices=["baseline", "developed", "noresnet"],
    help="The type of controller study to load."
)
args = parser.parse_args()

# --- Configuration Mapping ---
CONFIGS = {
    "noresnet": {
        "db_url": "sqlite:///phase1_tuning.db",
        "study_name": "phase1_noresnet_baseline_tuning"
    },
    "baseline": {
        "db_url": "sqlite:///phase2_baseline_wind.db",
        "study_name": "phase2_baseline_wind_optimization"
    },
    "developed": {
        "db_url": "sqlite:///phase2_developed_wind.db",
        "study_name": "phase2_developed_wind_optimization"
    }
}

current_config = CONFIGS[args.controller_type]
db_url = current_config["db_url"]
study_name = current_config["study_name"]

# --- Main Logic ---
print(f"[*] Loading {args.controller_type} study '{study_name}' from {db_url}...")

try:
    study = optuna.load_study(study_name=study_name, storage=db_url)
except Exception as e:
    print(f"[!] Could not load study. Are you sure the name is correct? Error: {e}")
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

# --- Determine if User Attrs Exist ---
# Check the first completed trial to see if RMS attributes are present
has_rms = False
if completed_trials:
    sample_attrs = completed_trials[0].user_attrs
    # Adjust string keys below if your script saved them under slightly different names
    if "error_rms" in sample_attrs or "control_effort_rms" in sample_attrs:
        has_rms = True

# 3. Print Top 15 Trials in a Formatted Table
print("\n" + "="*95)

# Generate table headers dynamic framework
if args.controller_type in ["baseline", "developed"]:
    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'num_blocks':<10} | {'hidden_width':<12} | {'k_0':<6} | {'k_i':<6} | {'gamma':<7} | {'sigma_mod':<9} | {'W_s':<6}"
else:
    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'k1':<7} | {'k2':<7} | {'k3':<7} | {'k_rise':<9}"

# Append extra columns if RMS data is tracked
if has_rms:
    header_str += f" | {'Err RMS':<9} | {'Eff RMS':<9}"

print(header_str)
print("-" * len(header_str))

# Print rows
for i, trial in enumerate(completed_trials[:15]):
    cost = trial.value
    
    # Base controller string tracking
    if args.controller_type in ["baseline", "developed"]:
        nb = trial.params.get("num_blocks", 0)
        hw = trial.params.get("hidden_width", 0)
        k0 = trial.params.get("k_0", 0)
        ki = trial.params.get("k_i", 0)
        g  = trial.params.get("gamma", 0)
        sm = trial.params.get("sigma_mod", 0)
        ws = trial.params.get("initial_weight_scale_factor", 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {cost:<9.3g} | {nb:<10} | {hw:<12} | {k0:<6.3g} | {ki:<6.3g} | {g:<7.3g} | {sm:<9.3g} | {ws:<6.3g}"
    else:
        k1 = trial.params.get('k1', 0)
        k2 = trial.params.get('k2', 0)
        k3 = trial.params.get('k3', 0)
        krise = trial.params.get('k_rise', 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {cost:<9.3g} | {k1:<7.3g} | {k2:<7.3g} | {k3:<7.3g} | {krise:<9.3g}"
    
    # Safely append extra columns if found
    if has_rms:
        err_rms = trial.user_attrs.get("error_rms", 0)
        eff_rms = trial.user_attrs.get("control_effort_rms", 0)
        row_str += f" | {err_rms:<9.3g} | {eff_rms:<9.3g}"
        
    print(row_str)

print("=" * len(header_str) + "\n")