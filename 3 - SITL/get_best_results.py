import optuna

# --- Configuration ---
db_url = "sqlite:///phase1_tuning.db"
# CRITICAL: Change this to whatever you named your new warped trajectory study!
study_name = "phase1_noresnet_baseline_tuning" 

print(f"[*] Loading study '{study_name}' from {db_url}...")

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

# 2. Print Top 5 Trials in a Formatted Table (3 Sig Figs)
print("\n" + "="*65)
print(f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'k1':<7} | {'k2':<7} | {'k3':<7} | {'k_rise':<9}")
print("-" * 65)

# Sort all completed, valid trials by value
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
completed_trials.sort(key=lambda t: t.value)

for i, trial in enumerate(completed_trials[:15]):
    # Extract values
    cost = trial.value
    k1 = trial.params.get('k1', 0)
    k2 = trial.params.get('k2', 0)
    k3 = trial.params.get('k3', 0)
    krise = trial.params.get('k_rise', 0)
    
    # Format to 3 significant figures using the 'g' specifier
    print(f"{i+1:<6} | {trial.number:<7} | {cost:<9.3g} | {k1:<7.3g} | {k2:<7.3g} | {k3:<7.3g} | {krise:<9.3g}")

print("="*65 + "\n")