import subprocess
import time

import os
import subprocess
import time

def run_study(controller_type, study_name):
    print(f"\n{'='*50}")
    print(f"[*] STARTING PHASE 2: {controller_type.upper()} CONTROLLER")
    print(f"[*] Study Name: {study_name}")
    print(f"{'='*50}\n")
    
    # 1. Copy the current environment (preserves venv and PYTHONPATH)
    env = os.environ.copy()
    
    # 2. Inject the custom variables
    env["CONTROLLER_TYPE"] = controller_type
    env["STUDY_NAME"] = study_name
    
    try:
        # Run the orchestrator with the preserved environment
        subprocess.run(["python3", "unified_orchestrator.py"], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] ERROR: {controller_type} run failed: {e}")
        exit(1)

if __name__ == "__main__":
    # 1. Run the Baseline Phase 2
    run_study(controller_type="baseline", study_name="phase2_baseline_tuning")
    
    print("\n[*] Baseline complete. Cooling down CPU for 60 seconds...")
    time.sleep(60)
    
    # 2. Run the Developed Phase 2
    run_study(controller_type="developed", study_name="phase2_developed_tuning")
    
    print("\n[*] ALL PHASE 2 RUNS COMPLETED SUCCESSFULLY.")