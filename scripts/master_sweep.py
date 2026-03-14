import subprocess
import argparse
import os
from pathlib import Path

# --- EXPERIMENT SETTINGS ---
SYSTEMS = [1, 2, 3]
MC_TRIALS = 20
TARGET_PARAMS = {
    "small": 150,
    "medium": 500,
    "large": 1500
}

def run_cmd(cmd: str):
    print(f"\n[EXEC] {cmd}")
    
    # Clone the current environment and disable JAX's aggressive memory preallocation
    # This is critical for Apple Silicon (M-series) unified memory when running n_jobs > 1
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    subprocess.run(cmd, shell=True, check=True, env=env)

def find_matched_architecture(target_p: int, d_in: int, d_out: int = 2) -> dict:
    from architecture_matcher import get_total_parameters
    best_diff = float('inf')
    best = {}
    for w in range(2, 64):
        for b in range(0, 10):
            for k_0 in range(1, 4):
                for k_i in range(1, 4):
                    p = get_total_parameters(d_in, w, d_out, b, k_0, k_i)
                    if abs(p - target_p) < best_diff:
                        best_diff = abs(p - target_p)
                        best = {"b": b, "k_0": k_0, "k_i": k_i, "hidden_width": w, "actual_p": p}
                    if best_diff == 0:
                        return best
    return best

def phase_1_tune_baselines():
    print("="*50 + "\nPHASE 1: TUNING BASELINE ROBUST GAINS\n" + "="*50)
    
    for sys_id in SYSTEMS:
        print(f"\n--- Tuning System {sys_id} ---")
        cmd = (
            f"python main.py -m "
            f"simulation.sys_id={sys_id} "
            f"math_constants.k_theta_hat=0.0 "
            f"neural_network.init_mean=0.0 "
            f"neural_network.init_std=0.0 "
            f"math_constants.k_1='interval(0.1, 15.0)' "
            f"math_constants.k_2='interval(0.1, 15.0)' "
            f"math_constants.beta='interval(0.0, 10.0)' "
            f"hydra.sweeper.n_trials=50 "
            f"hydra.sweeper.n_jobs=2" # Parallelize Optuna trials
        )
        run_cmd(cmd)
        
    print("\n[PHASE 1 COMPLETE] Check your multirun logs to find the best gains.")
    print("Update the 'HARDCODED_GAINS' dictionary in this script before running Phase 2.")

def phase_2_massive_sweep(gains_dict: dict):
    print("\n" + "="*50 + "\nPHASE 2: MONTE CARLO MASSIVE SWEEP\n" + "="*50)
    
    controllers = [
        ("baseline", 2),          
        ("nn_in_integral", 4)
    ]
    
    for sys_id in SYSTEMS:
        # Failsafe if you forgot to update the gains dictionary
        if sys_id not in gains_dict:
            print(f"Skipping System {sys_id}: No gains defined in dictionary.")
            continue
            
        gains = gains_dict[sys_id]
        
        for (ctrl_name, d_in) in controllers:
            for size_name, target_p in TARGET_PARAMS.items():
                
                arch = find_matched_architecture(target_p, d_in=d_in)
                print(f"\n[SWEEP] Sys: {sys_id} | Ctrl: {ctrl_name} | Size: {size_name} (P={arch['actual_p']})")
                
                seeds = ",".join([str(1000 + i) for i in range(MC_TRIALS)])
                
                cmd = (
                    f"python main.py -m "
                    f"simulation.sys_id={sys_id} "
                    f"simulation.controller_type='{ctrl_name}' "
                    f"simulation.random_seed='choice({seeds})' "
                    f"math_constants.k_1={gains['k_1']} "
                    f"math_constants.k_2={gains['k_2']} "
                    f"math_constants.beta={gains['beta']} "
                    f"neural_network.d_in={d_in} "
                    f"neural_network.b={arch['b']} "
                    f"neural_network.k_0={arch['k_0']} "
                    f"neural_network.k_i={arch['k_i']} "
                    f"neural_network.hidden_width={arch['hidden_width']} "
                    f"hydra.sweeper.n_jobs=2" # Parallelize Monte Carlo trials
                )
                run_cmd(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Orchestrator for the RISE ResNet Control Sweep")
    parser.add_argument("--tune", action="store_true", help="Run Phase 1: Tune baseline robust gains with NN off.")
    parser.add_argument("--sweep", action="store_true", help="Run Phase 2: Execute the massive Monte Carlo sweep.")
    args = parser.parse_args()

    # --- INPUT YOUR OPTUNA RESULTS HERE AFTER RUNNING --tune ---
    HARDCODED_GAINS = {
        1: {"k_1": 5.0, "k_2": 5.0, "beta": 2.0},
        2: {"k_1": 5.0, "k_2": 5.0, "beta": 2.0},
        3: {"k_1": 5.0, "k_2": 5.0, "beta": 2.0}
    }

    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample usage:\n  python scripts/master_sweep.py --tune\n  python scripts/master_sweep.py --sweep")
        exit()

    if args.tune:
        phase_1_tune_baselines()
        
    if args.sweep:
        phase_2_massive_sweep(HARDCODED_GAINS)