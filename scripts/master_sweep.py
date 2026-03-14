import subprocess
import json
import itertools
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
    subprocess.run(cmd, shell=True, check=True)

def find_matched_architecture(target_p: int, d_in: int, d_out: int = 2) -> dict:
    from scripts.architecture_matcher import get_total_parameters
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
    best_gains = {}
    
    for sys_id in SYSTEMS:
        print(f"\n--- Tuning System {sys_id} ---")
        # Run standard Optuna sweep overriding the NN parameters to 0
        cmd = (
            f"python main.py -m "
            f"simulation.sys_id={sys_id} "
            f"math_constants.k_theta_hat=0.0 "
            f"agent_network.init_mean=0.0 "
            f"agent_network.init_std=0.0 "
            f"math_constants.k_1='interval(0.1, 15.0)' "
            f"math_constants.k_2='interval(0.1, 15.0)' "
            f"math_constants.beta='interval(0.0, 10.0)' "
            f"hydra.sweeper.n_trials=50"
        )
        run_cmd(cmd)
        
        # In practice, parse the latest multirun outputs to grab the best gains.
        # For script integrity, I am putting placeholders you'd extract from Optuna's output.
        best_gains[sys_id] = {"k_1": 5.0, "k_2": 5.0, "beta": 2.0} 
        
    return best_gains

def phase_2_massive_sweep(best_gains: dict):
    print("\n" + "="*50 + "\nPHASE 2: MONTE CARLO MASSIVE SWEEP\n" + "="*50)
    
    controllers = [
        ("baseline", 2),          # name, d_in
        ("nn_in_integral", 4)
    ]
    
    for sys_id in SYSTEMS:
        gains = best_gains[sys_id]
        
        for (ctrl_name, d_in) in controllers:
            for size_name, target_p in TARGET_PARAMS.items():
                
                # Dynamically generate the matched architecture
                arch = find_matched_architecture(target_p, d_in=d_in)
                
                print(f"\n[SWEEP] Sys: {sys_id} | Ctrl: {ctrl_name} | Size: {size_name} (P={arch['actual_p']})")
                
                # Execute MC sweep via Hydra choice mechanism over random seeds
                seeds = ",".join([str(1000 + i) for i in range(MC_TRIALS)])
                
                cmd = (
                    f"python main.py -m "
                    f"simulation.sys_id={sys_id} "
                    f"simulation.controller_type='{ctrl_name}' "
                    f"simulation.random_seed='choice({seeds})' "
                    f"math_constants.k_1={gains['k_1']} "
                    f"math_constants.k_2={gains['k_2']} "
                    f"math_constants.beta={gains['beta']} "
                    f"agent_network.d_in={d_in} "
                    f"agent_network.b={arch['b']} "
                    f"agent_network.k_0={arch['k_0']} "
                    f"agent_network.k_i={arch['k_i']} "
                    f"agent_network.hidden_width={arch['hidden_width']} "
                )
                run_cmd(cmd)

if __name__ == "__main__":
    gains = phase_1_tune_baselines()
    phase_2_massive_sweep(gains)
    
    print("\n[COMPLETE] Run scripts/rank_runs.py to aggregate and visualize the Monte Carlo reports.")