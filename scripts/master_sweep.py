import subprocess
import argparse
import os
from pathlib import Path

# Imports for Phase 2 in-process JAX execution
import jax
from hydra import initialize, compose
import yaml
import dataclasses

# Adjust these imports if your core logic schemas are named differently
from src.conf.config_schema import (
    ExperimentConfig, DirectoriesConfig, SimulationConfig,
    MathConstantsConfig, NeuralNetworkConfig, DataLabelsConfig,
    PlotSettingsConfig, AnimationConfig
)
from src.simulation.runner import run_simulation
from src.io.statistics import calculate_and_save_statistics

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
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    subprocess.run(cmd, shell=True, check=True, env=env)

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
    
    for sys_id in SYSTEMS:
        print(f"\n--- Tuning System {sys_id} ---")
        cmd = (
            f"python main.py -m "
            f"simulation.sys_id={sys_id} "
            f"simulation.controller_type='baseline' "
            f"neural_network.d_in=2 "
            f"math_constants.k_theta_hat=0.0 "
            f"neural_network.init_mean=0.0 "
            f"neural_network.init_std=0.0 "
            f"math_constants.k_1='interval(0.1, 15.0)' "
            f"math_constants.k_2='interval(0.1, 15.0)' "
            f"math_constants.beta='interval(0.0, 10.0)' "
            f"hydra.sweeper.n_trials=50 "
            f"hydra.sweeper.n_jobs=2" 
        )
        run_cmd(cmd)
        
    print("\n[PHASE 1 COMPLETE] Check your multirun logs to find the best gains.")
    print("Update the 'HARDCODED_GAINS' dictionary in this script before running Phase 2.")

def build_config(sys_id, ctrl_name, seed, gains, arch, d_in):
    """Uses Hydra's Compose API to build the configuration dynamically."""
    # Assuming this script is run from the root directory or scripts/ directory
    # Adjust config_path if needed (e.g., "conf" if run from root)
    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="config")
    except Exception:
        # Fallback if executed from project root rather than inside /scripts
        with initialize(version_base=None, config_path="conf"):
            cfg = compose(config_name="config")
        
    config = ExperimentConfig(
        directories=DirectoriesConfig(**cfg.directories),
        simulation=SimulationConfig(**cfg.simulation),
        math_constants=MathConstantsConfig(**cfg.math_constants),
        neural_network=NeuralNetworkConfig(**cfg.neural_network),
        data_labels=DataLabelsConfig(**cfg.data_labels),
        plot_settings=PlotSettingsConfig(**cfg.plot_settings),
        animation=AnimationConfig(**cfg.animation)
    )
    
    # Apply Monte Carlo and Sweep Overrides
    config.simulation.sys_id = sys_id
    config.simulation.controller_type = ctrl_name
    config.simulation.randomize_x0 = True
    config.simulation.random_seed = seed
    
    config.math_constants.k_1 = gains["k_1"]
    config.math_constants.k_2 = gains["k_2"]
    config.math_constants.beta = gains["beta"]
    
    config.neural_network.d_in = d_in
    config.neural_network.b = arch["b"]
    config.neural_network.k_0 = arch["k_0"]
    config.neural_network.k_i = arch["k_i"]
    config.neural_network.hidden_width = arch["hidden_width"]
    
    return config

def phase_2_massive_sweep(gains_dict: dict):
    print("\n" + "="*50 + "\nPHASE 2: MONTE CARLO MASSIVE SWEEP (IN-PROCESS JAX)\n" + "="*50)
    
    controllers = [
        ("baseline", 2),          
        ("nn_in_integral", 4)
    ]
    
    base_output_dir = Path("outputs/massive_sweep")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for sys_id in SYSTEMS:
        if sys_id not in gains_dict:
            print(f"Skipping System {sys_id}: No gains defined in dictionary.")
            continue
            
        gains = gains_dict[sys_id]
        
        for (ctrl_name, d_in) in controllers:
            for size_name, target_p in TARGET_PARAMS.items():
                
                arch = find_matched_architecture(target_p, d_in=d_in)
                print(f"\n[COMPILING & RUNNING] Sys: {sys_id} | Ctrl: {ctrl_name} | Size: {size_name} (P={arch['actual_p']})")
                
                for i in range(MC_TRIALS):
                    seed = 1000 + i
                    config = build_config(sys_id, ctrl_name, seed, gains, arch, d_in)
                    
                    # Create structured output directories for the aggregator
                    run_dir = base_output_dir / f"sys_{sys_id}" / ctrl_name / f"p_{arch['actual_p']}" / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    hydra_dir = run_dir / ".hydra"
                    hydra_dir.mkdir(exist_ok=True)
                    
                    try:
                        if i == 0:
                            print(f"  -> Trial {i+1}/{MC_TRIALS} (JIT Compiling... this takes a moment)")
                        else:
                            print(f"  -> Trial {i+1}/{MC_TRIALS} (Running from JIT cache...)")
                            
                        sim_data = run_simulation(config)
                        calculate_and_save_statistics(sim_data, run_dir, config)
                        
                        # Dump config mimicking Hydra's output for the aggregator script
                        with open(hydra_dir / "config.yaml", "w") as f:
                            yaml.dump(dataclasses.asdict(config), f)
                            
                    except RuntimeError:
                        print(f"  -> Trial {i+1} FAILED (Finite-Time Escape / Stiff Dynamics)")
                
                # Crucial step: Free up your Mac's RAM before loading the next ResNet architecture
                jax.clear_caches()
                
    print("\n[PHASE 2 COMPLETE] Run your aggregator script on outputs/massive_sweep/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Orchestrator for the DNN Adaptive Control Sweep")
    parser.add_argument("--tune", action="store_true", help="Run Phase 1: Tune baseline robust gains with NN off.")
    parser.add_argument("--sweep", action="store_true", help="Run Phase 2: Execute the massive Monte Carlo sweep.")
    args = parser.parse_args()

    # --- INPUT YOUR OPTUNA RESULTS HERE AFTER RUNNING --tune ---
    HARDCODED_GAINS = {
        1: {"k_1": 14.0, "k_2": 14.0, "beta": 5.0},
        2: {"k_1": 14.0, "k_2": 14.0, "beta": 5.0},
        3: {"k_1": 14.0, "k_2": 14.0, "beta": 5.0}
    }

    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample usage:\n  python scripts/master_sweep.py --tune\n  python scripts/master_sweep.py --sweep")
        exit()

    if args.tune:
        phase_1_tune_baselines()
        
    if args.sweep:
        phase_2_massive_sweep(HARDCODED_GAINS)