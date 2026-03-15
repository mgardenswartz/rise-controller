import sys
import os
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import optuna
import yaml
import dataclasses
from master_sweep import find_matched_architecture, build_config
from hydra import initialize, compose

# Force Python to see the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conf.config_schema import (
    ExperimentConfig, DirectoriesConfig, SimulationConfig,
    MathConstantsConfig, NeuralNetworkConfig, DataLabelsConfig,
    PlotSettingsConfig, AnimationConfig
)
from src.simulation.runner import run_simulation
from src.io.statistics import calculate_and_save_statistics

# --- MULTIDIMENSIONAL EXPERIMENT SETTINGS ---
SYSTEMS = [4, 5, 6]
MC_TRIALS = 20
TARGET_PARAMS = {"small": 100, "medium": 200, "large": 400}
N_STATES_MAP = {4: 2, 5: 3, 6: 4}

def phase_1_tune_baselines():
    print("="*50 + "\nPHASE 1: NATIVE OPTUNA TUNING (MULTI-D)\n" + "="*50)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_gains = {}
    
    for sys_id in SYSTEMS:
        d_out = N_STATES_MAP[sys_id]
        print(f"\n--- Tuning System {sys_id} ({d_out}D) ---")
        
        def objective(trial):
            k_1 = trial.suggest_float("k_1", 0.1, 15.0)
            k_2 = trial.suggest_float("k_2", 0.1, 15.0)
            beta = trial.suggest_float("beta", 0.0, 10.0)
            
            arch = {"b": 0, "k_0": 1, "k_i": 1, "hidden_width": 2, "actual_p": 0}
            config = build_config(sys_id, 'baseline', seed=42, 
                                  gains={"k_1": k_1, "k_2": k_2, "beta": beta}, 
                                  arch=arch, d_in=d_out, d_out=d_out)
            
            config.math_constants.k_theta_hat = 0.0
            config.neural_network.init_mean = 0.0
            config.neural_network.init_std = 0.0
            
            try:
                config.simulation.debug_print = False
                sim_data = run_simulation(config)
                e = sim_data[config.data_labels.tracking_error]
                cost = float(jnp.sqrt(jnp.mean(jnp.sum(e**2, axis=-1))))
                return cost
            except RuntimeError:
                raise optuna.TrialPruned()

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"Best Cost: {study.best_value:.4f} | Gains: {study.best_params}")
        best_gains[sys_id] = study.best_params
        jax.clear_caches()
        
    print("\n[PHASE 1 COMPLETE] Update your script's HARDCODED_GAINS dictionary to:")
    print("HARDCODED_GAINS = {")
    for k, v in best_gains.items():
        print(f"    {k}: {v},")
    print("}")

def phase_2_poly_sweep(gains_dict: dict):
    print("\n" + "="*50 + "\nPHASE 2: POLYNOMIAL SCALING SWEEP\n" + "="*50)
    
    controllers = ["baseline", "nn_in_integral"]
    base_output_dir = Path("outputs/poly_sweep")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for sys_id in SYSTEMS:
        if sys_id not in gains_dict:
            continue
            
        d_out = N_STATES_MAP[sys_id]
        gains = gains_dict[sys_id]
        
        for ctrl_name in controllers:
            d_in = d_out if ctrl_name == "baseline" else d_out * 2
            
            for size_name, target_p in TARGET_PARAMS.items():
                arch = find_matched_architecture(target_p, d_in=d_in, d_out=d_out)
                print(f"\n[SWEEP] Sys: {sys_id} ({d_out}D) | Ctrl: {ctrl_name} | Size: {size_name} (P={arch['actual_p']})")
                
                for i in range(MC_TRIALS):
                    seed = 1000 + i
                    config = build_config(sys_id, ctrl_name, seed, gains, arch, d_in, d_out)
                    
                    run_dir = base_output_dir / f"sys_{sys_id}" / ctrl_name / f"p_{arch['actual_p']}" / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / ".hydra").mkdir(exist_ok=True)
                    
                    try:
                        if i == 0:
                            print(f"  -> Trial {i+1}/{MC_TRIALS} (JIT Compiling...)")
                        else:
                            print(f"  -> Trial {i+1}/{MC_TRIALS} (Running from JIT cache...)")
                            
                        sim_data = run_simulation(config)
                        calculate_and_save_statistics(sim_data, run_dir, config)
                        
                        with open(run_dir / ".hydra" / "config.yaml", "w") as f:
                            yaml.dump(dataclasses.asdict(config), f)
                    except RuntimeError:
                        print(f"  -> Trial {i+1} FAILED (Finite Escape)")
                
                jax.clear_caches()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Sweep for Multi-Dimensional Polynomial Systems")
    parser.add_argument("--tune", action="store_true", help="Run Phase 1: Native Optuna Tuning")
    parser.add_argument("--sweep", action="store_true", help="Run Phase 2: Massive Monte Carlo Sweep")
    args = parser.parse_args()

    # Update this after running --tune
    HARDCODED_GAINS = {
        4: {"k_1": 7.0, "k_2": 7.0, "beta": 4.5},
        5: {"k_1": 7.0, "k_2": 7.0, "beta": 4.5},
        6: {"k_1": 7.0, "k_2": 7.0, "beta": 4.5}
    }

    if not any(vars(args).values()):
        parser.print_help()
        exit()

    if args.tune:
        phase_1_tune_baselines()
    if args.sweep:
        phase_2_poly_sweep(HARDCODED_GAINS)