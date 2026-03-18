import sys
import os
import argparse
from pathlib import Path
import yaml
import dataclasses

import jax
import jax.numpy as jnp
import optuna
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

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

# --- UNIFIED EXPERIMENT SETTINGS ---
SYSTEMS = list(range(1, 10))  # Systems 1 through 9
MC_TRIALS = 20
TARGET_PARAMS = {"micro": 50, "small": 100, "medium": 200, "large": 400}
N_STATES_MAP = {1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 2, 8: 3, 9: 4}

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

def build_config(sys_id, ctrl_name, seed, gains, arch, d_in, d_out=2):
    GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="config")
    except Exception:
        with initialize(version_base=None, config_path="src/conf"):
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
    
    config.simulation.sys_id = sys_id
    config.simulation.controller_type = ctrl_name
    config.simulation.randomize_x0 = False # Handled manually in tuning
    config.simulation.random_seed = seed
    
    config.math_constants.k_1 = gains["k_1"]
    config.math_constants.k_2 = gains["k_2"]
    config.math_constants.beta = gains["beta"]
    
    config.neural_network.d_in = d_in
    if hasattr(config.neural_network, 'd_out'):
        config.neural_network.d_out = d_out
        
    config.neural_network.b = arch["b"]
    config.neural_network.k_0 = arch["k_0"]
    config.neural_network.k_i = arch["k_i"]
    config.neural_network.hidden_width = arch["hidden_width"]
    
    return config

def generate_monte_carlo_x0(num_samples: int, key: jax.Array, bounds: float = 2.5, d_out: int = 2) -> jax.Array:
    """Generates a batch of random initial conditions uniformly distributed within [-bounds, bounds]."""
    return jax.random.uniform(key, shape=(num_samples, d_out), minval=-bounds, maxval=bounds)

def phase_1_tune_baselines():
    print("="*60 + "\nPHASE 1: MONTE CARLO OPTUNA TUNING (ALL SYSTEMS)\n" + "="*60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_gains = {}
    
    for sys_id in SYSTEMS:
        d_out = N_STATES_MAP[sys_id]
        print(f"\n--- Tuning System {sys_id} ({d_out}D) ---")
        
        def objective(trial):
            # Wide bounds to accommodate highly unstable polynomial systems
            k_1 = trial.suggest_float("k_1", 0.1, 50.0)
            k_2 = trial.suggest_float("k_2", 0.1, 50.0)
            beta = trial.suggest_float("beta", 0.0, 30.0)
            
            arch = {"b": 0, "k_0": 1, "k_i": 1, "hidden_width": 2, "actual_p": 0}
            config = build_config(sys_id, 'baseline', seed=42, 
                                  gains={"k_1": k_1, "k_2": k_2, "beta": beta}, 
                                  arch=arch, d_in=d_out, d_out=d_out)
            
            # Disable neural network learning
            config.math_constants.k_theta_hat = 0.0
            config.neural_network.init_mean = 0.0
            config.neural_network.init_std = 0.0
            config.math_constants.learning_rate_upper_bound_mult = 1.0
            config.math_constants.learning_rate_lower_bound_mult = 1.0
            config.simulation.debug_print = False

            # Monte Carlo Initial Conditions
            num_mc_samples = 5
            key = jax.random.PRNGKey(trial.number)
            x0_batch = generate_monte_carlo_x0(num_mc_samples, key, bounds=2.0, d_out=d_out)
            
            mc_tracking_errors = []
            mc_control_efforts = []
            
            for i in range(num_mc_samples):
                config.simulation.x0 = x0_batch[i].tolist()
                try:
                    sim_data = run_simulation(config)
                    e = sim_data[config.data_labels.tracking_error]
                    u = sim_data[config.data_labels.control_effort]
                    
                    rms_e = float(jnp.sqrt(jnp.mean(jnp.sum(e**2, axis=-1))))
                    rms_u = float(jnp.sqrt(jnp.mean(jnp.sum(u**2, axis=-1))))
                    
                    mc_tracking_errors.append(rms_e)
                    mc_control_efforts.append(rms_u)
                except RuntimeError:
                    raise optuna.TrialPruned()

            # Aggregate and calculate composite cost
            avg_rms_e = jnp.mean(jnp.array(mc_tracking_errors))
            avg_rms_u = jnp.mean(jnp.array(mc_control_efforts))
            
            target_rms_e = 1.25
            error_penalty = float(abs(avg_rms_e - target_rms_e))
            u_penalty = float(0.01 * avg_rms_u)
            
            return error_penalty + u_penalty

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        try:
            print(f"Best Cost: {study.best_value:.4f} | Gains: {study.best_params}")
            best_gains[sys_id] = study.best_params
        except ValueError:
            print(f"System {sys_id} yielded 0 successful trials. Manual bound adjustment required.")
        
        jax.clear_caches()
        
    print("\n[PHASE 1 COMPLETE] Update your script's HARDCODED_GAINS dictionary to:")
    print("HARDCODED_GAINS = {")
    for k, v in best_gains.items():
        print(f"    {k}: {v},")
    print("}")

def phase_2_unified_sweep(gains_dict: dict):
    print("\n" + "="*60 + "\nPHASE 2: UNIFIED MASSIVE SWEEP (SYSTEMS 1-9)\n" + "="*60)
    
    controllers = ["baseline", "nn_in_integral"]
    base_output_dir = Path("outputs/unified_sweep")
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
                    config.simulation.randomize_x0 = True # Enable random x0 for final sweep
                    
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
                        print(f"  -> Trial {i+1} FAILED (Finite Escape / Numerical Instability)")
                
                jax.clear_caches()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Master Sweep for Systems 1-9")
    parser.add_argument("--tune", action="store_true", help="Run Phase 1: Native Optuna Tuning")
    parser.add_argument("--sweep", action="store_true", help="Run Phase 2: Massive Monte Carlo Sweep")
    args = parser.parse_args()

    # Populate this dictionary with Phase 1 outputs prior to executing Phase 2
    HARDCODED_GAINS = {
        1: {'k_1': 14.0, 'k_2': 14.0, 'beta': 5.0},
        2: {'k_1': 14.0, 'k_2': 14.0, 'beta': 5.0},
        3: {'k_1': 14.0, 'k_2': 14.0, 'beta': 5.0},
        4: {'k_1': 14.0, 'k_2': 14.0, 'beta': 9.0},
        5: {'k_1': 14.0, 'k_2': 14.0, 'beta': 9.0},
        6: {'k_1': 14.0, 'k_2': 14.0, 'beta': 9.0},
        7: {'k_1': 35.0, 'k_2': 35.0, 'beta': 20.0},
        8: {'k_1': 7.0,  'k_2': 30.0, 'beta': 2.0},
        9: {'k_1': 1.5,  'k_2': 44.5, 'beta': 22.0},
    }

    if not any(vars(args).values()):
        parser.print_help()
        exit()

    if args.tune:
        phase_1_tune_baselines()
    if args.sweep:
        phase_2_unified_sweep(HARDCODED_GAINS)