import sys
import argparse
from pathlib import Path
import yaml
import dataclasses
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import gc

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
from src.math.networks import get_total_parameters

# --- UNIFIED EXPERIMENT SETTINGS ---
MC_TRIALS = 10
SYSTEMS = [1, 2, 4, 6, 8]
N_STATES_MAP = {1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 6}
NUMERICAL_SEED = 42
STAGE_1_NUM_TRIALS = 40

DETUNE_FACTORS = [0.5, 0.2, 0.1]

# Persistent Gains File
GAINS_FILE = PROJECT_ROOT / "src" / "conf" / "tuned_gains.yaml"

# Explicitly locked architectures (Scaling Width, Blocks, k_0, k_i)
TARGET_ARCHS = {
    "small":  {"width_multiplier": 4,  "b": 0, "k_0": 3, "k_i": 2},
    # "medium": {"width_multiplier": 2,  "b": 1, "k_0": 2, "k_i": 2},
    # "large":  {"width_multiplier": 2,  "b": 2, "k_0": 2, "k_i": 2},
}

# --- YAML GAINS MANAGEMENT ---
def load_tuned_gains() -> dict:
    if GAINS_FILE.exists():
        with open(GAINS_FILE, "r") as f:
            raw_gains = yaml.safe_load(f) or {}
            # Keys are now strings to handle detune prefixes (e.g., "8_0.5")
            return {str(k): v for k, v in raw_gains.items()}
    return {}

def save_tuned_gains(new_gains: dict):
    existing_gains = load_tuned_gains()
    for sys_key, gains in new_gains.items():
        if sys_key not in existing_gains:
            existing_gains[sys_key] = {}
        existing_gains[sys_key].update(gains)
        
    yaml_ready_gains = {str(k): v for k, v in existing_gains.items()}
    GAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GAINS_FILE, "w") as f:
        yaml.dump(yaml_ready_gains, f, default_flow_style=False, sort_keys=True)
    print(f"\n[SAVED] Tuned gains successfully merged into {GAINS_FILE}")

def build_config(sys_id, ctrl_name, seed, gains, arch, state_space_dim):
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
    config.simulation.randomize_x0 = False
    config.simulation.random_seed = seed
    
    config.simulation.state_space_dim = state_space_dim

    config.math_constants.k_1 = gains.get("k_1", 5.0)
    config.math_constants.k_2 = gains.get("k_2", 5.0)
    config.math_constants.beta = gains.get("beta", 0.5)

    config.neural_network.b = arch["b"]
    config.neural_network.k_0 = arch["k_0"]
    config.neural_network.k_i = arch["k_i"]
    config.neural_network.hidden_width = arch["hidden_width"]
    
    if ctrl_name.startswith("integral_"):
        config.neural_network.d_in = state_space_dim * 2
    else:
        config.neural_network.d_in = state_space_dim
    config.neural_network.d_out = state_space_dim
    
    return config

def generate_monte_carlo_x0(num_samples: int, key: jax.Array, bounds: float = 2.5, dim: int = 2) -> jax.Array:
    return jax.random.uniform(key, shape=(num_samples, dim), minval=-bounds, maxval=bounds)

def save_diagnostic_plots(sim_data: dict, run_dir: Path, config: ExperimentConfig):
    t = sim_data[config.data_labels.time]
    e = sim_data[config.data_labels.tracking_error]
    u = sim_data[config.data_labels.control_effort]
    
    e_norm = jnp.linalg.norm(e, axis=1)
    u_norm = jnp.linalg.norm(u, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(t, e_norm, color='red', linewidth=1.5)
    ax1.set_title(f"Tracking Error Norm ||e|| (Sys {config.simulation.sys_id})")
    ax1.set_ylabel("Error Norm")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, u_norm, color='blue', linewidth=1.5)
    ax2.set_title("Control Effort Norm ||u||")
    ax2.set_ylabel("Effort Norm")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(run_dir / "diagnostic_plot.png", dpi=150)
    plt.close(fig)

# --- THE UNIVERSAL OBJECTIVE EVALUATOR ---
def evaluate_trial(config: ExperimentConfig, trial: optuna.Trial, num_mc_samples: int = 5) -> float:
    key = jax.random.PRNGKey(trial.number)
    x0_batch = generate_monte_carlo_x0(num_mc_samples, key, bounds=2.5, dim=config.simulation.state_space_dim)
    
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
            
            if jnp.isnan(rms_e) or jnp.isnan(rms_u) or jnp.isinf(rms_e):
                raise optuna.TrialPruned()
                
            mc_tracking_errors.append(rms_e)
            mc_control_efforts.append(rms_u)
        except Exception as e:
            print(f"Trial pruned due to exception: {e}")
            raise optuna.TrialPruned()

    avg_rms_e = float(jnp.mean(jnp.array(mc_tracking_errors)))
    avg_rms_u = float(jnp.mean(jnp.array(mc_control_efforts)))
    
    error_cost = float(jnp.exp(8.0 * avg_rms_e) - 1.0)
    effort_cost = 1e-6 * (avg_rms_u ** 4)
    
    gain_cost = 0.05 * (config.math_constants.k_1**2 + config.math_constants.k_2**2 + config.math_constants.beta**2) + (5 * config.math_constants.k_theta_hat)**4
    
    return error_cost + effort_cost + gain_cost

# --- PHASE 1: THE THREE-STAGE TUNER ---
def phase_1_tune_all():
    print("="*70 + "\nPHASE 1: EXHAUSTIVE DETUNING OPTIMIZATION\n" + "="*70)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    for sys_id in SYSTEMS:
        d_out = N_STATES_MAP[sys_id]
        print(f"\n[{sys_id}] TUNING SYSTEM {sys_id} ({d_out}D)")
        
        # -----------------------------------------------------------------
        # STAGE 1: FIND OPTIMAL KINEMATIC GAINS
        # -----------------------------------------------------------------
        print(f"  -> Stage 1/3: Finding Absolute Optimal Linear Kinematics...")
        def obj_kinematic(trial):
            k_1 = trial.suggest_float("k_1", 0.1, 100.0)
            k_2 = trial.suggest_float("k_2", 0.1, 100.0)
            beta = trial.suggest_float("beta", 0.0, 50.0)
            
            arch = {"hidden_width": 2, "b": 0, "k_0": 1, "k_i": 1}
            config = build_config(sys_id, 'baseline', seed=NUMERICAL_SEED, gains={"k_1": k_1, "k_2": k_2, "beta": beta}, arch=arch, state_space_dim=d_out)
            config.simulation.enable_learning = False
            config.neural_network.init_mean = 0.0
            config.neural_network.init_std = 0.0
            return evaluate_trial(config, trial)

        study_kin = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=NUMERICAL_SEED))
        study_kin.optimize(obj_kinematic, n_trials=STAGE_1_NUM_TRIALS, show_progress_bar=True)
        optimal_gains = study_kin.best_params
        print(f"     Found Ceiling Gains: k1={optimal_gains['k_1']:.2f}, k2={optimal_gains['k_2']:.2f}, beta={optimal_gains['beta']:.2f}")
        
        # -----------------------------------------------------------------
        # DETUNING LOOP: Methodically Starve the Linear Controller
        # -----------------------------------------------------------------
        for detune in DETUNE_FACTORS:
            sys_key = f"{sys_id}_{detune}"
            print(f"\n  >>> DETUNE FACTOR: {detune*100:.0f}% of Optimal Gains <<<")
            
            sys_gains = {
                "k_1": optimal_gains["k_1"] * detune,
                "k_2": optimal_gains["k_2"] * detune,
                "beta": optimal_gains["beta"] * detune
            }
            
            # STAGE 2: TUNE BASELINE
            print(f"  -> Stage 2/3: Tuning Baseline Learning Rate & k_theta_hat...")
            def obj_lr_base(trial):
                lr = trial.suggest_float("lr", 1e-4, 50.0, log=True) 
                k_theta = trial.suggest_float("k_theta_hat", 0.0, 5.0)
                
                arch = TARGET_ARCHS["small"].copy() 
                arch["hidden_width"] = int(d_out * arch.pop("width_multiplier"))
                
                config = build_config(sys_id, 'baseline', seed=NUMERICAL_SEED, gains=sys_gains, arch=arch, state_space_dim=d_out)
                config.simulation.enable_learning = True
                config.math_constants.learning_rate = lr
                config.math_constants.k_theta_hat = k_theta
                return evaluate_trial(config, trial)

            study_base = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=43))
            study_base.optimize(obj_lr_base, n_trials=30, show_progress_bar=True)
            sys_gains["lr_baseline"] = study_base.best_params["lr"]
            sys_gains["k_theta_hat_baseline"] = study_base.best_params["k_theta_hat"]

            # STAGE 3: TUNE INTEGRAL
            print(f"  -> Stage 3/3: Tuning Integral Learning Rate & k_theta_hat...")
            def obj_lr_int(trial):
                lr = trial.suggest_float("lr", 1e-4, 50.0, log=True)
                k_theta = trial.suggest_float("k_theta_hat", 0.0, 5.0)

                arch = TARGET_ARCHS["small"].copy()
                arch["hidden_width"] = int(d_out * arch.pop("width_multiplier"))

                config = build_config(sys_id, 'nn_in_integral', seed=NUMERICAL_SEED, gains=sys_gains, arch=arch, state_space_dim=d_out)
                config.simulation.enable_learning = True
                config.math_constants.learning_rate = lr
                config.math_constants.k_theta_hat = k_theta
                return evaluate_trial(config, trial)

            study_int = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=44))
            study_int.optimize(obj_lr_int, n_trials=30, show_progress_bar=True)
            sys_gains["lr_integral"] = study_int.best_params["lr"]
            sys_gains["k_theta_hat_integral"] = study_int.best_params["k_theta_hat"]
            
            save_tuned_gains({sys_key: sys_gains})
            jax.clear_caches()

# --- PHASE 2: UNIFIED SWEEP ---
def phase_2_unified_sweep(gains_dict: dict, save_plots: bool = False):
    print("\n" + "="*70 + "\nPHASE 2: UNIFIED MASSIVE SWEEP\n" + "="*70)
    
    controllers = ["integral_r", "integral_e", "direct_r", "direct_e"]
    base_output_dir = Path("outputs/unified_sweep")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'rms_e': [], 'rms_u': [], 'actual_p': 0})))
    
    for sys_id in SYSTEMS:
        d_out = N_STATES_MAP[sys_id]
        
        # Iterate over all available detuning factors saved in the yaml
        available_keys = [k for k in gains_dict.keys() if k.startswith(f"{sys_id}_")]
        if not available_keys:
            print(f"[WARNING] Skipping System {sys_id} - No tuned gains found in YAML.")
            continue
            
        for sys_key in sorted(available_keys, reverse=True):
            detune_str = sys_key.split("_")[1]
            gains = gains_dict[sys_key]
            
            for ctrl_name in controllers:
                target_lr = gains.get("lr_integral", 1.0) if ctrl_name.startswith("integral_") else gains.get("lr_baseline", 1.0)
                target_k_theta = gains.get("k_theta_hat_integral", 0.0) if ctrl_name.startswith("integral_") else gains.get("k_theta_hat_baseline", 0.0)
                
                for size_name, arch_params in TARGET_ARCHS.items():
                    arch = arch_params.copy()
                    arch['hidden_width'] = int(d_out * arch.pop('width_multiplier'))
                    d_in_ctrl = d_out * 2 if ctrl_name.startswith("integral_") else d_out
                    arch['actual_p'] = get_total_parameters(d_in_ctrl, arch['hidden_width'], d_out, arch['b'], arch['k_0'], arch['k_i'])
                    
                    print(f"\n[SWEEP] Sys: {sys_id} | Detune: {detune_str} | Ctrl: {ctrl_name} | Arch: {size_name.upper()} (LR={target_lr:.4f} | k_th={target_k_theta:.4f})")
                    
                    for i in range(MC_TRIALS):
                        seed = 1000 + i
                        config = build_config(sys_id, ctrl_name, seed, gains, arch, state_space_dim=d_out)
                        config.simulation.randomize_x0 = True 
                        config.simulation.enable_learning = True
                        
                        config.math_constants.learning_rate = target_lr
                        config.math_constants.k_theta_hat = target_k_theta
                        
                        run_dir = base_output_dir / f"sys_{sys_id}_detune_{detune_str}" / ctrl_name / size_name / f"seed_{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        (run_dir / ".hydra").mkdir(exist_ok=True)
                        
                        try:
                            if i == 0:
                                print(f"  -> Trial {i+1}/{MC_TRIALS} (JIT Compiling...)")
                            else:
                                print(f"  -> Trial {i+1}/{MC_TRIALS} (Running from JIT cache...)")
                                
                            sim_data = run_simulation(config)
                            calculate_and_save_statistics(sim_data, run_dir, config)
                            
                            if save_plots:
                                if i == 0: print("    -> Generating diagnostic plot...")
                                save_diagnostic_plots(sim_data, run_dir, config)
                            
                            e = sim_data[config.data_labels.tracking_error]
                            u = sim_data[config.data_labels.control_effort]
                            
                            if jnp.any(jnp.isnan(e)) or jnp.any(jnp.isnan(u)) or jnp.any(jnp.isinf(e)) or jnp.any(jnp.isinf(u)):
                                print(f"  -> Trial {i+1} FAILED (Invalid numerical output)")
                                rms_e, rms_u = float('nan'), float('nan')
                            else:
                                rms_e = float(jnp.sqrt(jnp.mean(jnp.sum(e**2, axis=-1))))
                                rms_u = float(jnp.sqrt(jnp.mean(jnp.sum(u**2, axis=-1))))
                                
                            results_dict[sys_key][size_name][ctrl_name]['rms_e'].append(rms_e)
                            results_dict[sys_key][size_name][ctrl_name]['rms_u'].append(rms_u)
                            results_dict[sys_key][size_name][ctrl_name]['actual_p'] = arch['actual_p']
                            
                            with open(run_dir / ".hydra" / "config.yaml", "w") as f:
                                yaml.dump(dataclasses.asdict(config), f)
                                
                        except Exception as e:
                            print(f"  -> Trial {i+1} FAILED (Exception: {type(e).__name__})")
                            results_dict[sys_key][size_name][ctrl_name]['rms_e'].append(float('nan'))
                            results_dict[sys_key][size_name][ctrl_name]['rms_u'].append(float('nan'))
                            results_dict[sys_key][size_name][ctrl_name]['actual_p'] = arch['actual_p']

                        if 'sim_data' in locals(): del sim_data
                        if 'e' in locals(): del e
                        if 'u' in locals(): del u
                        gc.collect()
                    
                    jax.clear_caches()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Master Sweep for Systems 7-9")
    parser.add_argument("--tune", action="store_true", help="Run Phase 1: Exhaustive Detuning Optimization")
    parser.add_argument("--sweep", action="store_true", help="Run Phase 2: Massive Monte Carlo Sweep")
    parser.add_argument("--plot", action="store_true", help="WARNING: Generates plots for every trial. Slows down sweep.")
    args = parser.parse_args()

    if not any([args.tune, args.sweep]):
        parser.print_help()
        exit()

    if args.tune:
        phase_1_tune_all()
        
    if args.sweep:
        loaded_gains = load_tuned_gains()
        if not loaded_gains:
            print("\n[ERROR] No tuned gains found. Please run with --tune first to populate the YAML.\n")
            exit()
        phase_2_unified_sweep(loaded_gains, save_plots=args.plot)