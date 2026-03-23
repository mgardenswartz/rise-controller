import sys
import argparse
from pathlib import Path
import yaml
import dataclasses
from collections import defaultdict
import numpy as np

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
SYSTEMS = list(range(7, 10))  # Systems 7, 8, 9
MC_TRIALS = 10
N_STATES_MAP = {1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 2, 8: 3, 9: 4}

# Persistent Gains File
GAINS_FILE = PROJECT_ROOT / "src" / "conf" / "tuned_gains.yaml"

# Explicitly locked architectures (Width, Blocks, k_0, k_i)
TARGET_ARCHS = {
    "small":  {"hidden_width": 4,  "b": 1, "k_0": 1, "k_i": 2},
    "medium": {"hidden_width": 4, "b": 2, "k_0": 1, "k_i": 2},
    "large":  {"hidden_width": 4, "b": 4, "k_0": 1, "k_i": 2}
}

# --- YAML GAINS MANAGEMENT ---
def load_tuned_gains() -> dict:
    """Loads gains from YAML and converts string keys back to integers."""
    if GAINS_FILE.exists():
        with open(GAINS_FILE, "r") as f:
            raw_gains = yaml.safe_load(f) or {}
            return {int(k): v for k, v in raw_gains.items()}
    return {}

def save_tuned_gains(new_gains: dict):
    """Merges new gains with existing YAML data without overwriting untuned systems."""
    existing_gains = load_tuned_gains()
    existing_gains.update(new_gains)
    
    # Convert integer keys to strings for clean YAML formatting
    yaml_ready_gains = {str(k): v for k, v in existing_gains.items()}
    
    GAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GAINS_FILE, "w") as f:
        yaml.dump(yaml_ready_gains, f, default_flow_style=False, sort_keys=True)
    print(f"\n[SAVED] Tuned gains successfully merged into {GAINS_FILE}")

# --- ARCHITECTURE MATCHING ---
def get_actual_p(d_in: int, w: int, d_out: int, b: int, k_0: int, k_i: int) -> int:
    """Calculates exact parameter count based on fixed architecture."""
    p_in = (d_in * w) + w
    p_out = (w * d_out) + d_out
    p_k0 = (k_0 - 1) * ((w * w) + w)
    p_blocks = b * k_i * ((w * w) + w)
    return p_in + p_out + p_k0 + p_blocks

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
    config.simulation.randomize_x0 = False
    config.simulation.random_seed = seed
    
    config.math_constants.k_1 = gains["k_1"]
    config.math_constants.k_2 = gains["k_2"]
    config.math_constants.beta = gains["beta"]
    
    config.neural_network.d_in = d_in
    config.neural_network.d_out = d_out
        
    config.neural_network.b = arch["b"]
    config.neural_network.k_0 = arch["k_0"]
    config.neural_network.k_i = arch["k_i"]
    config.neural_network.hidden_width = arch["hidden_width"]
    
    return config

def generate_monte_carlo_x0(num_samples: int, key: jax.Array, bounds: float = 2.5, d_out: int = 2) -> jax.Array:
    return jax.random.uniform(key, shape=(num_samples, d_out), minval=-bounds, maxval=bounds)

# --- PHASE 1: TUNING ---
def phase_1_tune_baselines():
    print("="*60 + "\nPHASE 1: MONTE CARLO OPTUNA TUNING\n" + "="*60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_gains = {}
    
    for sys_id in SYSTEMS:
        d_out = N_STATES_MAP[sys_id]
        print(f"\n--- Tuning System {sys_id} ({d_out}D) ---")
        
        def objective(trial):
            k_1 = trial.suggest_float("k_1", 0.1, 100.0)
            k_2 = trial.suggest_float("k_2", 0.1, 100.0)
            beta = trial.suggest_float("beta", 0.0, 50.0)
            
            arch = {"hidden_width": 2, "b": 0, "k_0": 1, "k_i": 1}
            config = build_config(sys_id, 'baseline', seed=42, 
                                  gains={"k_1": k_1, "k_2": k_2, "beta": beta}, 
                                  arch=arch, d_in=d_out, d_out=d_out)
            
            config.simulation.enable_learning = False
            config.neural_network.init_mean = 0.0
            config.neural_network.init_std = 0.0
            config.simulation.debug_print = False

            num_mc_samples = 5
            key = jax.random.PRNGKey(trial.number)
            x0_batch = generate_monte_carlo_x0(num_mc_samples, key, bounds=2.5, d_out=d_out)
            
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
                except Exception:
                    raise optuna.TrialPruned()

            avg_rms_e = float(jnp.mean(jnp.array(mc_tracking_errors)))
            avg_rms_u = float(jnp.mean(jnp.array(mc_control_efforts)))
            
            error_cost = float(jnp.exp(8.0 * avg_rms_e) - 1.0)
            effort_cost = 1e-6 * (avg_rms_u ** 4)
            gain_cost = 0.05 * (k_1**2 + k_2**2 + beta**2)
            
            return error_cost + effort_cost + gain_cost

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        try:
            print(f"Best Cost: {study.best_value:.4f} | Gains: {study.best_params}")
            best_gains[sys_id] = study.best_params
        except ValueError:
            print(f"System {sys_id} yielded 0 successful trials. Manual bound adjustment required.")
        
        jax.clear_caches()
        
    # Merge and save the tuned gains directly to YAML
    save_tuned_gains(best_gains)

# --- PHASE 2: SWEEP ---
def phase_2_unified_sweep(gains_dict: dict):
    print("\n" + "="*60 + "\nPHASE 2: UNIFIED MASSIVE SWEEP\n" + "="*60)
    
    controllers = ["baseline", "nn_in_integral"]
    base_output_dir = Path("outputs/unified_sweep")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'rms_e': [], 'rms_u': [], 'actual_p': 0})))
    
    for sys_id in SYSTEMS:
        if sys_id not in gains_dict:
            print(f"[WARNING] Skipping System {sys_id} - No tuned gains found in YAML.")
            continue
            
        d_out = N_STATES_MAP[sys_id]
        gains = gains_dict[sys_id]
        
        for ctrl_name in controllers:
            d_in = d_out if ctrl_name == "baseline" else d_out * 2
            
            for size_name, arch_params in TARGET_ARCHS.items():
                arch = arch_params.copy()
                arch['actual_p'] = get_actual_p(d_in, arch['hidden_width'], d_out, arch['b'], arch['k_0'], arch['k_i'])
                
                print(f"\n[SWEEP] Sys: {sys_id} ({d_out}D) | Ctrl: {ctrl_name} | Arch: {size_name.upper()} (P={arch['actual_p']})")
                
                for i in range(MC_TRIALS):
                    seed = 1000 + i
                    config = build_config(sys_id, ctrl_name, seed, gains, arch, d_in, d_out)
                    config.simulation.randomize_x0 = True 
                    config.simulation.enable_learning = True 
                    
                    run_dir = base_output_dir / f"sys_{sys_id}" / ctrl_name / size_name / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / ".hydra").mkdir(exist_ok=True)
                    
                    try:
                        if i == 0:
                            print(f"  -> Trial {i+1}/{MC_TRIALS} (JIT Compiling...)")
                        else:
                            print(f"  -> Trial {i+1}/{MC_TRIALS} (Running from JIT cache...)")
                            
                        sim_data = run_simulation(config)
                        calculate_and_save_statistics(sim_data, run_dir, config)
                        
                        e = sim_data[config.data_labels.tracking_error]
                        u = sim_data[config.data_labels.control_effort]
                        
                        if jnp.any(jnp.isnan(e)) or jnp.any(jnp.isnan(u)) or jnp.any(jnp.isinf(e)) or jnp.any(jnp.isinf(u)):
                            print(f"  -> Trial {i+1} FAILED (Invalid numerical output)")
                            rms_e, rms_u = float('nan'), float('nan')
                        else:
                            rms_e = float(jnp.sqrt(jnp.mean(jnp.sum(e**2, axis=-1))))
                            rms_u = float(jnp.sqrt(jnp.mean(jnp.sum(u**2, axis=-1))))
                            
                        results_dict[sys_id][size_name][ctrl_name]['rms_e'].append(rms_e)
                        results_dict[sys_id][size_name][ctrl_name]['rms_u'].append(rms_u)
                        results_dict[sys_id][size_name][ctrl_name]['actual_p'] = arch['actual_p']
                        
                        with open(run_dir / ".hydra" / "config.yaml", "w") as f:
                            yaml.dump(dataclasses.asdict(config), f)
                            
                    except Exception as e:
                        print(f"  -> Trial {i+1} FAILED (Exception: {type(e).__name__})")
                        results_dict[sys_id][size_name][ctrl_name]['rms_e'].append(float('nan'))
                        results_dict[sys_id][size_name][ctrl_name]['rms_u'].append(float('nan'))
                        results_dict[sys_id][size_name][ctrl_name]['actual_p'] = arch['actual_p']
                
                jax.clear_caches()

    # =========================================================================
    # PHASE 2 PRINT SUMMARY TABLE
    # =========================================================================
    print("\n\n" + "="*112)
    print(f"{'Sys':<4} | {'Arch Size':<11} | {'Params (B/I)':<14} | {'Base RMS(e) [Surv]':<18} | {'Int. RMS(e) [Surv]':<18} | {'Base RMS(u)':<11} | {'Int. RMS(u)':<11}")
    print("-" * 112)

    for sys_id in sorted(results_dict.keys()):
        for size_name in ["micro", "small", "medium", "large"]:
            if size_name not in results_dict[sys_id]:
                continue
            
            data = results_dict[sys_id][size_name]
            b_data = data.get('baseline', {'rms_e': [], 'rms_u': [], 'actual_p': 0})
            i_data = data.get('nn_in_integral', {'rms_e': [], 'rms_u': [], 'actual_p': 0})
            
            b_e_clean = [x for x in b_data['rms_e'] if not np.isnan(x) and not np.isinf(x)]
            b_u_clean = [x for x in b_data['rms_u'] if not np.isnan(x) and not np.isinf(x)]
            i_e_clean = [x for x in i_data['rms_e'] if not np.isnan(x) and not np.isinf(x)]
            i_u_clean = [x for x in i_data['rms_u'] if not np.isnan(x) and not np.isinf(x)]
            
            b_surv, i_surv = len(b_e_clean), len(i_e_clean)
            
            b_e_mean = f"{np.mean(b_e_clean):.4f}" if b_surv > 0 else "FAILED"
            i_e_mean = f"{np.mean(i_e_clean):.4f}" if i_surv > 0 else "FAILED"
            b_u_mean = f"{np.mean(b_u_clean):.2f}" if b_surv > 0 else "FAILED"
            i_u_mean = f"{np.mean(i_u_clean):.2f}" if i_surv > 0 else "FAILED"
            
            p_str = f"{b_data['actual_p']:>4} / {i_data['actual_p']:<4}"
            b_e_str = f"{b_e_mean:>9}  [{b_surv}/{MC_TRIALS}]"
            i_e_str = f"{i_e_mean:>9}  [{i_surv}/{MC_TRIALS}]"
            
            print(f" {sys_id:<3} | {size_name:<11} | {p_str:<14} | {b_e_str:<18} | {i_e_str:<18} | {b_u_mean:>11} | {i_u_mean:>11}")
    print("="*112 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Master Sweep for Systems 1-9")
    parser.add_argument("--tune", action="store_true", help="Run Phase 1: Native Optuna Tuning")
    parser.add_argument("--sweep", action="store_true", help="Run Phase 2: Massive Monte Carlo Sweep")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit()

    if args.tune:
        phase_1_tune_baselines()
        
    if args.sweep:
        loaded_gains = load_tuned_gains()
        if not loaded_gains:
            print("\n[ERROR] No tuned gains found. Please run with --tune first to populate the YAML.\n")
            exit()
        phase_2_unified_sweep(loaded_gains)