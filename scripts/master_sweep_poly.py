import os
import argparse
from pathlib import Path
import jax
from hydra import initialize, compose
import yaml
import dataclasses
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

# Baseline gains from Phase 1
HARDCODED_GAINS = {
    4: {"k_1": 14.0, "k_2": 14.0, "beta": 5.0},
    5: {"k_1": 14.0, "k_2": 14.0, "beta": 5.0},
    6: {"k_1": 14.0, "k_2": 14.0, "beta": 5.0}
}

def find_matched_architecture(target_p: int, d_in: int, d_out: int) -> dict:
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

def build_config(sys_id, ctrl_name, seed, gains, arch, d_in, d_out):
    try:
        with initialize(version_base=None, config_path="../src/conf"):
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
    config.simulation.randomize_x0 = True
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

def phase_2_poly_sweep():
    print("\n" + "="*50 + "\nPHASE 2: POLYNOMIAL SCALING SWEEP\n" + "="*50)
    controllers = ["baseline", "nn_in_integral"]
    base_output_dir = Path("outputs/poly_sweep")
    
    for sys_id in SYSTEMS:
        d_out = N_STATES_MAP[sys_id]
        gains = HARDCODED_GAINS[sys_id]
        
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
                        sim_data = run_simulation(config)
                        calculate_and_save_statistics(sim_data, run_dir, config)
                        with open(run_dir / ".hydra" / "config.yaml", "w") as f:
                            yaml.dump(dataclasses.asdict(config), f)
                    except RuntimeError:
                        print(f"  -> Trial {i+1} FAILED (Finite Escape)")
                jax.clear_caches()

if __name__ == "__main__":
    phase_2_poly_sweep()