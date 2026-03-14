import os
import platform
from pathlib import Path

import jax
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

from src.core.config_schema import (
    DataLabelsConfig,
    DirectoriesConfig,
    ExperimentConfig,
    MathConstantsConfig,
    NeuralNetworkConfig,
    PlotSettingsConfig,
    SimulationConfig,
    AnimationConfig,
)
from src.io.data_exporter import export_to_pickle
from src.io.plotter import generate_all_plots
from src.io.statistics import calculate_and_save_statistics
from src.simulation.runner import run_simulation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
if platform.system() == "Darwin":
    os.environ["JAX_PLATFORMS"] = "cpu"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    
    # Check if we are in an Optuna multirun or a standard single run
    is_multirun = HydraConfig.get().mode == RunMode.MULTIRUN

    if not is_multirun:
        backend = jax.default_backend().upper()
        print("\n" + "="*40)
        if platform.system() == "Darwin":
            print(f"System: macOS detected.")
            print(f"Hardware: Apple GPU support is experimental for Diffrax.")
            print(f"Action: Falling back to highly optimized XLA {backend}.")
        else:
            print(f"System: {platform.system()} detected.")
            print(f"Hardware: JAX is hardware-accelerated on {backend}.")
        print("="*40 + "\n")
    
    config = ExperimentConfig(
        directories=DirectoriesConfig(**cfg.directories),
        simulation=SimulationConfig(**cfg.simulation),
        math_constants=MathConstantsConfig(**cfg.math_constants),
        neural_network=NeuralNetworkConfig(**cfg.neural_network),
        data_labels=DataLabelsConfig(**cfg.data_labels),
        plot_settings=PlotSettingsConfig(**cfg.plot_settings),
         animation=AnimationConfig(**cfg.animation)
    )

    try:
        sim_data = run_simulation(config)
    except RuntimeError as e:
        if is_multirun:
            print(f"\n[SWEEP WARNING] Finite-time escape detected. Rejecting parameters.")
            return 1e9  # Massive penalty for Optuna
        else:
            raise e # Let it crash loudly for a single run

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    figures_dir = output_dir / config.directories.figures_dir

    export_to_pickle(sim_data, output_dir, "simulation_data.pkl")
    generate_all_plots(output_dir / "simulation_data.pkl", figures_dir, config)
    
    # Calculate statistics
    stats = calculate_and_save_statistics(sim_data, output_dir, config)

    # Conditionally print statistics
    if not is_multirun:
        print(f"\n{'='*40}")
        print("SIMULATION STATISTICS")
        print(f"{'-'*40}")
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key}: {value:.3f}")
        print(f"{'='*40}\n")

    # --- CUSTOM OPTUNA COST FUNCTION ---
    # J = RMS(e) + 0.01 * RMS(u)
    cost = stats["rms_tracking_error_norm"] + 0.01 * stats["rms_control_input_norm"]
    
    if is_multirun:
        print(f"Trial Complete | Cost: {cost:.3f} | RMS(e): {stats['rms_tracking_error_norm']:.3f}")

    return cost

if __name__ == "__main__":
    main()