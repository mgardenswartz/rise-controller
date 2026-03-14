import json
from pathlib import Path
import numpy as np
from src.conf.config_schema import ExperimentConfig
from src.math.networks import get_total_parameters

def _rms_of_norm(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    norms = np.linalg.norm(arr, axis=1)
    return float(np.sqrt(np.mean(norms**2)))

def calculate_and_save_statistics(
    sim_data: dict[str, np.ndarray], 
    output_dir: Path, 
    config: ExperimentConfig
) -> dict[str, float]:
    
    t = np.asarray(sim_data[config.data_labels.time])
    t_pe = config.simulation.excitation_duration_seconds
    
    valid_idx = t > t_pe
    
    if not np.any(valid_idx):
        print(f"\n[STATISTICS WARNING] Simulation ended at t={t[-1]:.2f}s before PE ended at t={t_pe}s. Using available data.")
        valid_idx = np.ones_like(t, dtype=bool)

    e_post = np.asarray(sim_data[config.data_labels.tracking_error])[valid_idx]
    u_post = np.asarray(sim_data[config.data_labels.control_effort])[valid_idx]
    epsilon_post = np.asarray(sim_data[config.data_labels.reconstruction_error])[valid_idx]
    phi_post = np.asarray(sim_data[config.data_labels.nn_output])[valid_idx]

    p = get_total_parameters(
        config.neural_network.d_in, 
        config.neural_network.hidden_width, 
        config.neural_network.d_out,
        config.neural_network.b,
        config.neural_network.k_0,
        config.neural_network.k_i
    )
    flops_per_pass = 2 * p 

    stats = {
        "rms_tracking_error_norm": _rms_of_norm(e_post),
        "rms_control_input_norm": _rms_of_norm(u_post),
        "rms_reconstruction_error_norm": _rms_of_norm(epsilon_post),
        "rms_nn_output_norm": _rms_of_norm(phi_post),
        "total_trainable_parameters": float(p),
        "forward_pass_flops": float(flops_per_pass)
    }

    stats_path = output_dir / config.directories.statistics_filename
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    return stats