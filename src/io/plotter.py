import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.conf.config_schema import ExperimentConfig

def _setup_figure(config: ExperimentConfig) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(config.plot_settings.aspect_ratio_x, config.plot_settings.aspect_ratio_y))
    return fig, ax

def _save_and_close(
    fig: plt.Figure,
    ax: plt.Axes,
    output_dir: Path,
    filename_base: str,
    config: ExperimentConfig,
    y_label: str
) -> None:
    ax.set_xlabel(config.data_labels.time, fontsize=config.plot_settings.label_font_size)
    ax.set_ylabel(y_label, fontsize=config.plot_settings.label_font_size)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Only draw the legend if there are a reasonable number of lines
    if len(ax.lines) <= 15:
        ax.legend()
        
    fig.tight_layout()

    full_path = output_dir / f"{filename_base}.{config.plot_settings.save_extension}"
    fig.savefig(full_path, bbox_inches="tight", dpi=config.plot_settings.dpi)

    if config.plot_settings.show_figures:
        plt.show(block=False)
    
    plt.close(fig)

def generate_all_plots(
    data_filepath: Path,
    figures_dir: Path,
    config: ExperimentConfig
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    with data_filepath.open("rb") as f:
        data = pickle.load(f)

    t = np.asarray(data[config.data_labels.time])
    e = np.asarray(data[config.data_labels.tracking_error])
    u = np.asarray(data[config.data_labels.control_effort])
    x = np.asarray(data[config.data_labels.states])
    x_d = np.asarray(data[config.data_labels.desired_states])
    theta_hat = np.asarray(data[config.data_labels.parameter_estimate])
    gamma = np.asarray(data[config.data_labels.learning_rate_matrix])
    epsilon = np.asarray(data[config.data_labels.reconstruction_error])

    fig, ax = _setup_figure(config)
    e_norm = np.linalg.norm(e, axis=1)
    ax.plot(t, e_norm, label="||e||")
    _save_and_close(fig, ax, figures_dir, config.plot_settings.filename_tracking_error_norm, config, "Tracking Error Norm")

    fig, ax = _setup_figure(config)
    u_norm = np.linalg.norm(u, axis=1)
    ax.plot(t, u_norm, label="||u||")
    _save_and_close(fig, ax, figures_dir, config.plot_settings.filename_control_input_norm, config, "Control Input Norm")

    fig, ax = _setup_figure(config)
    num_states = x.shape[1]
    for i in range(num_states):
        ax.plot(t, x[:, i], label=f"State {i+1}")
        ax.plot(t, x_d[:, i], label=f"Desired State {i+1}", linestyle="--")
    _save_and_close(fig, ax, figures_dir, config.plot_settings.filename_states, config, "States")

    fig, ax = _setup_figure(config)
    num_params = theta_hat.shape[1]
    for i in range(num_params):
        ax.plot(t, theta_hat[:, i], label=f"Theta {i+1}")
    _save_and_close(fig, ax, figures_dir, config.plot_settings.filename_theta_hat, config, "Parameter Estimates")

    fig, ax = _setup_figure(config)
    # Gamma is shape (time_steps, p, p). eigvalsh computes eigenvalues for the stack of symmetric matrices.
    eigenvalues = np.linalg.eigvalsh(gamma) 
    num_eigenvalues = eigenvalues.shape[1]
    for i in range(num_eigenvalues):
        ax.plot(t, eigenvalues[:, i], label=f"Eigenvalue {i+1}")
    _save_and_close(fig, ax, figures_dir, config.plot_settings.filename_gamma, config, "Gamma Eigenvalues")

    fig, ax = _setup_figure(config)
    num_epsilon_dims = epsilon.shape[1]
    for i in range(num_epsilon_dims):
        ax.plot(t, epsilon[:, i], label=f"Epsilon {i+1}")
    _save_and_close(fig, ax, figures_dir, config.plot_settings.filename_reconstruction_error, config, "Reconstruction Error")