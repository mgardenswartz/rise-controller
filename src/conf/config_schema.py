from dataclasses import dataclass

@dataclass
class DirectoriesConfig:
    output_parent_dir: str
    recycle_bin_dir: str
    results_iterable_dir_prefix: str
    default_config_dir: str
    figures_dir: str
    raw_data_dir: str
    config_filename: str
    statistics_filename: str

@dataclass
class SimulationConfig:
    controller_type: str
    t0: float
    randomize_x0: bool
    random_x0_square_size: float
    sys_id: int
    duration_seconds: float
    excitation_duration_seconds: float
    x0: list[float]
    max_solver_steps: int
    rtol: float
    atol: float
    random_seed: int
    debug_print: bool
    noise_std: float
    noise_mean: float
    enable_learning: bool
    control_frequency_hz: float # also controls save interval

@dataclass
class MathConstantsConfig:
    learning_rate: float
    k_theta_hat: float
    k_1: float
    k_2: float
    beta: float
    theta_bar: float

@dataclass
class NeuralNetworkConfig:
    d_in: int
    d_out: int
    b: int
    k_0: int
    k_i: int
    hidden_width: int
    hidden_activation: str
    output_activation: str
    shortcut_activation: str
    init_mean: float
    init_std: float
    init_type: str

@dataclass
class DataLabelsConfig:
    tracking_error: str
    parameter_estimate: str
    states: str
    control_effort: str
    desired_states: str
    reconstruction_error: str
    learning_rate_matrix: str
    nn_output: str
    time: str

@dataclass
class PlotSettingsConfig:
    aspect_ratio_x: int
    aspect_ratio_y: int
    dpi: int
    label_font_size: int
    save_extension: str
    show_figures: bool
    filename_tracking_error_norm: str
    filename_control_input_norm: str
    filename_states: str
    filename_theta_hat: str
    filename_gamma: str
    filename_reconstruction_error: str

@dataclass
class AnimationConfig:
    fps: int
    dpi: int
    trail_duration_seconds: float
    filename: str
    color_state_dot: str
    color_state_trail: str
    color_desired_star: str
    color_desired_trail: str
    x_label: str
    y_label: str

@dataclass
class ExperimentConfig:
    directories: DirectoriesConfig
    simulation: SimulationConfig
    math_constants: MathConstantsConfig
    neural_network: NeuralNetworkConfig
    data_labels: DataLabelsConfig
    plot_settings: PlotSettingsConfig
    animation: AnimationConfig