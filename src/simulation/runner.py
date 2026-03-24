import diffrax
import jax
import jax.numpy as jnp

from src.conf.config_schema import ExperimentConfig
from src.math.dynamics import (
    get_desired_trajectory,
    get_desired_velocity,
    get_excitation_signal,
    f_sys_1, f_sys_2, f_sys_3, f_sys_4, f_sys_5, f_sys_6, f_sys_7, f_sys_8
)
from src.math.networks import compute_jacobian, get_total_parameters, resnet_network
from src.math.update_laws import compute_theta_hat_dot

def get_f_sys(t: float, x: jax.Array, sys_id: int) -> jax.Array:
    if sys_id == 1: return f_sys_1(t, x)
    if sys_id == 2: return f_sys_2(t, x)
    if sys_id == 3: return f_sys_3(t, x)
    if sys_id == 4: return f_sys_4(t, x)
    if sys_id == 5: return f_sys_5(t, x)
    if sys_id == 6: return f_sys_6(t, x)
    if sys_id == 7: return f_sys_7(t, x)
    if sys_id == 8: return f_sys_8(t, x)
    raise ValueError(f"Invalid sys_id: {sys_id}")

# --- 1. THE CONTINUOUS PHYSICAL PLANT ---
def create_plant_vector_field(sys_id: int):
    """Pure physical dynamics: x_dot = f(x) + u_held"""
    def plant_vector_field(t: float, x: jax.Array, args: tuple):
        u_held = args[0]
        return get_f_sys(t, x, sys_id) + u_held
    return plant_vector_field

# --- 2. THE DISCRETE CONTROLLER STEP ---
def create_discrete_controller(is_integral: bool, sys_id: int, dt_ctrl: float):
    def discrete_step(carry, noise_sample, args):
        t, x_true, theta_hat, I_state = carry
        
        (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
         excitation_duration, k_1, k_2, beta, k_theta_hat, learning_rate, theta_bar, 
         enable_learning, e_0, x_d_dot_0) = args

        x_d = get_desired_trajectory(t, sys_id)
        x_d_dot = get_desired_velocity(t, sys_id)
        
        # Sample noisy measurement
        x_meas = x_true + noise_sample
        e_meas = x_d - x_meas
        u_1 = get_excitation_signal(t, excitation_duration, d_out)
        
        # Construct Constant Diagonal Learning Rate Matrix
        p = theta_hat.shape[0]
        gamma = learning_rate * jnp.eye(p)

        # Active Network Branch
        if enable_learning:
            if is_integral:
                # Integral Controller (Psi inside the integral)
                u = u_1 + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) + I_state
                phi_eval = resnet_network(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                jacobian = compute_jacobian(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas - phi_eval
            else:
                # Baseline Controller (Psi outside the integral)
                phi_eval = resnet_network(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                jacobian = compute_jacobian(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                u = u_1 + phi_eval + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) + I_state
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas
                
            theta_hat_dot = compute_theta_hat_dot(e_meas, theta_hat, jacobian, gamma, k_theta_hat, theta_bar)
            
        # Linear Baseline Branch
        else:
            phi_eval = jnp.zeros(d_out)
            if is_integral:
                u = u_1 + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) + I_state
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas
            else:
                u = u_1 + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) + I_state
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas
                
            theta_hat_dot = jnp.zeros_like(theta_hat)

        # Discrete Euler Integration for Controller States
        theta_next = theta_hat + dt_ctrl * theta_hat_dot
        I_next = I_state + dt_ctrl * I_dot

        # Pack data for logging
        log_data = (t, x_true, theta_hat, gamma, x_d, e_meas, phi_eval, u)
        return (theta_next, I_next, u), log_data
    
    return discrete_step

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    sys_id = config.simulation.sys_id
    
    act_map = {"linear": 0, "swish": 1, "tanh": 2}
    h_act_idx = jnp.array(act_map[config.neural_network.hidden_activation.lower()])
    o_act_idx = jnp.array(act_map[config.neural_network.output_activation.lower()])
    shortcut_act_idx = jnp.array(act_map[config.neural_network.shortcut_activation.lower()])

    p = get_total_parameters(
        config.neural_network.d_in, config.neural_network.hidden_width, 
        config.neural_network.d_out, config.neural_network.b,
        config.neural_network.k_0, config.neural_network.k_i
    )

    # Split seeds
    key = jax.random.PRNGKey(config.simulation.random_seed)
    key, key_theta, key_x0, key_noise = jax.random.split(key, 4)

    theta_hat_0 = jnp.where(
        config.neural_network.init_type == "normal",
        config.neural_network.init_mean + config.neural_network.init_std * jax.random.normal(key_theta, (p,)),
        jnp.zeros((p,))
    )

    d_out = config.neural_network.d_out
    n = config.simulation.state_space_dim
    if config.simulation.randomize_x0:
        x_0 = jax.random.uniform(key_x0, shape=(n,),
                                 minval=-config.simulation.random_x0_square_size,
                                 maxval=config.simulation.random_x0_square_size)
    else:
        yaml_x0 = jnp.array(config.simulation.x0)
        x_0 = jnp.pad(yaml_x0, (0, max(0, n - len(yaml_x0))))[:n]

    # ---------------------------------------------------------
    # BOUNDARY CONDITION INITIALIZATION: Explicit Offsets
    # ---------------------------------------------------------
    x_d_0 = get_desired_trajectory(config.simulation.t0, sys_id)
    x_d_dot_0 = get_desired_velocity(config.simulation.t0, sys_id)
    e_0 = x_d_0 - x_0
    
    # Because the new control laws explicitly subtract e_0 and x_d_dot_0, 
    # we can safely initialize the integral state to pure zeros.
    I_0 = jnp.zeros_like(x_0) 
    
    t0 = config.simulation.t0
    t1 = config.simulation.duration_seconds
    
    # ---------------------------------------------------------
    # DIGITAL CLOCK SETUP
    # ---------------------------------------------------------
    dt_ctrl = 1.0 / config.simulation.control_frequency_hz
    num_steps = int(jnp.ceil((t1 - t0) / dt_ctrl))
    ts = jnp.linspace(t0, t1, num_steps)
    
    # Generate Band-Limited Noise
    noise_mean = config.simulation.noise_mean
    noise_std = config.simulation.noise_std
    raw_noise = noise_std * jax.random.normal(key_noise, shape=(num_steps, n)) + noise_mean
    clip_limit = 3.0 * noise_std
    noise_array = jnp.clip(raw_noise, noise_mean - clip_limit, noise_mean + clip_limit)

    is_integral = config.simulation.controller_type == "nn_in_integral"
    enable_learning = getattr(config.simulation, "enable_learning", True)

    math_args = (
        config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out,
        config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i,
        h_act_idx, o_act_idx, shortcut_act_idx, config.simulation.excitation_duration_seconds,
        config.math_constants.k_1, config.math_constants.k_2, config.math_constants.beta,
        config.math_constants.k_theta_hat, config.math_constants.learning_rate, 
        config.math_constants.theta_bar, enable_learning,
        e_0, x_d_dot_0
    )

    discrete_controller = create_discrete_controller(is_integral, sys_id, dt_ctrl)
    plant_vector_field = create_plant_vector_field(sys_id)
    term = diffrax.ODETerm(plant_vector_field)
    solver = diffrax.Tsit5()

    # ---------------------------------------------------------
    # THE HYBRID SIMULATION LOOP
    # ---------------------------------------------------------
    def hybrid_scan_step(carry, step_data):
        t_current, x_current, theta_hat_curr, I_curr = carry
        noise_samp = step_data
        
        # 1. Evaluate discrete controller
        ctrl_carry = (t_current, x_current, theta_hat_curr, I_curr)
        (theta_next, I_next, u_held), log_data = discrete_controller(ctrl_carry, noise_samp, math_args)
        
        # 2. Check for numerical explosions
        is_invalid = (
            jnp.any(jnp.isnan(x_current)) | jnp.any(jnp.isinf(x_current)) |
            jnp.any(jnp.isnan(u_held)) | jnp.any(jnp.isinf(u_held))
        )
        
        # 3. Conditionally bypass the continuous solver
        def integrate_plant(_):
            sol = diffrax.diffeqsolve(
                term, solver, t0=t_current, t1=t_current + dt_ctrl, dt0=dt_ctrl/10.0, 
                y0=x_current, args=(u_held,),
                stepsize_controller=diffrax.PIDController(rtol=config.simulation.rtol, atol=config.simulation.atol),
                max_steps=config.simulation.max_solver_steps,
                throw=False
            )
            return sol.ys[-1]
            
        def skip_integration(_):
            return jnp.full_like(x_current, jnp.nan)

        x_next = jax.lax.cond(is_invalid, skip_integration, integrate_plant, None)
        
        next_carry = (t_current + dt_ctrl, x_next, theta_next, I_next)
        return next_carry, log_data

    # Initial carry state (Gamma removed)
    init_carry = (t0, x_0, theta_hat_0, I_0)
    
    # Run the massive discrete loop using XLA compilation
    _, log_history = jax.lax.scan(hybrid_scan_step, init_carry, noise_array)
    
    # Unpack logged data (Gamma is still populated dynamically per step for output compatibility)
    (t_out, x_out, theta_hat_out, gamma_out, x_d_out, e_out, phi_eval_out, u_out) = log_history

    return {
        config.data_labels.time: t_out,
        config.data_labels.states: x_out,
        config.data_labels.parameter_estimate: theta_hat_out,
        config.data_labels.learning_rate_matrix: gamma_out,
        config.data_labels.desired_states: x_d_out,
        config.data_labels.tracking_error: e_out,
        config.data_labels.nn_output: phi_eval_out,
        config.data_labels.control_effort: u_out,
    }