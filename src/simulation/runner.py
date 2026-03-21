import diffrax
import jax
import jax.numpy as jnp

from src.conf.config_schema import ExperimentConfig
from src.math.dynamics import (
    get_desired_trajectory,
    get_desired_velocity,
    get_excitation_signal,
    f_sys_1,
    f_sys_2,
    f_sys_3,
    f_sys_4,
    f_sys_5,
    f_sys_6,
    f_sys_7,
    f_sys_8,
    f_sys_9
)
from src.math.networks import compute_jacobian, get_total_parameters, resnet_network
from src.math.update_laws import compute_gamma_dot, compute_theta_hat_dot

def get_f_sys(x: jax.Array, sys_id: int) -> jax.Array:
    if sys_id == 1: return f_sys_1(x)
    if sys_id == 2: return f_sys_2(x)
    if sys_id == 3: return f_sys_3(x)
    if sys_id == 4: return f_sys_4(x)
    if sys_id == 5: return f_sys_5(x)
    if sys_id == 6: return f_sys_6(x)
    if sys_id == 7: return f_sys_7(x)
    if sys_id == 8: return f_sys_8(x)
    if sys_id == 9: return f_sys_9(x)
    raise ValueError(f"Invalid sys_id: {sys_id}")

def create_vector_field(is_integral: bool, sys_id: int, noise_interpolant, enable_learning: bool):
    def vector_field(t: float, y: tuple, args: tuple):
        x_true, theta_hat, gamma, I_state = y
        gamma = 0.5 * (gamma + gamma.T)
        
        # enable_learning is removed from dynamic args
        (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
         excitation_duration, k_1, k_2, beta, k_theta_hat, learning_rate_upper_bound_mult, 
         learning_rate_lower_bound_mult, initial_gamma_scalar, nu, theta_bar, debug_print) = args 

        x_d = get_desired_trajectory(t, sys_id)
        x_d_dot = get_desired_velocity(t, sys_id)
        
        n_t = noise_interpolant.evaluate(t)
        x_meas = x_true + n_t
        e_meas = x_d - x_meas
        
        u_1 = get_excitation_signal(t, excitation_duration, d_out)

        # STATIC PRUNING: XLA only compiles the active branch
        if enable_learning:
            if is_integral:
                u = (k_1 + k_2) * e_meas + x_d_dot + I_state + u_1
                kappa = jnp.concatenate([x_meas, u])
                phi_eval = resnet_network(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                jacobian = compute_jacobian(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas - phi_eval
            else:
                phi_eval = resnet_network(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                jacobian = compute_jacobian(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                u = x_d_dot - phi_eval + (k_1 + k_2) * e_meas + I_state + u_1
                I_dot = (k_1 * k_2 + 1.0) * e_meas + beta * jnp.sign(e_meas)
                
            theta_hat_dot = compute_theta_hat_dot(e_meas, theta_hat, jacobian, gamma, k_theta_hat, theta_bar)
            p = theta_hat.shape[0]
            gamma_dot = compute_gamma_dot(gamma, jacobian, learning_rate_upper_bound_mult, learning_rate_lower_bound_mult, initial_gamma_scalar, nu, p)
            gamma_dot_sym = 0.5 * (gamma_dot + gamma_dot.T)
            
        else:
            # Pure Linear Controller (Zero NN overhead)
            if is_integral:
                u = (k_1 + k_2) * e_meas + x_d_dot + I_state + u_1
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas
            else:
                u = x_d_dot + (k_1 + k_2) * e_meas + I_state + u_1
                I_dot = (k_1 * k_2 + 1.0) * e_meas + beta * jnp.sign(e_meas)
                
            theta_hat_dot = jnp.zeros_like(theta_hat)
            gamma_dot_sym = jnp.zeros_like(gamma)

        x_dot = get_f_sys(x_true, sys_id) + u
        
        return x_dot, theta_hat_dot, gamma_dot_sym, I_dot
    return vector_field

def create_reconstruct_single_step(is_integral: bool, sys_id: int):
    def reconstruct_single_step(t, x_true, theta_hat, I_state, n_t, args):
        (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
         excitation_duration, k_1, k_2, beta) = args
        
        x_d = get_desired_trajectory(t, sys_id)
        x_d_dot = get_desired_velocity(t, sys_id)
        
        x_meas = x_true + n_t
        e_meas = x_d - x_meas
        u_1 = get_excitation_signal(t, excitation_duration, d_out)

        if is_integral:
            u = (k_1 + k_2) * e_meas + x_d_dot + I_state + u_1
            kappa = jnp.concatenate([x_meas, u])
            phi_eval = resnet_network(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
        else:
            phi_eval = resnet_network(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
            u = x_d_dot - phi_eval + (k_1 + k_2) * e_meas + I_state + u_1

        # Evaluate final reconstruction accuracy against true dynamics
        epsilon = phi_eval - get_f_sys(x_true, sys_id)
        return x_d, e_meas, phi_eval, u, epsilon
    return reconstruct_single_step

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    sys_id = config.simulation.sys_id
    
    act_map = {"linear": 0, "swish": 1, "tanh": 2}
    h_act_idx = jnp.array(act_map[config.neural_network.hidden_activation.lower()])
    o_act_idx = jnp.array(act_map[config.neural_network.output_activation.lower()])
    shortcut_act_idx = jnp.array(act_map[config.neural_network.shortcut_activation.lower()])

    p = get_total_parameters(
        config.neural_network.d_in, 
        config.neural_network.hidden_width, 
        config.neural_network.d_out,
        config.neural_network.b,
        config.neural_network.k_0,
        config.neural_network.k_i
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
    if config.simulation.randomize_x0:
        x_0 = jax.random.uniform(key_x0, shape=(d_out,),
                                 minval=-config.simulation.random_x0_square_size,
                                 maxval=config.simulation.random_x0_square_size
                                 )
    else:
        yaml_x0 = jnp.array(config.simulation.x0)
        x_0 = jnp.pad(yaml_x0, (0, max(0, d_out - len(yaml_x0))))[:d_out]

    gamma_0 = config.math_constants.initial_gamma_scalar * jnp.eye(p)
    I_0 = jnp.zeros_like(x_0)
    y0 = (x_0, theta_hat_0, gamma_0, I_0)
    
    t0 = config.simulation.t0
    t1 = config.simulation.duration_seconds
    
    # Generate Band-Limited Noise
    noise_mean = config.simulation.noise_mean
    noise_std = config.simulation.noise_std
    noise_freq = config.simulation.noise_freq
    num_noise_steps = int(jnp.ceil((t1 - t0) * noise_freq)) + 1
    noise_ts = jnp.linspace(t0, t1, num_noise_steps)
    raw_noise = noise_std * jax.random.normal(key_noise, shape=(num_noise_steps, d_out)) + noise_mean
    noise_interpolant = diffrax.LinearInterpolation(ts=noise_ts, ys=raw_noise)

    is_integral = config.simulation.controller_type == "nn_in_integral"

    math_args = (
        config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out,
        config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i,
        h_act_idx, o_act_idx, shortcut_act_idx, config.simulation.excitation_duration_seconds,
        config.math_constants.k_1, config.math_constants.k_2, config.math_constants.beta,
        config.math_constants.k_theta_hat, config.math_constants.learning_rate_upper_bound_mult,
        config.math_constants.learning_rate_lower_bound_mult, config.math_constants.initial_gamma_scalar,
        config.math_constants.nu, config.math_constants.theta_bar, config.simulation.debug_print
    )

    enable_learning = config.simulation.enable_learning
    vector_field = create_vector_field(is_integral, sys_id, noise_interpolant, enable_learning)
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    
    save_interval = config.simulation.save_interval_seconds
    num_save_steps = int(round((t1 - t0) / save_interval)) + 1
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, num_save_steps))
    stepsize_controller = diffrax.PIDController(rtol=config.simulation.rtol, atol=config.simulation.atol)

    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=save_interval, y0=y0, args=math_args,
        saveat=saveat, stepsize_controller=stepsize_controller,
        progress_meter=diffrax.NoProgressMeter(), max_steps=config.simulation.max_solver_steps 
    )

    if sol.result != diffrax.RESULTS.successful:
        raise RuntimeError(f"SIMULATION FAILED: Diffrax Error {sol.result}")

    t_out = sol.ts
    x_out, theta_hat_out, gamma_out, I_out = sol.ys

    recon_args = (
        config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out,
        config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i,
        h_act_idx, o_act_idx, shortcut_act_idx, config.simulation.excitation_duration_seconds,
        config.math_constants.k_1, config.math_constants.k_2, config.math_constants.beta
    )

    # Reconstruct data matching physical times
    reconstruct_single_step = create_reconstruct_single_step(is_integral, sys_id)
    vmap_reconstruct = jax.vmap(reconstruct_single_step, in_axes=(0, 0, 0, 0, 0, None))
    
    n_out = jax.vmap(noise_interpolant.evaluate)(t_out)
    x_d_out, e_out, phi_eval_out, u_out, epsilon_out = vmap_reconstruct(t_out, x_out, theta_hat_out, I_out, n_out, recon_args)
    
    return {
        config.data_labels.time: t_out,
        config.data_labels.states: x_out,
        config.data_labels.parameter_estimate: theta_hat_out,
        config.data_labels.learning_rate_matrix: gamma_out,
        config.data_labels.desired_states: x_d_out,
        config.data_labels.tracking_error: e_out,
        config.data_labels.nn_output: phi_eval_out,
        config.data_labels.control_effort: u_out,
        config.data_labels.reconstruction_error: epsilon_out
    }