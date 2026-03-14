import diffrax
import jax
import jax.numpy as jnp

from src.conf.config_schema import ExperimentConfig
from src.math.dynamics import (
    desired_trajectory,
    desired_velocity,
    excitation_signal,
    f_sys,
)
from src.math.networks import compute_jacobian, get_total_parameters, resnet_network
from src.math.update_laws import (
    compute_controller_in_integral,
    compute_controller_outside_integral,
    compute_gamma_dot,
    compute_theta_hat_dot,
)

def vector_field(
    t: float,
    y: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    args: tuple 
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x, theta_hat, gamma, z = y
    gamma = 0.5 * (gamma + gamma.T)
    
    (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
     excitation_duration, k_1, k_2, beta, k_theta_hat, learning_rate_upper_bound_mult, 
     learning_rate_lower_bound_mult, initial_gamma_scalar, nu, theta_bar, debug_print, 
     controller_flag) = args

    x_d = desired_trajectory(t)
    x_d_dot = desired_velocity(t)
    e = x_d - x

    phi_eval = resnet_network(theta_hat, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
    jacobian = compute_jacobian(theta_hat, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)

    u_1 = excitation_signal(t, excitation_duration)

    def in_integral_branch(_: None) -> tuple[jax.Array, jax.Array]:
        return compute_controller_in_integral(e, x_d_dot, z, u_1, phi_eval, k_1, k_2, beta)

    def outside_integral_branch(_: None) -> tuple[jax.Array, jax.Array]:
        return compute_controller_outside_integral(e, x_d_dot, z, u_1, phi_eval, k_1, k_2, beta)

    u, z_dot = jax.lax.cond(
        controller_flag == 1,
        in_integral_branch,
        outside_integral_branch,
        None
    )

    x_dot = f_sys(x) + u
    theta_hat_dot = compute_theta_hat_dot(e, theta_hat, jacobian, gamma, k_theta_hat, theta_bar)
    
    p = theta_hat.shape[0]
    gamma_dot = compute_gamma_dot(gamma, jacobian, learning_rate_upper_bound_mult, learning_rate_lower_bound_mult, initial_gamma_scalar, nu, p)
    gamma_dot_sym = 0.5 * (gamma_dot + gamma_dot.T)

    jax.lax.cond(
        debug_print,
        lambda _: jax.debug.print("t: {t} | ||e||: {e_norm} | ||u||: {u_norm}", 
                                  t=t, e_norm=jnp.linalg.norm(e), u_norm=jnp.linalg.norm(u)),
        lambda _: None, None
    )

    return x_dot, theta_hat_dot, gamma_dot_sym, z_dot

def reconstruct_single_step(
    t: float,
    x: jax.Array,
    theta_hat: jax.Array,
    z: jax.Array,
    args: tuple
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    
    (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
     excitation_duration, k_1, k_2, beta, controller_flag) = args
    
    x_d = desired_trajectory(t)
    x_d_dot = desired_velocity(t)
    e = x_d - x

    phi_eval = resnet_network(theta_hat, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
    u_1 = excitation_signal(t, excitation_duration)
    
    def in_integral_branch(_: None) -> tuple[jax.Array, jax.Array]:
        return compute_controller_in_integral(e, x_d_dot, z, u_1, phi_eval, k_1, k_2, beta)

    def outside_integral_branch(_: None) -> tuple[jax.Array, jax.Array]:
        return compute_controller_outside_integral(e, x_d_dot, z, u_1, phi_eval, k_1, k_2, beta)

    u, _ = jax.lax.cond(
        controller_flag == 1,
        in_integral_branch,
        outside_integral_branch,
        None
    )

    f_eval = f_sys(x)
    epsilon = phi_eval - f_eval

    return x_d, e, phi_eval, u, epsilon

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
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

    key = jax.random.PRNGKey(config.simulation.random_seed)
    if config.neural_network.init_type == "normal":
        theta_hat_0 = config.neural_network.init_mean + config.neural_network.init_std * jax.random.normal(key, (p,))
    else:
        theta_hat_0 = jnp.zeros((p,))

    x_0 = jnp.array(config.simulation.x0)
    gamma_0 = config.math_constants.initial_gamma_scalar * jnp.eye(p)
    
    controller_flag = jnp.array(1 if config.simulation.controller_type == "nn_in_integral" else 0)
    
    e_0 = desired_trajectory(config.simulation.t0) - x_0
    x_d_dot_0 = desired_velocity(config.simulation.t0)
    phi_eval_0 = resnet_network(theta_hat_0, x_0, config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out, config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i, h_act_idx, o_act_idx, shortcut_act_idx)
    u_1_0 = excitation_signal(config.simulation.t0, config.simulation.excitation_duration_seconds)

    def init_in_integral(_: None) -> jax.Array:
        return - (config.math_constants.k_1 + config.math_constants.k_2) * e_0 - x_d_dot_0 - u_1_0

    def init_outside_integral(_: None) -> jax.Array:
        return - x_d_dot_0 - phi_eval_0 - (config.math_constants.k_1 + config.math_constants.k_2) * e_0 - u_1_0

    z_0 = jax.lax.cond(
        controller_flag == 1,
        init_in_integral,
        init_outside_integral,
        None
    )

    y0 = (x_0, theta_hat_0, gamma_0, z_0)

    math_args = (
        config.neural_network.d_in,
        config.neural_network.hidden_width,
        config.neural_network.d_out,
        config.neural_network.b,
        config.neural_network.k_0,
        config.neural_network.k_i,
        h_act_idx,
        o_act_idx,
        shortcut_act_idx,
        config.simulation.excitation_duration_seconds,
        config.math_constants.k_1,
        config.math_constants.k_2,
        config.math_constants.beta,
        config.math_constants.k_theta_hat,
        config.math_constants.learning_rate_upper_bound_mult,
        config.math_constants.learning_rate_lower_bound_mult,
        config.math_constants.initial_gamma_scalar,
        config.math_constants.nu,
        config.math_constants.theta_bar,
        config.simulation.debug_print,
        controller_flag
    )

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    
    t0 = config.simulation.t0
    t1 = config.simulation.duration_seconds
    save_interval = config.simulation.save_interval_seconds
    num_save_steps = int(round((t1 - t0) / save_interval)) + 1
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, num_save_steps))
    stepsize_controller = diffrax.PIDController(rtol=config.simulation.rtol, atol=config.simulation.atol)

    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=save_interval, y0=y0, args=math_args,
        saveat=saveat, stepsize_controller=stepsize_controller,
        progress_meter=diffrax.TextProgressMeter(), max_steps=config.simulation.max_solver_steps 
    )

    if sol.result != diffrax.RESULTS.successful:
        raise RuntimeError(f"SIMULATION FAILED: Diffrax Error {sol.result}")

    t_out = sol.ts
    x_out, theta_hat_out, gamma_out, z_out = sol.ys

    recon_args = (
        config.neural_network.d_in,
        config.neural_network.hidden_width,
        config.neural_network.d_out,
        config.neural_network.b,
        config.neural_network.k_0,
        config.neural_network.k_i,
        h_act_idx,
        o_act_idx,
        shortcut_act_idx,
        config.simulation.excitation_duration_seconds,
        config.math_constants.k_1,
        config.math_constants.k_2,
        config.math_constants.beta,
        controller_flag
    )

    vmap_reconstruct = jax.vmap(reconstruct_single_step, in_axes=(0, 0, 0, 0, None))
    x_d_out, e_out, phi_eval_out, u_out, epsilon_out = vmap_reconstruct(t_out, x_out, theta_hat_out, z_out, recon_args)

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