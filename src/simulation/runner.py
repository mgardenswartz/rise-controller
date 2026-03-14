import diffrax
import jax
import jax.numpy as jnp

from src.core.config_schema import ExperimentConfig
from src.math.dynamics import (
    desired_trajectory,
    desired_velocity,
    excitation_signal,
    f_sys,
    g_sys,
)
from src.math.networks import get_total_parameters, phi_network
from src.math.update_laws import (
    compute_control_input,
    compute_gamma_dot,
    compute_theta_hat_dot,
)

def get_jacobian(
    theta_hat: jax.Array,
    x: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    num_layers: int,
    h_act_idx: jax.Array,
    o_act_idx: jax.Array
) -> jax.Array:
    return jax.jacfwd(phi_network, argnums=0)(
        theta_hat, x, d_in, hidden_width, d_out, num_layers, h_act_idx, o_act_idx
    )

def vector_field(
    t: float,
    y: tuple[jax.Array, jax.Array, jax.Array],
    args: tuple # Simplified type hint for brevity
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, theta_hat, gamma = y
    
    (d_in, hidden_width, d_out, num_layers, h_act_idx, o_act_idx, 
     excitation_duration, k_e, k_theta_hat, gamma_bar_upper, 
     gamma_bar_lower, nu, theta_bar, debug_print) = args

    x_d = desired_trajectory(t)
    x_d_dot = desired_velocity(t)
    e = x - x_d

    phi_eval = phi_network(theta_hat, x, d_in, hidden_width, d_out, num_layers, h_act_idx, o_act_idx)
    jacobian = get_jacobian(theta_hat, x, d_in, hidden_width, d_out, num_layers, h_act_idx, o_act_idx)

    u_1 = excitation_signal(t, excitation_duration)
    u = compute_control_input(x, x_d_dot, e, phi_eval, u_1, k_e, g_sys)

    x_dot = f_sys(x) + jnp.dot(g_sys(x), u)
    theta_hat_dot = compute_theta_hat_dot(e, theta_hat, jacobian, gamma, k_theta_hat, theta_bar)
    
    p = theta_hat.shape[0]
    gamma_dot = compute_gamma_dot(gamma, jacobian, gamma_bar_upper, gamma_bar_lower, nu, p)
    gamma_dot_sym = 0.5 * (gamma_dot + gamma_dot.T)

    jax.lax.cond(
        debug_print,
        lambda _: jax.debug.print("t: {t} | ||x||: {x_norm} | ||u||: {u_norm}", 
                                  t=t, x_norm=jnp.linalg.norm(x), u_norm=jnp.linalg.norm(u)),
        lambda _: None, None
    )

    return x_dot, theta_hat_dot, gamma_dot_sym

def reconstruct_single_step(
    t: float,
    x: jax.Array,
    theta_hat: jax.Array,
    args: tuple
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    
    (d_in, hidden_width, d_out, num_layers, h_act_idx, o_act_idx, excitation_duration, k_e) = args
    
    x_d = desired_trajectory(t)
    x_d_dot = desired_velocity(t)
    e = x - x_d

    phi_eval = phi_network(theta_hat, x, d_in, hidden_width, d_out, num_layers, h_act_idx, o_act_idx)
    u_1 = excitation_signal(t, excitation_duration)
    u = compute_control_input(x, x_d_dot, e, phi_eval, u_1, k_e, g_sys)
    f_eval = f_sys(x)
    epsilon = phi_eval - f_eval

    return x_d, e, phi_eval, u, epsilon

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    
    # Map the YAML strings to JAX-compatible integer arrays
    act_map = {"linear": 0, "swish": 1, "tanh": 2}
    h_act_idx = jnp.array(act_map[config.neural_network.hidden_activation.lower()])
    o_act_idx = jnp.array(act_map[config.neural_network.output_activation.lower()])

    p = get_total_parameters(
        config.neural_network.d_in, 
        config.neural_network.hidden_width, 
        config.neural_network.d_out,
        config.neural_network.num_layers
    )

    key = jax.random.PRNGKey(config.simulation.random_seed)
    if config.neural_network.init_type == "normal":
        theta_hat_0 = config.neural_network.init_mean + config.neural_network.init_std * jax.random.normal(key, (p,))
    else:
        theta_hat_0 = jnp.zeros((p,))

    x_0 = jnp.array(config.simulation.x0)
    gamma_0 = config.math_constants.initial_gamma_scalar * jnp.eye(p)
    y0 = (x_0, theta_hat_0, gamma_0)

    # Expand math_args to include the network topology and activations
    math_args = (
        config.neural_network.d_in,
        config.neural_network.hidden_width,
        config.neural_network.d_out,
        config.neural_network.num_layers,
        h_act_idx,
        o_act_idx,
        config.simulation.excitation_duration_seconds,
        config.math_constants.k_e,
        config.math_constants.k_theta_hat,
        config.math_constants.gamma_bar_upper,
        config.math_constants.gamma_bar_lower,
        config.math_constants.nu,
        config.math_constants.theta_bar,
        config.simulation.debug_print
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
    x_out, theta_hat_out, gamma_out = sol.ys

    recon_args = (
        config.neural_network.d_in,
        config.neural_network.hidden_width,
        config.neural_network.d_out,
        config.neural_network.num_layers,
        h_act_idx,
        o_act_idx,
        config.simulation.excitation_duration_seconds,
        config.math_constants.k_e
    )

    vmap_reconstruct = jax.vmap(reconstruct_single_step, in_axes=(0, 0, 0, None))
    x_d_out, e_out, phi_eval_out, u_out, epsilon_out = vmap_reconstruct(t_out, x_out, theta_hat_out, recon_args)

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