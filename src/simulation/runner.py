import diffrax
import jax
import jax.numpy as jnp

from src.conf.config_schema import ExperimentConfig
from src.math.dynamics import (
    desired_trajectory,
    desired_velocity,
    excitation_signal,
    f_sys_1,
    f_sys_2,
    f_sys_3,
)
from src.math.networks import compute_jacobian, get_total_parameters, resnet_network
from src.math.update_laws import compute_gamma_dot, compute_theta_hat_dot


def get_f_sys(x: jax.Array, sys_id: int) -> jax.Array:
    return jax.lax.switch(
        sys_id - 1,
        [f_sys_1, f_sys_2, f_sys_3],
        x
    )


def create_vector_field(is_integral: bool):
    def vector_field(
        t: float,
        y: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        args: tuple
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        x, theta_hat, gamma, I_state = y
        gamma = 0.5 * (gamma + gamma.T)
        
        (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
         excitation_duration, k_1, k_2, beta, k_theta_hat, learning_rate_upper_bound_mult, 
         learning_rate_lower_bound_mult, initial_gamma_scalar, nu, theta_bar, debug_print, 
         sys_id) = args

        x_d = desired_trajectory(t)
        x_d_dot = desired_velocity(t)
        e = x_d - x
        u_1 = excitation_signal(t, excitation_duration)

        if is_integral:
            u = (k_1 + k_2) * e + x_d_dot + I_state + u_1
            kappa = jnp.concatenate([x, u])
            
            phi_eval = resnet_network(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
            jacobian = compute_jacobian(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
            
            I_dot = beta * jnp.sign(e) + (k_1 * k_2 + 1.0) * e - phi_eval
        else:
            phi_eval = resnet_network(theta_hat, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
            jacobian = compute_jacobian(theta_hat, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
            
            u = x_d_dot - phi_eval + (k_1 + k_2) * e + I_state + u_1
            I_dot = (k_1 * k_2 + 1.0) * e + beta * jnp.sign(e)

        x_dot = get_f_sys(x, sys_id) + u
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

        return x_dot, theta_hat_dot, gamma_dot_sym, I_dot

    return vector_field


def create_reconstruct_single_step(is_integral: bool):
    def reconstruct_single_step(
        t: float,
        x: jax.Array,
        theta_hat: jax.Array,
        I_state: jax.Array,
        args: tuple
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        
        (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
         excitation_duration, k_1, k_2, beta, sys_id) = args
        
        x_d = desired_trajectory(t)
        x_d_dot = desired_velocity(t)
        e = x_d - x
        u_1 = excitation_signal(t, excitation_duration)

        if is_integral:
            u = (k_1 + k_2) * e + x_d_dot + I_state + u_1
            kappa = jnp.concatenate([x, u])
            phi_eval = resnet_network(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
        else:
            phi_eval = resnet_network(theta_hat, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
            u = x_d_dot - phi_eval + (k_1 + k_2) * e + I_state + u_1

        f_eval = get_f_sys(x, sys_id)
        epsilon = phi_eval - f_eval

        return x_d, e, phi_eval, u, epsilon

    return reconstruct_single_step


def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    sys_id = getattr(config.simulation, "sys_id", 1)
    
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

    # 1. Split the seed into two independent PRNG keys
    key = jax.random.PRNGKey(config.simulation.random_seed)
    key_theta, key_x0 = jax.random.split(key)

    # 2. Initialize weights using the first key
    theta_hat_0 = jnp.where(
        config.neural_network.init_type == "normal",
        config.neural_network.init_mean + config.neural_network.init_std * jax.random.normal(key_theta, (p,)),
        jnp.zeros((p,))
    )

    # 3. Initialize state using the boolean flag and the second key
    if config.simulation.randomize_x0:
        # Uniform distribution between -2.5 and 2.5
        x_0 = jax.random.uniform(key_x0, shape=(2,), minval=-2.5, maxval=2.5)
    else:
        # Strict deterministic coordinate from config.yaml
        x_0 = jnp.array(config.simulation.x0)

    gamma_0 = config.math_constants.initial_gamma_scalar * jnp.eye(p)
    I_0 = jnp.zeros_like(x_0)
    
    y0 = (x_0, theta_hat_0, gamma_0, I_0)
    is_integral = config.simulation.controller_type == "nn_in_integral"

    math_args = (
        config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out,
        config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i,
        h_act_idx, o_act_idx, shortcut_act_idx, config.simulation.excitation_duration_seconds,
        config.math_constants.k_1, config.math_constants.k_2, config.math_constants.beta,
        config.math_constants.k_theta_hat, config.math_constants.learning_rate_upper_bound_mult,
        config.math_constants.learning_rate_lower_bound_mult, config.math_constants.initial_gamma_scalar,
        config.math_constants.nu, config.math_constants.theta_bar, config.simulation.debug_print,
        sys_id
    )

    vector_field = create_vector_field(is_integral)
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
        progress_meter=diffrax.TqdmProgressMeter(), max_steps=config.simulation.max_solver_steps 
    )

    if sol.result != diffrax.RESULTS.successful:
        raise RuntimeError(f"SIMULATION FAILED: Diffrax Error {sol.result}")

    t_out = sol.ts
    x_out, theta_hat_out, gamma_out, I_out = sol.ys

    recon_args = (
        config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out,
        config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i,
        h_act_idx, o_act_idx, shortcut_act_idx, config.simulation.excitation_duration_seconds,
        config.math_constants.k_1, config.math_constants.k_2, config.math_constants.beta,
        sys_id
    )

    reconstruct_single_step = create_reconstruct_single_step(is_integral)
    vmap_reconstruct = jax.vmap(reconstruct_single_step, in_axes=(0, 0, 0, 0, None))
    x_d_out, e_out, phi_eval_out, u_out, epsilon_out = vmap_reconstruct(t_out, x_out, theta_hat_out, I_out, recon_args)

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