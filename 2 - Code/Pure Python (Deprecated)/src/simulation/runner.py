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
from src.math.update_laws import compute_theta_hat_dot, setup_rho_filter, update_rho_filter

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
def create_discrete_controller(ctrl_type: str, sys_id: int, dt_ctrl: float):
    def discrete_step(carry, noise_sample, args):
        t, x_true, theta_hat, I_state, zeta = carry

        (d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx,
         excitation_duration, k_1, k_2, beta, k_theta_hat, learning_rate, theta_bar,
         enable_learning, e_0, x_d_dot_0, A_d, B_q, B_s, rho_k4) = args

        x_d = get_desired_trajectory(t, sys_id)
        x_d_dot = get_desired_velocity(t, sys_id)

        x_meas = x_true + noise_sample
        e_meas = x_d - x_meas
        u_1 = get_excitation_signal(t, excitation_duration, d_out)

        p = theta_hat.shape[0]
        gamma = learning_rate * jnp.eye(p)

        # --- RHO FILTER PROPAGATION ---
        next_zeta, e_hat_dot_rho = update_rho_filter(zeta, e_meas, A_d, B_q, B_s, rho_k4)

        # Composite error signal r (used by _r variants)
        r = e_meas + k_1 * e_hat_dot_rho
        # Velocity estimate x_hat_dot (used by direct_r)
        x_hat_dot = x_d_dot - e_hat_dot_rho

        if enable_learning:
            if ctrl_type.startswith("integral_"):
                # Controllers 1 & 2: NN inside the integral, kappa = [x, u]
                u_next = u_1 + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) + I_state
                kappa = jnp.concatenate([x_meas, u_next])
                phi_eval = resnet_network(theta_hat, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas - phi_eval
                jac = jax.jacobian(lambda th: resnet_network(th, kappa, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx))(theta_hat)
                driving_signal = r if ctrl_type == "integral_r" else e_meas
                theta_hat_dot = compute_theta_hat_dot(driving_signal, theta_hat, jac, gamma, k_theta_hat, theta_bar)

            else:  # direct_r or direct_e
                # Controllers 3 & 4: NN outside the integral, input = x
                phi_eval = resnet_network(theta_hat, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                u_next = u_1 + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) - phi_eval + I_state
                I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas

                if ctrl_type == "direct_r":
                    # Regressor Omega = (dPhi/dx) * x_hat_dot; update with d(Omega)/d(theta)
                    def phi_fn(th, x_val):
                        return resnet_network(th, x_val, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx)
                    def omega_fn(th, x_val, x_dot_val):
                        dPhi_dx = jax.jacobian(phi_fn, argnums=1)(th, x_val)
                        return dPhi_dx @ x_dot_val
                    jac_update = jax.jacobian(omega_fn, argnums=0)(theta_hat, x_meas, x_hat_dot)
                    theta_hat_dot = compute_theta_hat_dot(r, theta_hat, jac_update, gamma, k_theta_hat, theta_bar)
                else:  # direct_e
                    jac = jax.jacobian(lambda th: resnet_network(th, x_meas, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx))(theta_hat)
                    theta_hat_dot = compute_theta_hat_dot(e_meas, theta_hat, jac, gamma, k_theta_hat, theta_bar)

        else:
            # Linear baseline branch (no learning)
            phi_eval = jnp.zeros(d_out)
            u_next = u_1 + (k_1 + k_2) * (e_meas - e_0) + (x_d_dot - x_d_dot_0) + I_state
            I_dot = beta * jnp.sign(e_meas) + (k_1 * k_2 + 1.0) * e_meas
            theta_hat_dot = jnp.zeros_like(theta_hat)

        # Discrete Euler integration for controller states
        theta_next = theta_hat + dt_ctrl * theta_hat_dot
        I_next = I_state + dt_ctrl * I_dot

        log_data = (t, x_true, theta_hat, gamma, x_d, e_meas, phi_eval, u_next)
        return (theta_next, I_next, next_zeta, u_next), log_data

    return discrete_step

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    sys_id = config.simulation.sys_id
    
    act_map = {"linear": 0, "swish": 1, "tanh": 2}
    h_act_idx = jnp.array(act_map[config.neural_network.hidden_activation.lower()])
    o_act_idx = jnp.array(act_map[config.neural_network.output_activation.lower()])
    shortcut_act_idx = jnp.array(act_map[config.neural_network.shortcut_activation.lower()])

    n = config.simulation.state_space_dim
    p = get_total_parameters(
        config.neural_network.d_in, config.neural_network.hidden_width,
        n, config.neural_network.b,
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

    if config.simulation.randomize_x0:
        x_0 = jax.random.uniform(key_x0, shape=(n,),
                                 minval=-config.simulation.random_x0_square_size,
                                 maxval=config.simulation.random_x0_square_size)
    else:
        yaml_x0 = jnp.array(config.simulation.x0)
        x_0 = jnp.pad(yaml_x0, (0, max(0, n - len(yaml_x0))))[:n]

    # Boundary Conditions
    x_d_0 = get_desired_trajectory(config.simulation.t0, sys_id)
    x_d_dot_0 = get_desired_velocity(config.simulation.t0, sys_id)
    e_0 = x_d_0 - x_0
    I_0 = jnp.zeros_like(x_0) 
    
    t0 = config.simulation.t0
    t1 = config.simulation.duration_seconds
    
    # Digital Clock
    dt_ctrl = 1.0 / config.simulation.control_frequency_hz
    num_steps = int(jnp.ceil((t1 - t0) / dt_ctrl))
    
    # Noise
    noise_mean = config.simulation.noise_mean
    noise_std = config.simulation.noise_std
    raw_noise = noise_std * jax.random.normal(key_noise, shape=(num_steps, n)) + noise_mean
    clip_limit = 3.0 * noise_std
    noise_array = jnp.clip(raw_noise, noise_mean - clip_limit, noise_mean + clip_limit)

    ctrl_type = config.simulation.controller_type
    enable_learning = getattr(config.simulation, "enable_learning", True)

    # --- SETUP STATIC RHO FILTER MATRICES ---
    A_d, B_q, B_s = setup_rho_filter(
        config.math_constants.rho_k1, 
        config.math_constants.rho_k2, 
        config.math_constants.rho_k3, 
        dt_ctrl
    )
    
    # Initial state is a (4, n) matrix of zeros
    zeta_0 = jnp.zeros((4, n))

    # --- SETUP STATIC RHO FILTER MATRICES ---
    A_d, B_q, B_s = setup_rho_filter(
        config.math_constants.rho_k1, 
        config.math_constants.rho_k2, 
        config.math_constants.rho_k3, 
        dt_ctrl
    )
    
    # Initial state is a (4, n) matrix of zeros
    zeta_0 = jnp.zeros((4, n))

    math_args = (
        config.neural_network.d_in, config.neural_network.hidden_width, config.neural_network.d_out,
        config.neural_network.b, config.neural_network.k_0, config.neural_network.k_i,
        h_act_idx, o_act_idx, shortcut_act_idx, config.simulation.excitation_duration_seconds,
        config.math_constants.k_1, config.math_constants.k_2, config.math_constants.beta,
        config.math_constants.k_theta_hat, config.math_constants.learning_rate, 
        config.math_constants.theta_bar, enable_learning,
        e_0, x_d_dot_0,
        A_d, B_q, B_s, config.math_constants.rho_k4
    )

    discrete_controller = create_discrete_controller(ctrl_type, sys_id, dt_ctrl)
    plant_vector_field = create_plant_vector_field(sys_id)
    term = diffrax.ODETerm(plant_vector_field)
    solver = diffrax.Tsit5()

    # ---------------------------------------------------------
    # THE HYBRID SIMULATION LOOP
    # ---------------------------------------------------------
    def hybrid_scan_step(carry, step_data):
        # 1. Unpack 5-element carry
        t_current, x_current, theta_hat_curr, I_curr, zeta_curr = carry
        # 1. Unpack 5-element carry
        t_current, x_current, theta_hat_curr, I_curr, zeta_curr = carry
        noise_samp = step_data
        
        # 2. Evaluate discrete controller (returns 4-element next-state tuple + log data)
        ctrl_carry = (t_current, x_current, theta_hat_curr, I_curr, zeta_curr)
        (theta_next, I_next, zeta_next, u_held), log_data = discrete_controller(ctrl_carry, noise_samp, math_args)
        # 2. Evaluate discrete controller (returns 4-element next-state tuple + log data)
        ctrl_carry = (t_current, x_current, theta_hat_curr, I_curr, zeta_curr)
        (theta_next, I_next, zeta_next, u_held), log_data = discrete_controller(ctrl_carry, noise_samp, math_args)
        
        # 3. Check for numerical explosions
        # 3. Check for numerical explosions
        is_invalid = (
            jnp.any(jnp.isnan(x_current)) | jnp.any(jnp.isinf(x_current)) |
            jnp.any(jnp.isnan(u_held)) | jnp.any(jnp.isinf(u_held))
        )
        
        # 4. Conditionally bypass the continuous solver
        # 4. Conditionally bypass the continuous solver
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
        
        # 5. Pack 5-element carry for next step
        next_carry = (t_current + dt_ctrl, x_next, theta_next, I_next, zeta_next)
        # 5. Pack 5-element carry for next step
        next_carry = (t_current + dt_ctrl, x_next, theta_next, I_next, zeta_next)
        return next_carry, log_data

    # Initial 5-element carry state
    init_carry = (t0, x_0, theta_hat_0, I_0, zeta_0)
    # Initial 5-element carry state
    init_carry = (t0, x_0, theta_hat_0, I_0, zeta_0)
    
    # Run the massive discrete loop using XLA compilation
    _, log_history = jax.lax.scan(hybrid_scan_step, init_carry, noise_array)
    
    # Unpack logged data
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