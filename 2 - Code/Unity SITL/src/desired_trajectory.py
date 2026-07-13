import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import math
from typing import Tuple, Any
from functools import partial

class TrajectoryGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.desired_traj = config['desired_trajectory']
        
        # Traj 1 params
        self.traj1_target_z = config['traj1_center_z_m_ned_aviary']
        self.traj1_period = config['traj1_period_s']
        self.traj1_x_amp = config['traj1_x_amp_m_ned_aviary']
        self.traj1_y_amp = config['traj1_y_amp_m_ned_aviary']
        self.traj1_z_amp = config['traj1_z_amp_m_ned_aviary']
        self.traj1_alpha = config['traj1_alpha_warp']
        self.traj1_warp_c = 1.0 / math.sqrt(1.0 - self.traj1_alpha) if self.traj1_alpha < 1.0 else 1.0
        
        # Traj 2 params
        self.traj2_target_z = config['traj2_center_z_m_ned_aviary']
        self.traj2_A = config['traj2_petal_radius_m']
        self.traj2_v = config['traj2_target_speed_mps']

        self.max_sim_time = config['sim_length_s'] +5
        self._precompute_phases()

        # Warmup JIT functions
        _ = self._get_traj1_jax(0.0)
        _ = self._get_traj2_jax(0.0)

    def _precompute_phases(self) -> None:
        """Numerically integrates the 1D phase variables once during initialization."""
        # Trajectory 1: Integrate d(tau)/dt
        w = (2.0 * math.pi) / self.traj1_period
        def dtau_dt(t: float, tau: np.ndarray) -> float:
            return self.traj1_warp_c * (1.0 - self.traj1_alpha * math.sin(w * tau[0])**2) # type: ignore
        
        sol1 = solve_ivp(dtau_dt, [0, self.max_sim_time], [0.0], max_step=0.01)
        self.t_grid_1 = jnp.array(sol1.t)
        self.tau_grid = jnp.array(sol1.y[0])

        # Trajectory 2: Integrate d(theta)/dt
        def dtheta_dt(t: float, theta: np.ndarray) -> float:
            f_theta = 1.0 + 3.0 * math.sin(2.0 * theta[0])**2
            return self.traj2_v / (self.traj2_A * math.sqrt(f_theta)) # type: ignore
        
        sol2 = solve_ivp(dtheta_dt, [0, self.max_sim_time], [0.0], max_step=0.01)
        self.t_grid_2 = jnp.array(sol2.t)
        self.theta_grid = jnp.array(sol2.y[0])

    @partial(jax.jit, static_argnums=(0,))
    def _get_traj1_jax(self, t: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # 1. Look up the exact phase (tau) for the current time
        tau = jnp.interp(t, self.t_grid_1, self.tau_grid)
        
        w = (2.0 * jnp.pi) / self.traj1_period
        wx, wy, wz = 2.0 * w, 1.0 * w, 4.0 * w

        # 2. Compute analytical temporal derivatives of tau
        tau_dot = self.traj1_warp_c * (1.0 - self.traj1_alpha * (jnp.sin(w * tau)**2))
        tau_ddot = -2.0 * self.traj1_warp_c * self.traj1_alpha * w * jnp.sin(w * tau) * jnp.cos(w * tau) * tau_dot

        def pos_fn(tau_val: jax.Array) -> jax.Array:
            return jnp.array([
                self.traj1_x_amp * jnp.sin(wx * tau_val),
                self.traj1_y_amp * jnp.sin(wy * tau_val),
                self.traj1_z_amp * jnp.sin(wz * tau_val) + self.traj1_target_z
            ])
            
        # 3. Apply the exact chain rule
        pos = pos_fn(tau)
        dp_dtau = jax.jacfwd(pos_fn)(tau)
        d2p_dtau2 = jax.jacfwd(jax.jacfwd(pos_fn))(tau)
        
        vel = dp_dtau * tau_dot
        acc = d2p_dtau2 * (tau_dot**2) + dp_dtau * tau_ddot
        return pos, vel, acc

    @partial(jax.jit, static_argnums=(0,))
    def _get_traj2_jax(self, t: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # 1. Look up the exact phase (theta) for the current time
        theta = jnp.interp(t, self.t_grid_2, self.theta_grid)
        
        # 2. Compute analytical temporal derivatives of theta
        f_theta = 1.0 + 3.0 * (jnp.sin(2.0 * theta)**2)
        theta_dot = self.traj2_v / (self.traj2_A * jnp.sqrt(f_theta))
        sin_4theta = jnp.sin(4.0 * theta)
        theta_ddot = - (3.0 * (self.traj2_v**2) * sin_4theta) / ((self.traj2_A**2) * (f_theta**2))
        
        def pos_fn(th: jax.Array) -> jax.Array:
            r = self.traj2_A * jnp.cos(2.0 * th)
            return jnp.array([
                r * jnp.cos(th),
                r * jnp.sin(th),
                self.traj2_target_z
            ])
            
        # 3. Apply the exact chain rule
        pos = pos_fn(theta)
        dp_dth = jax.jacfwd(pos_fn)(theta)
        d2p_dth2 = jax.jacfwd(jax.jacfwd(pos_fn))(theta)
        
        vel = dp_dth * theta_dot
        acc = d2p_dth2 * (theta_dot**2) + dp_dth * theta_ddot
        return pos, vel, acc

    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.desired_traj == 1:
            pos, vel, acc = self._get_traj1_jax(t)
        else:
            pos, vel, acc = self._get_traj2_jax(t)
            
        return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64), np.array(acc, dtype=np.float64)