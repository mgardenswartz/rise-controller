import jax
import jax.numpy as jnp
import math
import numpy as np
from typing import Tuple
from functools import partial

class TrajectoryGenerator:
    def __init__(self, config: dict):
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
        
        # Warmup JIT functions
        _ = self._get_traj1_jax(0.0)
        _ = self._get_traj2_jax(0.0)

    @partial(jax.jit, static_argnums=(0,))
    def _get_traj1_jax(self, t: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
        def pos_fn(t_val: float) -> jax.Array:
            w = (2.0 * jnp.pi) / self.traj1_period
            
            # Exact integral of tau_dot(t) = warp_c * (1 - alpha * sin^2(w t))
            tau = self.traj1_warp_c * (t_val - self.traj1_alpha * (t_val / 2.0 - jnp.sin(2.0 * w * t_val) / (4.0 * w)))
            
            wx, wy, wz = 2.0 * w, 1.0 * w, 4.0 * w
            return jnp.array([
                self.traj1_x_amp * jnp.sin(wx * tau),
                self.traj1_y_amp * jnp.sin(wy * tau),
                self.traj1_z_amp * jnp.sin(wz * tau) + self.traj1_target_z
            ])
            
        pos = pos_fn(t)
        vel = jax.jacfwd(pos_fn)(t)
        acc = jax.jacfwd(jax.jacfwd(pos_fn))(t)
        return pos, vel, acc

    @partial(jax.jit, static_argnums=(0,))
    def _get_traj2_jax(self, t: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # User's simplified integration for theta:
        theta = (self.traj2_v / self.traj2_A) * t
        f_theta = 1.0 + 3.0 * (jnp.sin(2.0 * theta)**2)
        theta_dot = self.traj2_v / (self.traj2_A * jnp.sqrt(f_theta))
        
        sin_4theta = jnp.sin(4.0 * theta)
        theta_ddot = - (3.0 * (self.traj2_v**2) * sin_4theta) / ((self.traj2_A**2) * (f_theta**2))
        
        def pos_fn(th: float) -> jax.Array:
            r = self.traj2_A * jnp.cos(2.0 * th)
            return jnp.array([
                r * jnp.cos(th),
                r * jnp.sin(th),
                self.traj2_target_z
            ])
            
        pos = pos_fn(theta)
        dp_dth = jax.jacfwd(pos_fn)(theta)
        d2p_dth2 = jax.jacfwd(jax.jacfwd(pos_fn))(theta)
        
        # Apply chain rule inside JAX
        vel = dp_dth * theta_dot
        acc = d2p_dth2 * (theta_dot**2) + dp_dth * theta_ddot
        return pos, vel, acc

    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.desired_traj == 1:
            pos, vel, acc = self._get_traj1_jax(t)
        else:
            pos, vel, acc = self._get_traj2_jax(t)
            
        return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64), np.array(acc, dtype=np.float64)
