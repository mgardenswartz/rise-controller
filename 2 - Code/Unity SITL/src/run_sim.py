import math
import time
import numpy as np
import yaml
from typing import Any, Tuple
from scipy.spatial.transform import Rotation

import jax
import jax.numpy as jnp

from src.proj import discrete_projection
from src.desired_trajectory import traj1_spatial_derivs, traj2_spatial_derivs
from jax_resnet import resnet_network
from quadsim import QuadSim

jax.config.update("jax_platform_name", "cpu") # type: ignore
jax.config.update("jax_enable_x64", True) # type: ignore

class SimRun:
    def __init__(self, param_dict: dict[str, Any], yaml_config_path: str) -> None:
        # 1. Load base configuration
        with open(yaml_config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        self.config = full_config['aviary_rise_node']['ros__parameters']
        
        # 2. Override base config with Optuna suggestions
        for k, v in param_dict.items():
            self.config[k] = v
            
        # 3. Simulation & Environment Constants
        self.control_hz = self.config['control_frequency']
        self.dt = 1.0 / self.control_hz
        self.t_f = self.config['sim_time']
        self.controller_type = self.config['controller_type']
        self.desired_traj = self.config['desired_trajectory']
        
        # Safety & Limits
        self.acc_hor_max = self.config['mpc_acc_hor_max']
        self.acc_vert_max = self.config['mpc_acc_vert_max']
        self.safe_x_max = self.config['safe_x_max']
        self.safe_y_max = self.config['safe_y_max']
        self.safe_z_max = self.config['safe_z_max']
        self.safe_z_min = self.config['safe_z_min']
        self.w_fail = self.config['w_fail']
        
        # Trajectory Parameters
        self.target_z = self.config['traj1_center_z'] if self.desired_traj == 1 else self.config['traj2_center_z']
        self.traj1_warp_c = 1.0 / math.sqrt(1.0 - self.config['traj1_alpha_warp']) if self.config['traj1_alpha_warp'] < 1.0 else 1.0
        
        # Gains
        self.k_1 = self.config['k_1']
        self.k_2 = self.config['k_2']
        self.k_3 = self.config['k_3']
        self.K_P = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
        self.K_I = (self.k_1 * self.k_2 * self.k_3) + self.k_1
        self.K_D = self.k_1 + self.k_2 + self.k_3
        self.K_RISE = self.config['k_rise']
        self.q_e = self.config['q_e']
        self.r_u = self.config['r_u']

        # Initial Conditions
        self.init_x_ned_global = self.config['init_x']
        self.init_y_ned_global = self.config['init_y']
        self.init_z_ned_global = self.config['init_z']
        self.init_ned_global = np.array([self.init_x_ned_global, self.init_y_ned_global, self.init_z_ned_global], dtype=np.float64)
        self.hover_start_z_ned_global = self.config['hover_start_z']
        
        # State & Cost Memory
        self.cost_J = 0.0
        self.last_cost_integrand = 0.0
        self.integral_term = np.zeros(3, dtype=np.float64)
        self.last_integrand = np.zeros(3, dtype=np.float64)
        
        # NN Setup
        if self.controller_type in ["baseline", "developed"]:
            self.setup_neural_network()

    def setup_neural_network(self) -> None:
        self.theta_bar = self.config['theta_bar']
        self.sigma_mod = self.config['sigma_mod']
        self.gamma_diag = jnp.ones(len(self.config['initial_weights'])) * self.config['gamma']
        init_scale = self.config['initial_weight_scale_factor']
        self.theta_hat = jnp.array(self.config['initial_weights']) * init_scale
        
        self.bound_resnet = jax.jit(jax.tree_util.Partial( # type: ignore
            resnet_network,
            d_in=self.config['d_in'],
            hidden_width=self.config['hidden_width'],
            d_out=self.config['d_out'],
            b=self.config['num_blocks'],
            k_0=self.config['k_0'],
            k_i=self.config['k_i'],
            h_act_func=self.config['h_act_func'],
            o_act_func=self.config['o_act_func'],
            shortcut_act_func=self.config['shortcut_act_func'],
        ))
        
        @jax.jit
        def compiled_update_step(
            theta_hat: jax.Array, x_vec: jax.Array, r1_vec: jax.Array,
            dt: float, theta_bar: float, gamma_diag: jax.Array,
            s_mod: float, saturated: bool
        ) -> Tuple[jax.Array, Any]:
            phi_val, vjp_fn = jax.vjp(lambda t: self.bound_resnet(t, x_vec), theta_hat)
            grad_term = vjp_fn(r1_vec)[0]
            theta_dot_unprojected = gamma_diag * (grad_term - s_mod * theta_hat)
            theta_next = discrete_projection(theta_hat, theta_dot_unprojected, dt, theta_bar, gamma_diag)
            final_theta = jax.lax.select(saturated, theta_hat, theta_next)
            return final_theta, phi_val

        self.compiled_update_step = compiled_update_step
        
        # JIT Warmup
        dummy_x = jnp.zeros(self.config['d_in'])
        dummy_r1 = jnp.zeros(self.config['d_out'])
        self.theta_hat, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.dt, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()

    @staticmethod
    def unity_to_ned_global(vec: np.ndarray | list[float]) -> np.ndarray:
        """
        Maps Unity to NED.
        Unity: (X, Y, Z)
        NED: (X, -Z, -Y)
        """
        return np.array([vec[0], -vec[2], -vec[1]], dtype=np.float64)

    @staticmethod
    def ned_to_unity(vec: np.ndarray | list[float]) -> np.ndarray:
        """
        Maps NED to Unity.
        NED: (X, Y, Z)
        Unity: (X, -Z, -Y)
        """
        return np.array([vec[0], -vec[2], -vec[1]], dtype=np.float64)

    @staticmethod
    def gps_frame_to_unity(vec: np.ndarray | list[float]) -> np.ndarray:
        """
        NED: (X, Y, Z)
        Unity: (Y, -Z, X)
        """
        return np.array([vec[1], -vec[2], vec[0]], dtype=np.float64)

    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.desired_traj == 1:
            w = (2.0 * math.pi) / self.config['traj1_period']
            tau_dot = self.traj1_warp_c * (1.0 - self.config['traj1_alpha_warp'] * (math.sin(w * t)**2))
            tau_ddot = -self.traj1_warp_c * self.config['traj1_alpha_warp'] * w * math.sin(2.0 * w * t) * tau_dot
            
            pos_jnp, dp_dtau, d2p_dtau2 = traj1_spatial_derivs(
                t, self.target_z, self.config['traj1_period'], 
                self.config['traj1_x_amp'], self.config['traj1_y_amp'], self.config['traj1_z_amp']
            )
            
            qd = np.array(pos_jnp, dtype=np.float64)
            qd_dot = np.array(dp_dtau, dtype=np.float64) * tau_dot
            qd_ddot = (np.array(d2p_dtau2, dtype=np.float64) * (tau_dot**2)) + (np.array(dp_dtau, dtype=np.float64) * tau_ddot)
            
        else:
            theta = (self.config['traj2_target_speed'] / self.config['traj2_petal_radius']) * t # Simplified integration for example
            f_theta = 1.0 + 3.0 * (math.sin(2.0 * theta)**2)
            theta_dot = self.config['traj2_target_speed'] / (self.config['traj2_petal_radius'] * math.sqrt(f_theta))
            
            sin_4theta = math.sin(4.0 * theta)
            theta_ddot = - (3.0 * (self.config['traj2_target_speed']**2) * sin_4theta) / ((self.config['traj2_petal_radius']**2) * (f_theta**2))
            
            pos_jnp, dp_dth, d2p_dth2 = traj2_spatial_derivs(theta, self.target_z, self.config['traj2_petal_radius'])
            
            qd = np.array(pos_jnp, dtype=np.float64)
            qd_dot = np.array(dp_dth, dtype=np.float64) * theta_dot
            qd_ddot = (np.array(d2p_dth2, dtype=np.float64) * (theta_dot**2)) + (np.array(dp_dth, dtype=np.float64) * theta_ddot)

        return qd, qd_dot, qd_ddot

  

    def run(self) -> float:
        """Executes the simulation loop and returns the cost."""
        with QuadSim() as sim:
            drone = sim.drone()
            
            status = sim.get_status()
            steps_per_tick = max(1, round((1.0 / self.control_hz) / status.fixed_dt))
            
            # Spawn in Unity coordinates using explicit config keys
            init_unity = self.ned_to_unity(self.init_ned_global)
            print(f"Init unity is s{init_unity}.")
            
            sqrt2_ = math.sqrt(2)/2
            # drone.reset_pose(x=init_unity[0], y=init_unity[2], z=init_unity[1], qx=0, qy=-sqrt2_2, qz=0, qw=sqrt2_2) # This function has a bug and hence I swapped the 2 and 1.
            drone.reset_pose(x=init_unity[0], y=init_unity[2], z=init_unity[1], qx=0, qy=0, qz=0, qw=1)
            total_steps = round(self.control_hz * self.t_f)
            
            traj_t = 0.0
            
            for step in range(total_steps):
                # We use fixed dt for the discrete math
                sensors = drone.get_sensors()
                
                gps_position_unity_frame = np.array([sensors.gps_position[0], sensors.gps_position[2], sensors.gps_position[1]], dtype=np.float64) # This function has a bug and hence I swapped the 2 and 1.
                #print(f"GPS pos. meas. in unity frame is q = {gps_position_unity_frame}")
                q = self.unity_to_ned_global(gps_position_unity_frame) # NED Global
                #print(f"GPS pos. meas. in NED is q = {q}")
                gps_vel_unity_frame = np.array([sensors.gps_vel_ned[0], sensors.gps_vel_ned[2], sensors.gps_vel_ned[1]], dtype=np.float64)
                q_dot = self.unity_to_ned_global(gps_vel_unity_frame) # NED Global
                # print(f"Velocity meas. says {q_dot}")

                quat = np.array(sensors.imu_orientation)
                if np.linalg.norm(quat) < 1e-6: quat = np.array([0.0, 0.0, 0.0, 1.0])
                r = Rotation.from_quat(quat)
                yaw_enu = r.as_euler('zyx')[0]
                current_yaw = (np.pi / 2.0 - yaw_enu + np.pi) % (2 * np.pi) - np.pi
                
                # Boundary Failsafe Check
                if not (-self.safe_x_max <= q[0] <= self.safe_x_max) or \
                   not (-self.safe_y_max <= q[1] <= self.safe_y_max) or \
                   not (self.safe_z_min <= q[2] <= self.safe_z_max):
                    # Only penalize if we've actually started the trajectory
                    # if not is_takeoff:
                    #     self.cost_J += self.w_fail * ((self.t_f - traj_t) ** 2)
                    #break
                    pass
                
                # --- STATE MACHINE: Takeoff vs Trajectory ---
                qd = np.array([self.init_x_ned_global, self.init_y_ned_global, self.hover_start_z_ned_global], dtype=np.float64)
                qd_dot = np.zeros(3, dtype=np.float64)
                e = qd - q
                print(f"e: {e}")
                e_dot = qd_dot - q_dot

                u = 0.02 * e + 0.4 * e_dot
                # if is_takeoff:
                #     qd = np.array([self.init_x_ned_global, self.init_y_ned_global, self.hover_start_z_ned_global], dtype=np.float64)
                #     qd_dot = np.zeros(3, dtype=np.float64)
                #     e = qd - q
                #     e_dot = qd_dot - q_dot
                    
                    # if np.linalg.norm(e) <= self.config['init_tol']:
                    #     is_takeoff = False
                    #     # Reset memory integrals so they don't unwind from the takeoff effort
                    #     self.integral_term = np.zeros(3, dtype=np.float64)
                    #     self.last_integrand = np.zeros(3, dtype=np.float64)
                    #     self.st_integral = np.zeros(3, dtype=np.float64)
                # else:
                #     traj_t += self.dt

                    # # TEMPORARY
                    # qd = np.array([self.init_x_ned_global, self.init_y_ned_global, self.hover_start_z_ned_global], dtype=np.float64)
                    # qd_dot = np.zeros(3, dtype=np.float64)
                    # qd, qd_dot, _ = self.get_desired_state(traj_t)
                    # e = qd - q
                    # e_dot = qd_dot - q_dot
                
                r1 = e_dot + (self.k_1 * e)
                
                u = np.zeros(3, dtype=np.float64)
                phi_val = np.zeros(3, dtype=np.float64)
                
                # # --- CONTROL LAW EVALUATION ---
                if self.controller_type == "noresnet":
                    current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                    delta_int = (self.dt / 2.0) * (current_integrand + self.last_integrand)
                    self.last_integrand = current_integrand
                    u_unclamped = (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                    
                # elif self.controller_type == "baseline":
                #     x_vec = jnp.array(np.concatenate((q, q_dot, qd, qd_dot)))
                #     self.theta_hat, phi_out = self.compiled_update_step(
                #         self.theta_hat, x_vec, jnp.array(r1), self.dt, self.theta_bar, self.gamma_diag, self.sigma_mod, False
                #     )
                #     phi_val = np.array(phi_out, dtype=np.float64)
                #     current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                #     delta_int = (self.dt / 2.0) * (current_integrand + self.last_integrand)
                #     self.last_integrand = current_integrand
                #     u_unclamped = phi_val + (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                    
                # elif self.controller_type == "developed":
                #     u_last = (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                #     kappa_vec = jnp.array(np.concatenate((q, q_dot, qd, qd_dot, u_last)))
                #     self.theta_hat, phi_out = self.compiled_update_step(
                #         self.theta_hat, kappa_vec, jnp.array(r1), self.dt, self.theta_bar, self.gamma_diag, self.sigma_mod, False
                #     )
                #     phi_val = np.array(phi_out, dtype=np.float64)
                #     current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1)) + phi_val
                #     delta_int = (self.dt / 2.0) * (current_integrand + self.last_integrand)
                #     self.last_integrand = current_integrand
                #     u_unclamped = (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                    
                # elif self.controller_type == "supertwisting":
                #     norm_r1 = np.linalg.norm(r1)
                #     sgn_r1 = np.sign(r1)
                #     # NOTE: ST integral is not conditionally clamped in standard literature, 
                #     # but bounds can be added if it winds up.
                #     self.integral_term += sgn_r1 * self.dt
                #     u_unclamped = self.k_2 * np.sqrt(norm_r1) * sgn_r1 + self.k_3 * self.integral_term + self.k_1 * e_dot

                # # --- SATURATION & ANTI-WINDUP ---
                # u = np.copy(u_unclamped)
                # freeze_int_xy = False
                # freeze_int_z = False

                # u_xy = u[0:2]
                # norm_uxy = float(np.linalg.norm(u_xy))
                # if norm_uxy > self.acc_hor_max:
                #     u[0:2] = u_xy * (self.acc_hor_max / norm_uxy)
                #     if np.dot(e[0:2], u[0:2]) > 0.0:
                #         freeze_int_xy = True
                        
                # if abs(u[2]) > self.acc_vert_max:
                #     u[2] = self.acc_vert_max * np.sign(u[2])
                #     if np.sign(e[2]) == np.sign(u[2]):
                #         freeze_int_z = True

                # # Apply integrations based on freeze flags (for continuous controllers)
                # if self.controller_type != "supertwisting":
                #     if not freeze_int_xy:
                #         self.integral_term[0:2] += delta_int[0:2]
                #     if not freeze_int_z:
                #         self.integral_term[2] += delta_int[2]

                # # --- COST INTEGRATION (Only if Tracking) ---
                # if not is_takeoff:
                #     norm_e = float(np.linalg.norm(e))
                #     norm_u = float(np.linalg.norm(u))
                #     current_cost_integrand = (traj_t * self.q_e * (norm_e ** 2)) + (self.r_u * (norm_u ** 2))
                    
                #     if traj_t == self.dt: # First timestep of trajectory
                #         self.last_cost_integrand = current_cost_integrand
                        
                #     self.cost_J += (self.dt / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                #     self.last_cost_integrand = current_cost_integrand

                u = u_unclamped


                # Convert control action from Global NED to Body FLU
                cy = np.cos(current_yaw)
                sy = np.sin(current_yaw)
                u_flu = np.array([
                    cy * u[0] + sy * u[1],   # Forward
                    sy * u[0] - cy * u[1],   # Left
                    -u[2]                    # Up
                ], dtype=np.float64)
                print(u_flu)

                yaw_rate_cmd = 2.0 * (0.0 - current_yaw)

                # drone.step_with_acceleration(
                #     ax=u_flu[0],
                #     ay=u_flu[1],
                #     az=u_flu[2],
                #     yaw_rate=0.0, #yaw_rate_cmd,
                #     count=steps_per_tick
                # )

                drone.step_with_acceleration(
                    ax=1.0,
                    ay=0.0,
                    az=0.0,
                    yaw_rate=0.0, #yaw_rate_cmd,
                    count=steps_per_tick
                )


            sim.pause()
            return self.cost_J