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
        with open(yaml_config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        self.config = full_config['aviary_rise_node']['ros__parameters']
        
        for k, v in param_dict.items():
            self.config[k] = v
        
        self.control_frequency_hz = self.config['control_frequency_hz']
        self.control_period_s = 1.0 / self.control_frequency_hz
        self.sim_time_s = self.config['sim_time_s']
        self.controller_type = self.config['controller_type']
        self.desired_traj = self.config['desired_trajectory']
        
        self.acc_hor_max_mps2 = self.config['mpc_acc_hor_max_mps2']
        self.acc_vert_max_mps2 = self.config['mpc_acc_vert_max_mps2']
        self.safe_x_min_m_ned_aviary = self.config['safe_x_min_m_ned']
        self.safe_x_max_m_ned_aviary = self.config['safe_x_max_m_ned']
        self.safe_y_min_m_ned_aviary = self.config['safe_y_min_m_ned']
        self.safe_y_max_m_ned_aviary = self.config['safe_y_max_m_ned']
        self.safe_z_max_m_ned_aviary = self.config['safe_z_max_m_ned']
        self.safe_z_min_m_ned_aviary = self.config['safe_z_min_m_ned']
        self.w_fail = self.config['w_fail']
        
        self.target_z_m_ned = self.config['traj1_center_z_m_ned'] if self.desired_traj == 1 else self.config['traj2_center_z_m_ned']
        self.traj1_warp_c = 1.0 / math.sqrt(1.0 - self.config['traj1_alpha_warp']) if self.config['traj1_alpha_warp'] < 1.0 else 1.0
        
        self.k_1 = self.config['k_1']
        self.k_2 = self.config['k_2']
        self.k_3 = self.config['k_3']
        self.K_P = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
        self.K_I = (self.k_1 * self.k_2 * self.k_3) + self.k_1
        self.K_D = self.k_1 + self.k_2 + self.k_3
        self.K_RISE = self.config['k_rise']
        self.q_e = self.config['q_e']
        self.r_u = self.config['r_u']
        self.K_yaw = self.config['K_yaw']

        self.init_x_ned = self.config['init_x_m_ned']
        self.init_y_ned = self.config['init_y_m_ned']
        self.init_z_ned = self.config['init_z_m_ned']
        self.init_ned = np.array([self.init_x_ned, self.init_y_ned, self.init_z_ned], dtype=np.float64)
        self.hover_start_z_m_ned = self.config['hover_start_z_m_ned']
        self.init_tol_m = self.config['init_tol_m']
        self.yaw_des_deg = self.config['yaw_des_deg']
        self.init_yaw_deg = self.config['init_yaw_deg']
        
        self.cost_J = 0.0
        self.last_cost_integrand = 0.0
        self.integral_term = np.zeros(3, dtype=np.float64)
        self.last_integrand = np.zeros(3, dtype=np.float64)
        
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
        
        dummy_x = jnp.zeros(self.config['d_in'])
        dummy_r1 = jnp.zeros(self.config['d_out'])
        self.theta_hat, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period_s, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()

    # @staticmethod
    # def unity_to_ned(vec: np.ndarray | list[float]) -> np.ndarray:
    #     """
    #     Maps Unity to NED.
    #     Unity: (X, Y, Z)
    #     NED: (X, -Z, -Y)
    #     """
    #     return np.array([vec[0], -vec[2], -vec[1]], dtype=np.float64)

    # @staticmethod
    # def ned_to_unity(vec: np.ndarray | list[float]) -> np.ndarray:
    #     """
    #     Maps NED to Unity.
    #     NED: (X, Y, Z)
    #     Unity: (X, -Z, -Y)
    #     """
    #     return np.array([vec[0], -vec[2], -vec[1]], dtype=np.float64)

    @staticmethod
    def swap_ned_aviary_and_enu(vec: np.ndarray | list[float]) -> np.ndarray:
        """
        Maps NED to ENU.
        ENU: (X, Y, Z)
        NED: (Y, )
        """
        return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)

    # @staticmethod
    # def enu_to_ned(vec: np.ndarray | list[float]) -> np.ndarray:
    #     """
    #     Maps NED to ENU.
    #     ENU: (X, Y, Z)
    #     NED: (X, -Y, -Z)
    #     """
    #     return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)

    # @staticmethod
    # def enu_to_ned(vec: np.ndarray | list[float]) -> np.ndarray:
    #     """
    #     Maps NED to ENU.
    #     ENU: (X, Y, Z)
    #     NED: (X, -Y, -Z)
    #     """
    #     return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)


    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: Fix repeated use of [''] in favor of a self attribute. And, move these to src/desired_trajectory.py
        if self.desired_traj == 1:
            w = (2.0 * math.pi) / self.config['traj1_period_s']
            tau_dot = self.traj1_warp_c * (1.0 - self.config['traj1_alpha_warp'] * (math.sin(w * t)**2))
            tau_ddot = -self.traj1_warp_c * self.config['traj1_alpha_warp'] * w * math.sin(2.0 * w * t) * tau_dot
            
            pos_jnp, dp_dtau, d2p_dtau2 = traj1_spatial_derivs(
                t, self.target_z_m_ned, self.config['traj1_period_s'], 
                self.config['traj1_x_amp_m_ned'], self.config['traj1_y_amp_m_ned'], self.config['traj1_z_amp_m_ned']
            )
            
            qd = np.array(pos_jnp, dtype=np.float64)
            qd_dot = np.array(dp_dtau, dtype=np.float64) * tau_dot
            qd_ddot = (np.array(d2p_dtau2, dtype=np.float64) * (tau_dot**2)) + (np.array(dp_dtau, dtype=np.float64) * tau_ddot)
            
        else:
            theta = (self.config['traj2_target_speed_mps'] / self.config['traj2_petal_radius_m']) * t # Simplified integration for example
            f_theta = 1.0 + 3.0 * (math.sin(2.0 * theta)**2)
            theta_dot = self.config['traj2_target_speed_mps'] / (self.config['traj2_petal_radius_m'] * math.sqrt(f_theta))
            
            sin_4theta = math.sin(4.0 * theta)
            theta_ddot = - (3.0 * (self.config['traj2_target_speed_mps']**2) * sin_4theta) / ((self.config['traj2_petal_radius_m']**2) * (f_theta**2))
            
            pos_jnp, dp_dth, d2p_dth2 = traj2_spatial_derivs(theta, self.target_z_m_ned, self.config['traj2_petal_radius_m'])
            
            qd = np.array(pos_jnp, dtype=np.float64)
            qd_dot = np.array(dp_dth, dtype=np.float64) * theta_dot
            qd_ddot = (np.array(d2p_dth2, dtype=np.float64) * (theta_dot**2)) + (np.array(dp_dth, dtype=np.float64) * theta_ddot)

        return qd, qd_dot, qd_ddot

    def check_boundary_escape(self, q_ned: np.ndarray) -> bool:
        if not (self.safe_x_min_m_ned <= q_ned[0] <= self.safe_x_max_m_ned) or \
        not (self.safe_y_min_m_ned <= q_ned[1] <= self.safe_y_max_m_ned) or \
        not (self.safe_z_min_m_ned <= q_ned[2] <= self.safe_z_max_m_ned):
            return True
        return False

    def run(self) -> float:
        """Executes the simulation loop and returns the cost."""
        with QuadSim() as sim:
            status = sim.get_status()
            steps_per_tick = max(1, round((1.0 / self.control_frequency_hz) / status.fixed_dt))
            
            drone = sim.drone()
            init_position_enu = self.swap_ned_and_enu(self.init_ned)
            drone.reset_pose(
                x=init_position_enu[0],
                y=init_position_enu[1], 
                z=init_position_enu[2],
                qx=0,
                qy=0,
                qz=math.sin(math.radians(self.init_yaw_deg / 2)),
                qw=math.cos(math.radians(self.init_yaw_deg / 2))
            )
            
            takeoff_complete = False
            step = 0
            traj_t = 0.0
            total_steps = round(self.control_frequency_hz * self.sim_time_s)
            while step < total_steps: 
                print(f"step: {step}")
                sensors = drone.get_sensors()

                # Read sensors
                q_ned = self.swap_ned_and_enu(sensors.gps_position) # GPS is ENU
                q_dot_ned = sensors.velocity_ned# np.array([sensors.velocity_ned[1], -sensors.velocity_ned[0]])
                
                # print(f"q_enu is {sensors.gps_position} <--Correct!")
                # print(f"your sensors.position_enu say {sensors.position_enu} <--Also correct!")
                # print(f"q_ned {q_ned} <--Correct!")
                # print(f"your sensors.position_ned says {sensors.position_ned} <--Wrong!")
                # print(f"translating the above gives {self.swap_ned_and_enu(sensors.position_ned)}")
                yaw_enu = np.array(sensors.imu_attitude)[-1] # RPY

                if self.check_boundary_escape(q_ned=q_ned):
                    print("Hit a wall! Exiting.")
                    if takeoff_complete:
                        self.cost_J += self.w_fail * ((self.sim_time_s - traj_t) ** 2)
                    else:
                        self.cost_J = 1e6
                    break
        
                # --- STATE MACHINE: Takeoff vs Trajectory ---
                # qd = np.array([self.init_x_ned, self.init_y_ned, self.hover_start_z_ned], dtype=np.float64)
                qd_ned = np.array([self.init_x_ned, 0.0, self.init_z_ned], dtype=np.float64)
                qd_dot_ned = np.zeros(3, dtype=np.float64)
                e_ned = qd_ned - q_ned
                #print(f"e_ned {e_ned}")
                e_dot_ned = qd_dot_ned - q_dot_ned

                # if is_takeoff:
                #     qd = np.array([self.init_x_ned, self.init_y_ned, self.hover_start_z_ned], dtype=np.float64)
                #     qd_dot = np.zeros(3, dtype=np.float64)
                #     e = qd - q
                #     e_dot = qd_dot - q_dot_ned
                    
                    # if np.linalg.norm(e) <= self.config['init_tol']:
                    #     is_takeoff = False
                    #     # Reset memory integrals so they don't unwind from the takeoff effort
                    #     self.integral_term = np.zeros(3, dtype=np.float64)
                    #     self.last_integrand = np.zeros(3, dtype=np.float64)
                    #     self.st_integral = np.zeros(3, dtype=np.float64)
                # else:
                #     traj_t += self.dt

                    # # TEMPORARY
                    # qd = np.array([self.init_x_ned, self.init_y_ned, self.hover_start_z_ned], dtype=np.float64)
                    # qd_dot = np.zeros(3, dtype=np.float64)
                    # qd, qd_dot, _ = self.get_desired_state(traj_t)
                    # e = qd - q
                    # e_dot = qd_dot - q_dot_ned
                
                r1_ned = e_dot_ned + (self.k_1 * e_ned)
                
                u = np.zeros(3, dtype=np.float64)
                phi_val = np.zeros(3, dtype=np.float64)
                
                # # --- CONTROL LAW EVALUATION ---
                if self.controller_type == "noresnet":
                    current_integrand = self.K_I * e_ned + (self.K_RISE * np.sign(r1_ned))
                    #delta_int = (self.dt / 2.0) * (current_integrand + self.last_integrand)
                    self.last_integrand = current_integrand
                    u_unclamped = (self.K_P * e_ned) + (self.K_D * e_dot_ned) + self.integral_term
                else:
                    raise ValueError
                    
                # elif self.controller_type == "baseline":
                #     x_vec = jnp.array(np.concatenate((q, q_dot_ned, qd, qd_dot)))
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
                #     kappa_vec = jnp.array(np.concatenate((q, q_dot_ned, qd, qd_dot, u_last)))
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

                u_ned = u_unclamped
                #print(f"u_ned {u_ned}")

                u_enu = self.swap_ned_and_enu(u_ned)
                quat_rotation_enu = Rotation.from_quat(sensors.imu_orientation, scalar_first=False)
                u_flu = quat_rotation_enu.inv().apply(u_enu) 

                # P Controller for yaw (with wrap-around fix)
                yaw_target_deg = 0.0
                current_yaw_deg = yaw_enu
                # Shortest angular error in degrees: (target - current + 180) % 360 - 180
                e_yaw_deg = (yaw_target_deg - current_yaw_deg + 180.0) % 360.0 - 180.0
                
                yaw_rate_cmd = self.K_yaw * e_yaw_deg

                # drone.step_with_acceleration(
                #     ax=u_flu[0],
                #     ay=u_flu[1],
                #     az=u_flu[2],
                #     yaw_rate=yaw_rate_cmd,
                #     count=steps_per_tick
                # )

                drone.step_with_acceleration(
                    ax=8,
                    ay=0,
                    az=0,
                    yaw_rate=0,
                    count=steps_per_tick
                )

                step += 1

            sim.pause()
            return self.cost_J