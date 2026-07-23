import math
import numpy as np
import yaml
from typing import Any, Tuple
from scipy.spatial.transform import Rotation
import time
import os
import csv
from datetime import datetime

import jax
import jax.numpy as jnp

from src.proj import discrete_projection
from jax_resnet import resnet_network
from quadsim import QuadSim
from jax_resnet import init_resnet_weights
from src.desired_trajectory import TrajectoryGenerator

jax.config.update("jax_platform_name", "cpu") # type: ignore
jax.config.update("jax_enable_x64", True) # type: ignore

class SimRun:
    def __init__(self, param_dict: dict[str, Any], yaml_config_path: str, runner, px4) -> None:
        self.runner = runner
        self.px4 = px4
        with open(yaml_config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        self.config = full_config['aviary_rise_node']['ros__parameters']
        
        for k, v in param_dict.items():
            self.config[k] = v
        
        self.control_frequency_hz = self.config['control_frequency_hz']
        self.control_period_s = 1.0 / self.control_frequency_hz
        self.run_length_s = self.config['run_length_s']
        self.controller_type = self.config['controller_type']
        self.desired_traj = self.config['desired_trajectory']
        
        self.acc_hor_max_mps2 = self.config['mpc_acc_hor_max_mps2']
        self.acc_vert_max_mps2 = self.config['mpc_acc_vert_max_mps2']
        self.safe_x_min_m_ned = self.config['safe_x_min_m_ned']
        self.safe_x_max_m_ned = self.config['safe_x_max_m_ned']
        self.safe_y_min_m_ned = self.config['safe_y_min_m_ned']
        self.safe_y_max_m_ned = self.config['safe_y_max_m_ned']
        self.safe_z_max_m_ned = self.config['safe_z_max_m_ned']
        self.safe_z_min_m_ned = self.config['safe_z_min_m_ned']
        self.w_fail = self.config['w_fail']
        self.takeoff_timeout_s = self.config['takeoff_timeout_s']
        
        self.k_1 = self.config['k_1']
        self.k_2 = self.config['k_2']
        self.k_3 = self.config['k_3']
        if self.controller_type == "pid":
            self.K_P = self.config["K_P"]
            self.K_I = self.config["K_I"]
            self.K_D = self.config["K_D"]
        else:
            self.K_P = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
            self.K_I = (self.k_1 * self.k_2 * self.k_3) + self.k_1
            self.K_D = self.k_1 + self.k_2 + self.k_3
        self.K_RISE = self.config['k_rise']
        self.q_e = self.config['q_e']
        self.r_u = self.config['r_u']
        self.K_P_yaw = self.config['K_P_yaw']

        self.init_x_ned = self.config[f'traj{self.desired_traj}_init_x_m_ned']
        self.init_y_ned = self.config[f'traj{self.desired_traj}_init_y_m_ned']
        self.init_z_ned = self.config['init_z_m_ned']
        self.init_ned = np.array([self.init_x_ned, self.init_y_ned, self.init_z_ned], dtype=np.float64)
        self.hover_start_z_m_ned = self.config['hover_start_z_m_ned']
        self.init_tol_m = self.config['init_tol_m']
        self.yaw_des_deg = self.config['yaw_des_deg']
        self.init_yaw_deg = self.config['init_yaw_deg']
        
        self.cost_J = 0.0
        self.last_cost_integrand = 0.0
        self.integral_control_term = np.zeros(3, dtype=np.float64)
        self.last_control_integrand = np.zeros(3, dtype=np.float64)
        self.is_saturated = False
        self.sim_speed = self.config['sim_speed']
        self.initial_tracking_error_norm = None

        
        self.time_history: list[float] = []
        self.error_norm_history: list[float] = []
        self.control_output_norm_history: list[float] = []
        self.weight_history: list[list[float]] = []
        self.q_history: list[list[float]] = []
        self.qd_history: list[list[float]] = []
        self.u_history: list[list[float]] = []
        self.e_history: list[list[float]] = []
        
        self.traj_gen = TrajectoryGenerator(self.config)
        
        if self.controller_type == "resnet":
            self.config['d_in'] = 12
        elif self.controller_type == "integrated_resnet":
            self.config['d_in'] = 15
            
        if self.controller_type in ["resnet", "integrated_resnet"]:
            self.setup_neural_network()

    def setup_neural_network(self) -> None:
        self.theta_bar = self.config['theta_bar']
        self.sigma_mod = self.config['sigma_mod']

        key = jax.random.PRNGKey(self.config['base_seed'])
        init_scale = self.config['initial_weight_scale_factor']
        self.theta_hat = init_scale * init_resnet_weights(
            key=key,
            d_in=self.config['d_in'],
            hidden_width=self.config['hidden_width'],
            d_out=self.config['d_out'],
            b=self.config['num_blocks'],
            k_0=self.config['k_0'],
            k_i=self.config['k_i'],
            h_method=self.config['h_method'],
            o_method=self.config['o_method']
        )

        self.gamma = jnp.ones(len(self.theta_hat)) * self.config['gamma'] 
        
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
            theta_hat: jax.Array,
            x_vec: jax.Array,
            r1_vec: jax.Array,
            dt: float,
            theta_bar: float,
            gamma_diag: jax.Array,
            s_mod: float,
            saturated: bool
        ) -> Tuple[jax.Array, Any]:
            phi_val, vjp_fn = jax.vjp(lambda theta: self.bound_resnet(theta, x_vec), theta_hat)
            grad_term = vjp_fn(r1_vec)[0]
            theta_dot_unprojected = gamma_diag * (grad_term - s_mod * theta_hat)
            theta_next = discrete_projection(theta_hat, theta_dot_unprojected, dt, theta_bar, gamma_diag)
            final_theta = jax.lax.select(saturated, theta_hat, theta_next)
            return final_theta, phi_val

        self.compiled_update_step = compiled_update_step
        
        dummy_x = jnp.zeros(self.config['d_in'])
        dummy_r1 = jnp.zeros(self.config['d_out'])
        self.theta_hat, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period_s, self.theta_bar, self.gamma, self.sigma_mod, False)
        self.theta_hat.block_until_ready()


    def check_boundary_escape(self, q_ned: np.ndarray) -> bool:
        if not (self.safe_x_min_m_ned <= q_ned[0] <= self.safe_x_max_m_ned) or \
        not (self.safe_y_min_m_ned <= q_ned[1] <= self.safe_y_max_m_ned) or \
        not (self.safe_z_min_m_ned <= q_ned[2] <= self.safe_z_max_m_ned):
            return True
        return False

    def run(self) -> tuple[float, float, float]:
        """Executes the simulation loop and returns the cost."""
        if True:
            step = 0
            traj_t = 0.0
            
            sim_start_time_perf = time.perf_counter()
            
            while traj_t < self.run_length_s:
                # Read sensors from PX4
                q_ned = np.asarray(self.px4.state.pos_ned, dtype=np.float64)
                q_dot_ned = np.asarray(self.px4.state.vel_ned, dtype=np.float64)

                if self.check_boundary_escape(q_ned=q_ned):
                    print(f"[!] Got too close to a wall! Position: {q_ned}. Exiting.")
                    self.cost_J += self.w_fail * ((self.run_length_s - traj_t) ** 2)
                    break
        
                current_t_for_cost = traj_t
                qd_ned, qd_dot_ned, qd_ddot_ned = self.traj_gen.get_desired_state(traj_t)
                traj_t = self.control_period_s * step

                # --- TRACKING ERROR COMPUTATION (in NED frame) ---
                e_ned = qd_ned - q_ned
                
                if self.initial_tracking_error_norm is None:
                    self.initial_tracking_error_norm = float(np.linalg.norm(e_ned))
                else:
                    if float(np.linalg.norm(e_ned)) > self.initial_tracking_error_norm + 1.0:
                        print(f"[!] Tracking error diverged! Current: {float(np.linalg.norm(e_ned)):.2f}, Initial: {self.initial_tracking_error_norm:.2f}. Exiting.")
                        self.cost_J += self.w_fail * ((self.run_length_s - traj_t) ** 2)
                        break
                        
                e_dot_ned = qd_dot_ned - q_dot_ned
                r1_ned = e_dot_ned + (self.k_1 * e_ned)
                
                u_provisional = np.zeros(3, dtype=np.float64)
                current_control_integrand = np.zeros(3, dtype=np.float64)

                # --- CONTROL LAW EVALUATION ---
                match self.controller_type:
                    case "baseline" | "baseline_no_wind" | "pid":
                        current_control_integrand = self.K_I * e_ned + (self.controller_type in ["baseline", "baseline_no_wind"]) * (self.K_RISE * np.sign(r1_ned))
                        u_provisional = (self.K_P * e_ned) + (self.K_D * e_dot_ned) + self.integral_control_term
                        
                    case "resnet":
                        x_vec = jnp.array(np.concatenate((q_ned, q_dot_ned, qd_ned, qd_dot_ned)))
                        self.theta_hat, phi_out = self.compiled_update_step(
                            self.theta_hat, x_vec, jnp.array(r1_ned), self.control_period_s, self.theta_bar, self.gamma, self.sigma_mod, self.is_saturated
                        )
                        phi_val = np.array(phi_out)
                        u_nn = phi_val
                        current_control_integrand = self.K_I * e_ned + (self.K_RISE * np.sign(r1_ned))
                        u_provisional = u_nn + (self.K_P * e_ned) + (self.K_D * e_dot_ned) + self.integral_control_term
                        
                    case "integrated_resnet":
                        u_last = (self.K_P * e_ned) + (self.K_D * e_dot_ned) + self.integral_control_term
                        kappa_vec = jnp.array(np.concatenate((q_ned, q_dot_ned, qd_ned, qd_dot_ned, u_last)))
                        self.theta_hat, phi_out = self.compiled_update_step(
                            self.theta_hat, kappa_vec, jnp.array(r1_ned), self.control_period_s, self.theta_bar, self.gamma, self.sigma_mod, self.is_saturated
                        )
                        phi_val = np.array(phi_out)
                        u_nn = phi_val
                        current_control_integrand = self.K_I * e_ned + (self.K_RISE * np.sign(r1_ned)) + u_nn
                        u_provisional = (self.K_P * e_ned) + (self.K_D * e_dot_ned) + self.integral_control_term

                    case "supertwisting":
                        norm_r1 = np.linalg.norm(r1_ned)
                        sgn_r1 = np.sign(r1_ned)
                        current_control_integrand = sgn_r1
                        proposed_integral = self.integral_control_term + (sgn_r1 * self.control_period_s)
                        u_provisional = qd_ddot_ned + self.k_2 * np.sqrt(norm_r1) * sgn_r1 + self.k_3 * proposed_integral + self.k_1 * e_dot_ned

                    case _:
                        raise ValueError(f"Unknown controller type {self.controller_type}")
                        
                # --- SATURATION & ANTI-WINDUP ---
                self.is_saturated = False
                
                # 1. Create a mask to track which specific axes should freeze integration
                freeze_integrator = np.array([False, False, False])
                
                u_xy = u_provisional[0:2]
                norm_uxy = float(np.linalg.norm(u_xy))
                if norm_uxy > self.acc_hor_max_mps2:
                    self.is_saturated = True
                    if np.dot(e_ned[0:2], u_xy) > 0.0:
                        freeze_integrator[0:2] = True
                        
                if abs(u_provisional[2]) > self.acc_vert_max_mps2:
                    self.is_saturated = True
                    if np.sign(e_ned[2]) == np.sign(u_provisional[2]):
                        freeze_integrator[2] = True

                # 2. Integrate properly, strictly freezing the update on saturated axes
                if self.controller_type != "supertwisting":
                    # Trapezoidal update for linear/NN controllers
                    integral_update = (self.control_period_s / 2.0) * (current_control_integrand + self.last_control_integrand)
                else:
                    # Euler update for the supertwisting sliding mode controller
                    integral_update = current_control_integrand * self.control_period_s
                
                # Apply the boolean clamping mask to freeze saturated axes
                integral_update[freeze_integrator] = 0.0
                
                # Commit the actual clamped integral update
                self.integral_control_term += integral_update
                self.last_control_integrand = current_control_integrand
                
                # Recalculate unclamped control using the strictly clamped integral term
                if self.controller_type != "supertwisting":
                    u_unclamped_ned = (self.K_P * e_ned) + (self.K_D * e_dot_ned) + self.integral_control_term
                else:
                    # Recalculate ST control using the globally available r1_ned
                    u_unclamped_ned = (self.k_2 * np.sqrt(np.linalg.norm(r1_ned)) * np.sign(r1_ned)) + (self.k_3 * self.integral_control_term) + (self.k_1 * e_dot_ned)

                # Final control action clipping
                u_clamped_ned = np.copy(u_unclamped_ned)
                u_xy_final = u_clamped_ned[0:2]
                norm_uxy_final = float(np.linalg.norm(u_xy_final))
                if norm_uxy_final > self.acc_hor_max_mps2:
                    u_clamped_ned[0:2] = u_xy_final * (self.acc_hor_max_mps2 / norm_uxy_final)
                
                if abs(u_clamped_ned[2]) > self.acc_vert_max_mps2:
                    u_clamped_ned[2] = self.acc_vert_max_mps2 * np.sign(u_clamped_ned[2])


                # --- CONTROL ACTION TRANSFORMS ---
                # We skip transform to ENU/FLU because PX4 runner takes standard NED directly.

                # New cost function integrand: J = int( q_e * t * ||e(t)||^2 + r_u * ||u(t)||^2 ) dt
                current_cost_integrand = (self.q_e * current_t_for_cost * (np.linalg.norm(e_ned)**2)) + \
                                         (self.r_u * (np.linalg.norm(u_clamped_ned)**2))
                self.cost_J += (self.control_period_s / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                self.last_cost_integrand = current_cost_integrand

                # --- DATA COLLECTION ---
                sim_time_current = step * self.control_period_s
                self.time_history.append(sim_time_current)
                self.error_norm_history.append(float(np.linalg.norm(e_ned)))
                self.control_output_norm_history.append(float(np.linalg.norm(u_clamped_ned)))
                self.q_history.append(q_ned.tolist())
                self.qd_history.append(qd_ned.tolist())
                self.u_history.append(u_clamped_ned.tolist())
                self.e_history.append(e_ned.tolist())
                
                if hasattr(self, 'theta_hat'):
                    self.weight_history.append(np.array(self.theta_hat).tolist())
                else:
                    self.weight_history.append([])

                if not np.all(np.isfinite(u_clamped_ned)):
                    print("[!] Non-finite acceleration output!")
                    self.cost_J += self.w_fail * ((self.run_length_s - traj_t) ** 2)
                    break

                self.runner.step_with_acceleration_ned(
                    u_clamped_ned[0],
                    u_clamped_ned[1],
                    u_clamped_ned[2],
                    yaw_ned=math.radians(self.yaw_des_deg)
                )

                step += 1
                sim_time = step * self.control_period_s
                
                # LockstepRunner manages simulation speed, so we don't need sleep here.
            
            sim_stop_time_perf = time.perf_counter()
            sim_run_time_realworld = sim_stop_time_perf - sim_start_time_perf
            print(f"[*] Simulation took {round(sim_run_time_realworld, 1)} real seconds for {self.run_length_s} simulated seconds. Scale: {round(self.run_length_s / sim_run_time_realworld , 1)}. Specified speed was {self.sim_speed}.")

            # --- COMPUTE RMS ---
            e_rms = 0.0
            u_rms = 0.0
            if len(self.e_history) > 0:
                e_arr = np.array(self.e_history)
                u_arr = np.array(self.u_history)
                
                e_rms = float(np.sqrt(np.mean(np.sum(e_arr**2, axis=1))))
                u_rms = float(np.sqrt(np.mean(np.sum(u_arr**2, axis=1))))
                
                print(f"Tracking error RMS (e_RMS): {e_rms:.4f} m")
                print(f"Control effort RMS (u_RMS): {u_rms:.4f} m/s^2")

            # --- SAVE DATA ---
            if self.config['save_data']:
                traj_num = self.config['desired_trajectory']
                controller = self.controller_type
                
                output_dir = os.path.join("output", f"traj{traj_num}", controller)
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                csv_path = os.path.join(output_dir, f"{timestamp}.csv")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    header = ["Time_s", "Error_Norm_m", "Control_Output_Norm_mps2", "x", "y", "z", "xd", "yd", "zd"]
                    if len(self.weight_history) > 0 and len(self.weight_history[0]) > 0:
                        num_weights = len(self.weight_history[0])
                        header.extend([f"W{i}" for i in range(num_weights)])
                    
                    writer.writerow(header)
                    
                    for i in range(len(self.time_history)):
                        row = [
                            self.time_history[i],
                            self.error_norm_history[i],
                            self.control_output_norm_history[i],
                            self.q_history[i][0], self.q_history[i][1], self.q_history[i][2],
                            self.qd_history[i][0], self.qd_history[i][1], self.qd_history[i][2]
                        ]
                        if len(self.weight_history[i]) > 0:
                            row.extend(self.weight_history[i])
                        writer.writerow(row)
                        
                print(f"Saved data to {csv_path}")

            return self.cost_J, e_rms, u_rms