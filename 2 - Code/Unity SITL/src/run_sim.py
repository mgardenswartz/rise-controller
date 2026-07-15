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
    def __init__(self, param_dict: dict[str, Any], yaml_config_path: str) -> None:
        with open(yaml_config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        self.config = full_config['aviary_rise_node']['ros__parameters']
        
        for k, v in param_dict.items():
            self.config[k] = v
        
        self.control_frequency_hz = self.config['control_frequency_hz']
        self.control_period_s = 1.0 / self.control_frequency_hz
        self.sim_length_s = self.config['sim_length_s']
        self.controller_type = self.config['controller_type']
        self.desired_traj = self.config['desired_trajectory']
        
        self.acc_hor_max_mps2 = self.config['mpc_acc_hor_max_mps2']
        self.acc_vert_max_mps2 = self.config['mpc_acc_vert_max_mps2']
        self.safe_x_min_m_ned_aviary_aviary = self.config['safe_x_min_m_ned_aviary']
        self.safe_x_max_m_ned_aviary_aviary = self.config['safe_x_max_m_ned_aviary']
        self.safe_y_min_m_ned_aviary_aviary = self.config['safe_y_min_m_ned_aviary']
        self.safe_y_max_m_ned_aviary_aviary = self.config['safe_y_max_m_ned_aviary']
        self.safe_z_max_m_ned_aviary_aviary = self.config['safe_z_max_m_ned_aviary']
        self.safe_z_min_m_ned_aviary_aviary = self.config['safe_z_min_m_ned_aviary']
        self.w_fail = self.config['w_fail']
        self.takeoff_timeout_s = self.config['takeoff_timeout_s']
        
        self.k_1 = self.config['k_1']
        self.k_2 = self.config['k_2']
        self.k_3 = self.config['k_3']
        if self.controller_type == "pid":
            self.K_P = self.config.get("K_P", 0.0)
            self.K_I = self.config.get("K_I", 0.0)
            self.K_D = self.config.get("K_D", 0.0)
        else:
            self.K_P = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
            self.K_I = (self.k_1 * self.k_2 * self.k_3) + self.k_1
            self.K_D = self.k_1 + self.k_2 + self.k_3
        self.K_RISE = self.config['k_rise']
        self.q_e = self.config['q_e']
        self.r_u = self.config['r_u']
        self.K_P_yaw = self.config['K_P_yaw']

        self.init_x_ned = self.config['init_x_m_ned_aviary']
        self.init_y_ned = self.config['init_y_m_ned_aviary']
        self.init_z_ned = self.config['init_z_m_ned_aviary']
        self.init_ned = np.array([self.init_x_ned, self.init_y_ned, self.init_z_ned], dtype=np.float64)
        self.hover_start_z_m_ned_aviary = self.config['hover_start_z_m_ned_aviary']
        self.init_tol_m = self.config['init_tol_m']
        self.yaw_des_deg = self.config['yaw_des_deg']
        self.init_yaw_deg = self.config['init_yaw_deg']
        
        self.cost_J = 0.0
        self.last_cost_integrand = 0.0
        self.integral_control_term = np.zeros(3, dtype=np.float64)
        self.last_control_integrand = np.zeros(3, dtype=np.float64)
        self.is_saturated = False
        self.sim_speed = self.config['sim_speed']
        
        self.time_history: list[float] = []
        self.error_norm_history: list[float] = []
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

        self.gamma_diag = jnp.ones(len(self.theta_hat)) * self.config['gamma'] 
        
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
        self.theta_hat, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period_s, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()

    @staticmethod
    def swap_ned_aviary_and_enu(vec: np.ndarray | list[float]) -> np.ndarray:
        """
        Maps NED to ENU.
        ENU: (X, Y, Z)
        NED_Aviary: (X, -Y, -Z)
        """
        return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)

    def check_boundary_escape(self, q_ned_aviary: np.ndarray) -> bool:
        if not (self.safe_x_min_m_ned_aviary_aviary <= q_ned_aviary[0] <= self.safe_x_max_m_ned_aviary_aviary) or \
        not (self.safe_y_min_m_ned_aviary_aviary <= q_ned_aviary[1] <= self.safe_y_max_m_ned_aviary_aviary) or \
        not (self.safe_z_min_m_ned_aviary_aviary <= q_ned_aviary[2] <= self.safe_z_max_m_ned_aviary_aviary):
            return True
        return False

    def run(self) -> tuple[float, float, float]:
        """Executes the simulation loop and returns the cost."""
        with QuadSim() as sim:
            sim.pause()
            status = sim.get_status()
            steps_per_tick = max(1, round((self.control_period_s) / status.fixed_dt))
            
            drone = sim.drone()
            init_position_enu = self.swap_ned_aviary_and_enu(self.init_ned)
            drone.reset_pose(
                x=init_position_enu[0],
                y=init_position_enu[1], 
                z=init_position_enu[2],
                qx=0,
                qy=0,
                qz=math.sin(math.radians(self.init_yaw_deg / 2)),
                qw=math.cos(math.radians(self.init_yaw_deg / 2))
            )
            flight_mode = 'TAKEOFF'
            step = 0
            traj_t = 0.0
            takeoff_steps = 0
            max_takeoff_steps = round(self.control_frequency_hz * self.takeoff_timeout_s)
            
            wall_start = time.perf_counter()
            
            while traj_t < self.sim_length_s:
                sensors = drone.get_sensors()

                # Read sensors
                q_ned = self.swap_ned_aviary_and_enu(sensors.gps_position) # GPS is ENU
                q_dot_ned = self.swap_ned_aviary_and_enu(sensors.velocity_enu) # Velocity is ENU
                yaw_enu = np.array(sensors.imu_attitude)[-1] # RPY

                if self.check_boundary_escape(q_ned_aviary=q_ned):
                    print(f"[!] Got too close to a wall! Position: {q_ned}. Exiting.")
                    if flight_mode == 'TRAJECTORY':
                        self.cost_J += self.w_fail * ((self.sim_length_s - traj_t) ** 2)
                    else:
                        self.cost_J = 1e6
                    break
        
                # State machine
                current_t_for_cost = traj_t
                match flight_mode:
                    case 'TAKEOFF':
                        takeoff_steps += 1
                        qd_ned_aviary = np.array([self.init_x_ned, self.init_y_ned, self.hover_start_z_m_ned_aviary], dtype=np.float64)
                        qd_dot_ned_aviary = np.zeros(3, dtype=np.float64)
                        
                        dist = np.linalg.norm(q_ned - qd_ned_aviary)
                        if dist <= self.init_tol_m:
                            flight_mode = 'TRAJECTORY'
                            sim_start_time_perf = time.perf_counter()
                            print("Starting desired trajectory now.")
                        elif takeoff_steps > max_takeoff_steps:
                            print(f"[!] Takeoff timeout exceeded ({self.takeoff_timeout_s}s)! Exiting.")
                            self.cost_J = 1e6
                            break
                    
                    case 'TRAJECTORY':
                        qd_ned_aviary, qd_dot_ned_aviary, _ = self.traj_gen.get_desired_state(traj_t)
                        traj_t = self.control_period_s * step
                    case _:
                        raise ValueError(f'Invalid flight_mode selected: {flight_mode}.')

                # --- TRACKING ERROR COMPUTATION (in NED_aviary frame) ---
                e_ned_aviary = qd_ned_aviary - q_ned
                e_dot_ned_aviary = qd_dot_ned_aviary - q_dot_ned
                r1_ned_aviary = e_dot_ned_aviary + (self.k_1 * e_ned_aviary)
                
                u_provisional = np.zeros(3, dtype=np.float64)
                current_control_integrand = np.zeros(3, dtype=np.float64)

                # --- CONTROL LAW EVALUATION ---
                match self.controller_type:
                    case "baseline" | "baseline_no_wind" | "pid":
                        current_control_integrand = self.K_I * e_ned_aviary + (self.controller_type in ["baseline", "baseline_no_wind"]) * (self.K_RISE * np.sign(r1_ned_aviary))
                        u_provisional = (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term
                        
                    case "resnet":
                        x_vec = jnp.array(np.concatenate((q_ned, q_dot_ned, qd_ned_aviary, qd_dot_ned_aviary)))
                        self.theta_hat, phi_out = self.compiled_update_step(
                            self.theta_hat, x_vec, jnp.array(r1_ned_aviary), self.control_period_s, self.theta_bar, self.gamma_diag, self.sigma_mod, self.is_saturated
                        )
                        phi_val = np.array(phi_out)
                        u_nn = phi_val
                        current_control_integrand = self.K_I * e_ned_aviary + (self.K_RISE * np.sign(r1_ned_aviary))
                        u_provisional = u_nn + (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term
                        
                    case "integrated_resnet":
                        u_last = (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term
                        kappa_vec = jnp.array(np.concatenate((q_ned, q_dot_ned, qd_ned_aviary, qd_dot_ned_aviary, u_last)))
                        self.theta_hat, phi_out = self.compiled_update_step(
                            self.theta_hat, kappa_vec, jnp.array(r1_ned_aviary), self.control_period_s, self.theta_bar, self.gamma_diag, self.sigma_mod, self.is_saturated
                        )
                        phi_val = np.array(phi_out)
                        u_nn = phi_val
                        current_control_integrand = self.K_I * e_ned_aviary + (self.K_RISE * np.sign(r1_ned_aviary)) + u_nn
                        u_provisional = (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term

                    case "supertwisting":
                        norm_r1 = np.linalg.norm(r1_ned_aviary)
                        sgn_r1 = np.sign(r1_ned_aviary)
                        current_control_integrand = sgn_r1
                        proposed_integral = self.integral_control_term + (sgn_r1 * self.control_period_s)
                        u_provisional = self.k_2 * np.sqrt(norm_r1) * sgn_r1 + self.k_3 * proposed_integral + self.k_1 * e_dot_ned_aviary

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
                    if np.dot(e_ned_aviary[0:2], u_xy) > 0.0:
                        freeze_integrator[0:2] = True
                        
                if abs(u_provisional[2]) > self.acc_vert_max_mps2:
                    self.is_saturated = True
                    if np.sign(e_ned_aviary[2]) == np.sign(u_provisional[2]):
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
                    u_unclamped_ned_aviary = (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term
                else:
                    # Recalculate ST control using the globally available r1_ned_aviary
                    u_unclamped_ned_aviary = (self.k_2 * np.sqrt(np.linalg.norm(r1_ned_aviary)) * np.sign(r1_ned_aviary)) + (self.k_3 * self.integral_control_term) + (self.k_1 * e_dot_ned_aviary)

                # Final control action clipping
                u_clamped_ned_aviary = np.copy(u_unclamped_ned_aviary)
                u_xy_final = u_clamped_ned_aviary[0:2]
                norm_uxy_final = float(np.linalg.norm(u_xy_final))
                if norm_uxy_final > self.acc_hor_max_mps2:
                    u_clamped_ned_aviary[0:2] = u_xy_final * (self.acc_hor_max_mps2 / norm_uxy_final)
                
                if abs(u_clamped_ned_aviary[2]) > self.acc_vert_max_mps2:
                    u_clamped_ned_aviary[2] = self.acc_vert_max_mps2 * np.sign(u_clamped_ned_aviary[2])


                # --- CONTROL ACTION TRANSFORMS ---
                # 1. Start with clamped control output in NED_aviary frame
                # 2. Transform NED_aviary back to World ENU frame
                u_enu_mps2 = self.swap_ned_aviary_and_enu(u_clamped_ned_aviary)
                
                # 3. Transform World ENU into Body FLU frame
                # sensors.imu_orientation is the (Body FLU -> World ENU) active rotation quaternion.
                # Left-multiplying a World ENU vector by the inverse of this quaternion yields the Body FLU vector.
                quat_rotation_enu = Rotation.from_quat(sensors.imu_orientation, scalar_first=False)
                u_flu_mps2 = quat_rotation_enu.inv().apply(u_enu_mps2) 

                if flight_mode == 'TRAJECTORY':
                    # New cost function integrand: J = int( q_e * t * ||e(t)||^2 + r_u * ||u(t)||^2 ) dt
                    current_cost_integrand = (self.q_e * current_t_for_cost * (np.linalg.norm(e_ned_aviary)**2)) + \
                                             (self.r_u * (np.linalg.norm(u_clamped_ned_aviary)**2))
                    self.cost_J += (self.control_period_s / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                    self.last_cost_integrand = current_cost_integrand

                # P Controller for yaw (with wrap-around fix)
                # Shortest angular error in degrees: (target - current + 180) % 360 - 180
                e_yaw_deg = (self.yaw_des_deg - yaw_enu + 180.0) % 360.0 - 180.0
                yaw_rate_cmd = -self.K_P_yaw * e_yaw_deg
                
                # --- DATA COLLECTION ---
                if flight_mode == 'TRAJECTORY':
                    sim_time_current = step * self.control_period_s
                    self.time_history.append(sim_time_current)
                    self.error_norm_history.append(float(np.linalg.norm(e_ned_aviary)))
                    self.q_history.append(q_ned.tolist())
                    self.qd_history.append(qd_ned_aviary.tolist())
                    self.u_history.append(u_clamped_ned_aviary.tolist())
                    self.e_history.append(e_ned_aviary.tolist())
                    
                    if hasattr(self, 'theta_hat'):
                        self.weight_history.append(np.array(self.theta_hat).tolist())
                    else:
                        self.weight_history.append([])

                drone.step_with_acceleration(
                    ax=u_flu_mps2[0],
                    ay=u_flu_mps2[1],
                    az=u_flu_mps2[2],
                    yaw_rate=yaw_rate_cmd,
                    count=steps_per_tick
                )

                step += 1
                sim_time = step * self.control_period_s
                
                if self.sim_speed > 0.0:
                    target_wall_time = sim_time / self.sim_speed
                    sleep_time = target_wall_time - (time.perf_counter() - wall_start)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
            
            sim_stop_time_perf = time.perf_counter()
            sim_run_time_realworld = sim_stop_time_perf - sim_start_time_perf
            print(f"[*] Simulation took {round(sim_run_time_realworld, 1)} real seconds for {self.sim_length_s} simulated seconds. Scale: {round(self.sim_length_s / sim_run_time_realworld , 1)}. Specified speed was {self.sim_speed}.")

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
            if self.config.get('save_data', False):
                traj_num = self.config.get('desired_trajectory', 1)
                controller = self.controller_type
                
                output_dir = os.path.join("output", f"traj{traj_num}", controller)
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                csv_path = os.path.join(output_dir, f"{timestamp}.csv")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    header = ['time', 'error_norm', 'q_x', 'q_y', 'q_z', 'qd_x', 'qd_y', 'qd_z', 'u_x', 'u_y', 'u_z', 'e_x', 'e_y', 'e_z']
                    if len(self.weight_history) > 0 and len(self.weight_history[0]) > 0:
                        num_weights = len(self.weight_history[0])
                        header.extend([f"w_{i}" for i in range(num_weights)])
                    
                    writer.writerow(header)
                    
                    for i in range(len(self.time_history)):
                        row = [
                            self.time_history[i],
                            self.error_norm_history[i],
                            *self.q_history[i],
                            *self.qd_history[i],
                            *self.u_history[i],
                            *self.e_history[i]
                        ]
                        if len(self.weight_history[i]) > 0:
                            row.extend(self.weight_history[i])
                        writer.writerow(row)
                        
                print(f"Saved data to {csv_path}")

            sim.pause()
            return self.cost_J, e_rms, u_rms