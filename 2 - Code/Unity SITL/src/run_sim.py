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
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner
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
        self.run_length_s = self.config['run_length_s']
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

        self.init_x_ned = self.config[f'traj{self.desired_traj}_init_x_m_ned_aviary']
        self.init_y_ned = self.config[f'traj{self.desired_traj}_init_y_m_ned_aviary']
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

    def step_with_acceleration_ned(self, runner, px4, a_ned: np.ndarray, yaw_ned: float) -> Any:
        """
        Sends a raw acceleration setpoint directly to PX4 via MAVLink.
        This uses PX4's internal acceleration controller, bypassing position and velocity.
        """
        import time
        from pymavlink import mavutil
        
        # MAVLink acceleration mask: keep accel + yaw; ignore pos, vel, yaw_rate
        # IGNORE_POS (1+2+4) + IGNORE_VEL (8+16+32) + IGNORE_YAW_RATE (2048) = 2111
        _ACCEL_MASK = 2111 
        
        # Map a_ned from NED_Aviary (E, -N, -U) to standard NED (N, E, D)
        # N = -y_aviary
        # E = x_aviary
        # D = z_aviary
        a_ned_px4 = [
            -float(a_ned[1]),
            float(a_ned[0]),
            float(a_ned[2])
        ]
        
        with px4._lock:
            tgt_sys = px4._mav.target_system
            tgt_comp = px4._mav.target_component
            t_ms = int(time.time() * 1e3) & 0xFFFFFFFF
            px4._mav.mav.set_position_target_local_ned_send(
                t_ms, tgt_sys, tgt_comp, mavutil.mavlink.MAV_FRAME_LOCAL_NED, _ACCEL_MASK,
                0.0, 0.0, 0.0, # pos
                0.0, 0.0, 0.0, # vel
                a_ned_px4[0], a_ned_px4[1], a_ned_px4[2], # accel
                float(yaw_ned), 0.0 # yaw, yaw_rate
            )
            
        # Lockstep advance physics
        if runner.offboard_io_yield_s:
            time.sleep(runner.offboard_io_yield_s)
            
        n_hil = runner.hil_per_control
        for _ in range(n_hil):
            runner.hil_tick()
            
        return runner.sensors

    def run(self) -> tuple[float, float, float]:
        """Executes the simulation loop and returns the cost."""
        QUADSIM_HOST = "localhost"
        QUADSIM_PORT = 5555
        PX4_HOST = "127.0.0.1"
        PX4_INSTANCE = 0
        PX4_CLIENT = False
        
        sim = QuadSim(host=QUADSIM_HOST, command_port=QUADSIM_PORT, telemetry_port=QUADSIM_PORT + 1)
        sim.connect()
        # Set wind as requested
        sim.set_wind(enabled=True, wind_speed=0.0)

        bridge = HilBridge(px4_host=PX4_HOST, instance=PX4_INSTANCE, px4_client=PX4_CLIENT)
        px4 = Px4Link(stream_hz=0, instance=PX4_INSTANCE)
        runner = LockstepRunner(sim, bridge, px4, control_hz=self.control_frequency_hz)
        
        if self.sim_speed > 0.0:
            runner.speed_cap = self.sim_speed
            print(f"[*] Limiting to max real-time speed: {self.sim_speed}")
        else:
            print("[*] Running unthrottled (as fast as possible)")

        # Spawn the drone exactly where the trajectory starts (X, Y) but on the ground (Z=0)
        # We do this BEFORE PX4 starts so the EKF initializes perfectly flat on the ground.
        qd_ned_aviary, _, _ = self.traj_gen.get_desired_state(0.0)
        start_enu = self.swap_ned_aviary_and_enu(qd_ned_aviary)
        print(f"[*] Resetting drone pose to trajectory start (grounded): X={start_enu[0]:.2f}, Y={start_enu[1]:.2f}, Z=0.0")
        
        import math
        yaw_enu_rad = math.radians(-self.yaw_des_deg)
        qw = math.cos(yaw_enu_rad / 2.0)
        qz = math.sin(yaw_enu_rad / 2.0)
        
        # Connect to sim if not connected, so we can reset before runner.start()
        if not sim.connected:
            sim.connect()
        sim.drone().reset_pose(x=start_enu[0], y=start_enu[1], z=0.0, qw=qw, qz=qz)

        runner.start()
        runner.wait_px4()
        px4.configure_offboard_no_rc()
        
        print("[*] Setting PX4 MPC parameters for Unity mass...")
        px4.param_set("MPC_THR_HOVER", 0.8)
        px4.param_set("MPC_ACC_UP_MAX", 10.0)
        px4.param_set("MPC_ACC_DOWN_MAX", 10.0)
        px4.param_set("MPC_ACC_HOR_MAX", 10.0)

        print("[*] EKF Ready! Engaging OFFBOARD...")
        if not runner.wait_ekf_ready():
            raise SystemExit("EKF never set home")
        if not runner.wait_heading():
            raise SystemExit("heading chain failed")
        if px4.state.pos_enu is None:
            raise SystemExit("PX4 local position unavailable after EKF ready")
            
        traj_t = 0.0
        
        # --- PURE ACCELERATION OFFBOARD PRIME ---
        # Instead of using position commands and fly_to(), we prime the stream
        # with zero acceleration (which translates to hover thrust in PX4)
        # and then enter the main control loop directly from the ground!
        yaw_ned = math.pi/2 - math.radians(self.init_yaw_deg)
        print("Priming PX4 setpoint stream with pure acceleration commands...")
        
        # Prime the stream for 1.0 simulated seconds before requesting offboard
        prime_deadline = runner.sim_time + 1.0
        while runner.sim_time < prime_deadline:
            self.step_with_acceleration_ned(runner, px4, np.array([0.0, 0.0, 0.0]), float(yaw_ned))
            
        print("Engaging OFFBOARD and arming motors...")
        px4.request_offboard()
        px4.request_arm(arm=True)
        
        # Wait up to 5.0 seconds for PX4 to accept
        accept_deadline = runner.sim_time + 5.0
        while runner.sim_time < accept_deadline:
            self.step_with_acceleration_ned(runner, px4, np.array([0.0, 0.0, 0.0]), float(yaw_ned))
            if px4.in_offboard() and px4.is_armed():
                break
                
        if not (px4.in_offboard() and px4.is_armed()):
            raise SystemExit("PX4 refused OFFBOARD/arm with acceleration setpoint")
            
        print("Taking off using pure acceleration controller!")
        sim_start_time_perf = time.perf_counter()
        traj_start_time = runner.sim_time
        
        flight_mode = 'TRAJECTORY'
        try:
            while traj_t < self.run_length_s:
                # Read sensors from PX4
                q_ned = np.array(self.swap_ned_aviary_and_enu(px4.state.pos_enu))
                q_dot_ned = np.array(self.swap_ned_aviary_and_enu(px4.state.vel_enu))
                yaw_enu = px4.state.yaw_enu
                
                # Check bounds
                if self.check_boundary_escape(q_ned_aviary=q_ned):
                    print(f"[!] Got too close to a wall! Position: {q_ned}. Exiting.")
                    self.cost_J += self.w_fail * ((self.run_length_s - traj_t) ** 2)
                    break
        
                # State machine
                current_t_for_cost = traj_t
                qd_ned_aviary, qd_dot_ned_aviary, qd_ddot_ned_aviary = self.traj_gen.get_desired_state(traj_t)
                traj_t = runner.sim_time - traj_start_time
    
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
                            self.theta_hat, x_vec, jnp.array(r1_ned_aviary), self.control_period_s, self.theta_bar, self.gamma, self.sigma_mod, self.is_saturated
                        )
                        phi_val = np.array(phi_out)
                        u_nn = phi_val
                        current_control_integrand = self.K_I * e_ned_aviary + (self.K_RISE * np.sign(r1_ned_aviary))
                        u_provisional = u_nn + (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term
                        
                    case "integrated_resnet":
                        u_last = (self.K_P * e_ned_aviary) + (self.K_D * e_dot_ned_aviary) + self.integral_control_term
                        kappa_vec = jnp.array(np.concatenate((q_ned, q_dot_ned, qd_ned_aviary, qd_dot_ned_aviary, u_last)))
                        self.theta_hat, phi_out = self.compiled_update_step(
                            self.theta_hat, kappa_vec, jnp.array(r1_ned_aviary), self.control_period_s, self.theta_bar, self.gamma, self.sigma_mod, self.is_saturated
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
                        u_provisional = qd_ddot_ned_aviary + self.k_2 * np.sqrt(norm_r1) * sgn_r1 + self.k_3 * proposed_integral + self.k_1 * e_dot_ned_aviary
    
                    case _:
                        raise ValueError(f"Unknown controller type {self.controller_type}")
                        
                # --- SATURATION & ANTI-WINDUP ---
                self.is_saturated = False
                
                # 1. Create a mask to track which specific axes should freeze integration
                freeze_integrator = np.array([False, False, False])
                
                u_clamped_ned_aviary = np.copy(u_provisional)
                u_xy = u_provisional[0:2]
                norm_uxy = float(np.linalg.norm(u_xy))
                if norm_uxy > self.acc_hor_max_mps2:
                    self.is_saturated = True
                    u_clamped_ned_aviary[0:2] = u_xy * (self.acc_hor_max_mps2 / norm_uxy)
                    if np.dot(e_ned_aviary[0:2], u_xy) > 0.0:
                        freeze_integrator[0:2] = True
                        
                if abs(u_provisional[2]) > self.acc_vert_max_mps2:
                    self.is_saturated = True
                    u_clamped_ned_aviary[2] = self.acc_vert_max_mps2 * np.sign(u_provisional[2])
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

                # --- COST EVALUATION ---
                # New cost function integrand: J = int( q_e * t * ||e(t)||^2 + r_u * ||u(t)||^2 ) dt
                current_cost_integrand = (self.q_e * current_t_for_cost * (np.linalg.norm(e_ned_aviary)**2)) + \
                                         (self.r_u * (np.linalg.norm(u_clamped_ned_aviary)**2))
                self.cost_J += (self.control_period_s / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                self.last_cost_integrand = current_cost_integrand

                # P Controller for yaw (with wrap-around fix)
                # Shortest angular error in degrees: (target - current + 180) % 360 - 180
                yaw_enu_deg = math.degrees(yaw_enu)
                e_yaw_deg = (self.yaw_des_deg - yaw_enu_deg + 180.0) % 360.0 - 180.0
                yaw_rate_cmd = -self.K_P_yaw * e_yaw_deg
                
                # We need to send yaw_ned to step_with_acceleration_ned.
                # Yaw ENU = -Yaw NED. Let's just pass the desired yaw in NED directly?
                # Actually, our target is self.yaw_des_deg. 
                # Convert desired yaw from ENU degrees to NED radians
                yaw_cmd_ned = math.pi/2 - math.radians(self.yaw_des_deg)
                
                # --- DATA COLLECTION ---
                sim_time_current = runner.sim_time
                self.time_history.append(sim_time_current)
                self.error_norm_history.append(float(np.linalg.norm(e_ned_aviary)))
                self.control_output_norm_history.append(float(np.linalg.norm(u_clamped_ned_aviary)))
                self.q_history.append(q_ned.tolist())
                self.qd_history.append(qd_ned_aviary.tolist())
                self.u_history.append(u_clamped_ned_aviary.tolist())
                self.e_history.append(e_ned_aviary.tolist())
                
                if hasattr(self, 'theta_hat'):
                    self.weight_history.append(np.array(self.theta_hat).tolist())
                else:
                    self.weight_history.append([])
    
                sensors = self.step_with_acceleration_ned(
                    runner=runner,
                    px4=px4,
                    a_ned=u_clamped_ned_aviary,
                    yaw_ned=yaw_cmd_ned
                )
                if sensors is None:
                    break
        finally:
            if px4.is_armed():
                runner.disarm()
            runner.close()
            sim.disconnect()
            
        sim_stop_time_perf = time.perf_counter()
        if 'sim_start_time_perf' in locals():
            sim_run_time_realworld = sim_stop_time_perf - sim_start_time_perf
            print(f"[*] Trajectory execution took {round(sim_run_time_realworld, 1)} real seconds for {self.run_length_s} simulated seconds. Scale: {round(self.run_length_s / max(sim_run_time_realworld, 0.001) , 1)}. Specified speed was {self.sim_speed}.")


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