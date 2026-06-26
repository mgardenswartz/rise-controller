import sys
import math
import time
from functools import partial
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry

import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu") # Force CPU to avoid Docker GPU passthrough issues
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from jax_resnet import resnet_network, init_resnet_weights

# --- GLOBAL CONSTANTS ---
MPC_ACC_HOR_MAX = 6.0
MPC_ACC_VERT_MAX = 3.0

# Safety Boundaries (Mapped to your 25x12x8m Lab)
MAX_SAFE_X = 5.0
MAX_SAFE_Y = 11.5
MAX_SAFE_Z_CEIL = 0.0
MAX_SAFE_Z_FLOOR = -5.5

# Timeouts and Tolerances
ODOM_TIMEOUT_SEC = 5
SETTLE_TICKS = 50
HOVER_TOLERANCE = 0.2

# Trajectory Settings
TRAJ_PERIOD = 20.0
TRAJ_ALPHA_WARP = 0.8
# ------------------------

@jax.jit
def discrete_projection(
    theta_hat: jax.Array,
    theta_dot_unprojected: jax.Array,
    dt: float,
    theta_bar: float,
    gamma_diag: jax.Array
) -> jax.Array:
    theta_temp = theta_hat + dt * theta_dot_unprojected
    is_inside = jnp.sum(theta_temp**2) <= theta_bar**2
    
    def apply_projection(_: None) -> jax.Array:
        gamma_min = jnp.min(gamma_diag)
        norm_temp = jnp.linalg.norm(theta_temp)
        eta_upper_init = (norm_temp / theta_bar - 1.0) / gamma_min
        init_state = (0.0, eta_upper_init)
        
        def bisection_step(i, state):
            eta_low, eta_high = state
            eta_mid = 0.5 * (eta_low + eta_high)
            theta_test = theta_temp / (1.0 + eta_mid * gamma_diag)
            val = jnp.sum(theta_test**2) - theta_bar**2
            new_low = jnp.where(val > 0, eta_mid, eta_low)
            new_high = jnp.where(val > 0, eta_high, eta_mid)
            return (new_low, new_high)
        
        final_low, final_high = jax.lax.fori_loop(0, 30, bisection_step, init_state)
        eta_opt = 0.5 * (final_low + final_high)
        return theta_temp / (1.0 + eta_opt * gamma_diag)

    def bypass_projection(_: None) -> jax.Array:
        return theta_temp

    return jax.lax.cond(is_inside, bypass_projection, apply_projection, None)


class AviaryRiseNode(Node):
    def __init__(self,):
        super().__init__('aviary_rise_node')

        self.declare_parameter('vehicle_name', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('z', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k1', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k2', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k3', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k_rise', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('controller_type', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('gamma', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('sigma_mod', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k_0', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('k_i', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('plot', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('hidden_width', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('num_blocks', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('theta_bar', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('initial_weights', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('d_in', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('d_out', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('q_e', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('w_fail', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('r_u', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('sim_time', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('control_frequency', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('h_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('o_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('shortcut_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('T_period', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('alpha_warp', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.target_z = self.get_parameter('z').value
        self.controller_type = self.get_parameter('controller_type').value
        self.enable_plotting = self.get_parameter('plot').value
        
        self.k1 = self.get_parameter('k1').value
        k2 = self.get_parameter('k2').value
        k3 = self.get_parameter('k3').value
        
        self.K_P = (self.k1 * k2) + (self.k1 * k3) + (k2 * k3) + 1.0
        self.K_I = (self.k1 * k2 * k3) + self.k1
        self.K_D = self.k1 + k2 + k3
        self.K_RISE = self.get_parameter('k_rise').value

        self.q_e = self.get_parameter('q_e').value
        self.r_u = self.get_parameter('r_u').value
        self.w_fail = self.get_parameter('w_fail').value

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            # durability=DurabilityPolicy.TRANSIENT_LOCAL,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, f'/{self.vehicle_name}/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, f'/{self.vehicle_name}/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, f'/{self.vehicle_name}/fmu/in/vehicle_command', qos_profile)

        self.status_sub = self.create_subscription(
            VehicleStatus, f'/{self.vehicle_name}/fmu/out/vehicle_status', self.status_callback, qos_profile)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, f'/{self.vehicle_name}/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)

        self.nav_state = 0
        self.is_armed = False
        self.in_offboard_mode = False
        self.vehicle_system_id = 1
        self.vehicle_component_id = 1
        
        self.latest_odom = None
        self.initial_position_locked = False
        self.start_x = 0.0
        self.start_y = 0.0

        self.experiment_state = 0
        self.offboard_setpoint_counter = 0
        self.settle_counter = 0
        
        self.t_f = self.get_parameter('sim_time').value
        self.T_period = self.get_parameter('T_period').value
        self.t_0 = 0.0
        self.last_t = 0.0
        
        # Virtual Target State Tracking
        self.tau = 0.0 # Determines qd(t_0) and qd_dot(t_0)
        self.last_traj_time = 0.0
        self.alpha_warp = self.get_parameter('alpha_warp').value
        self.warp_c = 1.0 / math.sqrt(1.0 - self.alpha_warp)
        
        self.d_out = 3
        self.integral_term = np.zeros(self.d_out, dtype=np.float64)
        self.last_integrand = np.zeros(self.d_out, dtype=np.float64)
        self.cost_J = 0.0
        self.last_cost_integrand = 0.0
        
        self.is_saturated = False
        
        self.error_sq_integral = 0.0
        self.last_error_sq = 0.0
        self.time_history = []
        self.error_norm_history = []
        self.weight_history = []
        self.q_history = []
        self.qd_history = []

        control_frequency = self.get_parameter('control_frequency').value
        self.control_period = 1 / control_frequency
        self.control_timer = self.create_timer(self.control_period, self.control_timer_callback)
        self.offboard_spam_ticks = int(control_frequency) / 2.0

        # JAX NN Initializations (Bypassed if noresnet)
        if self.controller_type in ["baseline", "developed"]:
            self.d_in = self.get_parameter('d_in').value
        
            self.sigma_mod = self.get_parameter('sigma_mod').value
            self.theta_bar = self.get_parameter('theta_bar').value
            
            weights_list = self.get_parameter('initial_weights').value
            self.theta_hat = jnp.array(weights_list)
            
            # Utilizing the correct p-dimensional weight sizing for the projection matrix
            self.gamma_diag = jnp.ones(self.theta_hat.shape[0]) * self.get_parameter('gamma').value

            self.bound_resnet = jax.jit(partial(
                resnet_network,
                d_in=self.d_in,
                hidden_width=self.get_parameter('hidden_width').value,
                d_out=self.d_out,
                b=self.get_parameter('num_blocks').value,
                k_0=self.get_parameter('k_0').value,
                k_i=self.get_parameter('k_i').value,
                h_act_func=self.get_parameter('h_act_func').value,
                o_act_func=self.get_parameter('o_act_func').value,
                shortcut_act_func=self.get_parameter('shortcut_act_func').value,
            ))
            
            @jax.jit
            def compiled_update_step(theta_hat, x_vec, r1_vec, dt, theta_bar, gamma_diag, s_mod, saturated):
                phi_val, vjp_fn = jax.vjp(lambda t: self.bound_resnet(t, x_vec), theta_hat)
                grad_term = vjp_fn(r1_vec)[0]
                
                theta_dot_unprojected = gamma_diag * (grad_term - s_mod * theta_hat)
                theta_next = discrete_projection(
                    theta_hat=theta_hat,
                    theta_dot_unprojected=theta_dot_unprojected,
                    dt=dt,
                    theta_bar=theta_bar,
                    gamma_diag=gamma_diag,
                )
                final_theta = jax.lax.select(saturated, theta_hat, theta_next)
                return final_theta, phi_val
                
            self.compiled_update_step = compiled_update_step
            self.enforce_realtime_constraint()

        self.get_logger().info(f"Node Booted. Controller: {self.controller_type.upper()}")
        self.watchdog_timer = self.create_timer(1.0, self.watchdog_callback)
        self.seconds_without_odom = 0

    def enforce_realtime_constraint(self) -> None:
        dummy_x = jnp.zeros(self.d_in)
        dummy_r1 = jnp.zeros(self.d_out)
        
        self.get_logger().info("[JAX] Compiling XLA Graph on CPU...")
        _, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()
        
        start_time = time.perf_counter()
        _, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()
        hot_time = time.perf_counter() - start_time
        
        self.get_logger().info(f"[JAX] Hot-path latency: {hot_time*1000:.2f} ms")
        if hot_time > self.control_period:
            self.get_logger().fatal(f"[ERROR] Execution time {hot_time}s exceeds {self.control_period}s limit!")
            # sys.exit(1)
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 0.0)



    def status_callback(self, msg: VehicleStatus) -> None:
        self.nav_state = msg.nav_state
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_offboard_mode = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.vehicle_system_id = msg.system_id
        self.vehicle_component_id = msg.component_id

    def odom_callback(self, msg: VehicleOdometry) -> None:
        self.latest_odom = msg
        if not self.initial_position_locked:
            self.start_x = float(msg.position[0])
            self.start_y = float(msg.position[1])
            self.initial_position_locked = True

    def watchdog_callback(self) -> None:
        if not self.initial_position_locked:
            self.seconds_without_odom += 1
            if self.seconds_without_odom >= ODOM_TIMEOUT_SEC:
                self.get_logger().info(f"Warning - No odometry")
                sys.exit(1)

    def publish_vehicle_command(self, command: int, param1: float, param2: float) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = command
        msg.target_system = self.vehicle_system_id
        msg.target_component = self.vehicle_component_id
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)

    def publish_offboard_control_mode(self, position: bool, acceleration: bool) -> None:
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = position
        msg.velocity = False
        msg.acceleration = acceleration
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint_position(self, x: float, y: float, z: float) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = 0.0 
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_trajectory_setpoint_acceleration(self, ax: float, ay: float, az: float) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float(ax), float(ay), float(az)]
        msg.yaw = 0.0
        self.trajectory_setpoint_publisher.publish(msg)

    def check_safety_boundary(self, q: np.ndarray) -> bool:
        if not (-MAX_SAFE_X <= q[0] <= MAX_SAFE_X): return True 
        if not (-MAX_SAFE_Y <= q[1] <= MAX_SAFE_Y): return True 
        if not (MAX_SAFE_Z_FLOOR <= q[2] <= MAX_SAFE_Z_CEIL): return True 
        return False

    def get_desired_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = t - self.last_traj_time
        self.last_traj_time = t
        if dt < 0: dt = 0.0
        
        w = (2.0 * math.pi) / self.T_period
        
        # Applying the scalar to correct the phase integration speed
        tau_dot = self.warp_c * (1.0 - self.alpha_warp * (math.sin(w * self.tau)**2))
        tau_ddot = -self.warp_c * self.alpha_warp * w * math.sin(2.0 * w * self.tau) * tau_dot
        
        self.tau += tau_dot * dt
        
        wx = 2.0 * w
        wy = 1.0 * w
        wz = 4.0 * w
        zc = self.target_z 
        
        xd = 3.0 * math.sin(wx * self.tau)
        yd = 8.0 * math.sin(wy * self.tau)
        #zd = 0.5 * math.sin(wz * self.tau) + zc
        zd = zc
        
        vxd = (3.0 * wx * math.cos(wx * self.tau)) * tau_dot
        vyd = (8.0 * wy * math.cos(wy * self.tau)) * tau_dot
        #vzd = (0.5 * wz * math.cos(wz * self.tau)) * tau_dot
        vzd = 0.0
        
        axd = -(3.0 * wx**2 * math.sin(wx * self.tau)) * (tau_dot**2) + (3.0 * wx * math.cos(wx * self.tau)) * tau_ddot
        ayd = -(8.0 * wy**2 * math.sin(wy * self.tau)) * (tau_dot**2) + (8.0 * wy * math.cos(wy * self.tau)) * tau_ddot
        #azd = -(0.5 * wz**2 * math.sin(wz * self.tau)) * (tau_dot**2) + (0.5 * wz * math.cos(wz * self.tau)) * tau_ddot
        azd = 0.0
        
        return (
            np.array([xd, yd, zd], dtype=np.float64),
            np.array([vxd, vyd, vzd], dtype=np.float64),
            np.array([axd, ayd, azd], dtype=np.float64)
        )

    def control_timer_callback(self) -> None:
        if not self.initial_position_locked or self.latest_odom is None:
            return

        current_timestamp_s: float = self.latest_odom.timestamp / 1e6

        # Publish offboard control mode first so we return early from call back when we leave offboard mode via RC without interrupting offboard heartbeats
        if self.experiment_state == 3:
            self.publish_offboard_control_mode(position=False, acceleration=True)
        else:
            self.publish_offboard_control_mode(position=True, acceleration=False)
        

        if self.experiment_state == 0:
            # self.publish_offboard_control_mode(position=True, acceleration=False)
            # self.publish_trajectory_setpoint_position(self.start_x, self.start_y, self.target_z)
            self.publish_trajectory_setpoint_position(self.start_x, self.start_y, 0.0)
            # if self.offboard_setpoint_counter >= SETTLE_TICKS:
            #     # FIX: Now dynamically pulses at 2Hz regardless of control frequency
            #     if self.offboard_setpoint_counter % self.offboard_spam_ticks == 0:
            #         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            #         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
            # self.offboard_setpoint_counter += 1
            if self.nav_state == 14 and self.is_armed:
                self.experiment_state = 1

        # Now check nav state and return early if not in offboard mode, avoid sending setpoints if pilot tries to manually take control
        if not self.in_offboard_mode:
            return

        elif self.experiment_state == 1:
            # self.publish_offboard_control_mode(position=True, acceleration=False)
            self.publish_trajectory_setpoint_position(self.start_x, self.start_y, self.target_z)
            q_current = np.array(self.latest_odom.position, dtype=np.float64)
            if abs(q_current[2] - self.target_z) < HOVER_TOLERANCE:
                self.experiment_state = 2

        elif self.experiment_state == 2:
            # self.publish_offboard_control_mode(position=True, acceleration=False)
            self.publish_trajectory_setpoint_position(self.start_x, self.start_y, self.target_z)
            self.settle_counter += 1
            if self.settle_counter >= SETTLE_TICKS:
                self.t_0 = current_timestamp_s
                self.last_t = 0.0
                
                # Reset Virtual Target trackers for clean initialization
                self.tau = 0.0
                self.last_traj_time = 0.0
                
                self.experiment_state = 3

        elif self.experiment_state == 3:
            t = current_timestamp_s - self.t_0
            dt = t - self.last_t
            if dt <= 0.0: return
                
            q = np.array(self.latest_odom.position, dtype=np.float64)
            q_dot = np.array(self.latest_odom.velocity, dtype=np.float64)
            
            if self.check_safety_boundary(q):
                self.cost_J += self.w_fail * ((self.t_f - t) ** 2)
                self.get_logger().error(f"[RESULT] ITAE_COST = {self.cost_J:.4f} (BOUNDARY FAILURE)")
                sys.exit(0)

            qd, qd_dot, qd_ddot = self.get_desired_state(t)
            
            e = qd - q
            e_dot = qd_dot - q_dot
            r1 = e_dot + (self.k1 * e)

            u = np.zeros(3, dtype=np.float64)
            phi_val = np.zeros(3, dtype=np.float64)
            
            if self.controller_type == "noresnet":
                current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                if self.is_saturated: current_integrand = np.zeros(3, dtype=np.float64)
                
                self.integral_term += (dt / 2.0) * (current_integrand + self.last_integrand)
                self.last_integrand = current_integrand
                
                # noresnet actively uses the time-invariant feedforward term
                u = qd_ddot + (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                
            elif self.controller_type == "baseline":
                # Baseline strictly ignores qd_ddot in the input vector and feedforward
                x_vec = jnp.array(np.concatenate((q, q_dot, qd, qd_dot)))
                self.theta_hat, phi_out = self.compiled_update_step(
                    self.theta_hat, x_vec, jnp.array(r1), dt, self.theta_bar, self.gamma_diag, self.sigma_mod, self.is_saturated
                )
                phi_val = np.array(phi_out, dtype=np.float64)
                
                current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                if self.is_saturated: current_integrand = np.zeros(3, dtype=np.float64)
                
                self.integral_term += (dt / 2.0) * (current_integrand + self.last_integrand)
                self.last_integrand = current_integrand
                u = phi_val + (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                
            elif self.controller_type == "developed":
                # Developed strictly ignores qd_ddot in the input vector and feedforward
                u_last =  (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                kappa_vec = jnp.array(np.concatenate((q, q_dot, qd, qd_dot, u_last)))
                
                self.theta_hat, phi_out = self.compiled_update_step(
                    self.theta_hat, kappa_vec, jnp.array(r1), dt, self.theta_bar, self.gamma_diag, self.sigma_mod, self.is_saturated
                )
                phi_val = np.array(phi_out, dtype=np.float64)
                
                current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1)) + phi_val
                if self.is_saturated: current_integrand = np.zeros(3, dtype=np.float64)
                
                self.integral_term += (dt / 2.0) * (current_integrand + self.last_integrand)
                self.last_integrand = current_integrand
                u = (self.K_P * e) + (self.K_D * e_dot) + self.integral_term

            norm_e = float(np.linalg.norm(e))
            norm_u = float(np.linalg.norm(u))
            
            current_error_sq = float(norm_e ** 2)
            self.error_sq_integral += (dt / 2.0) * (current_error_sq + self.last_error_sq)
            self.last_error_sq = current_error_sq
            self.time_history.append(t)
            self.error_norm_history.append(norm_e)
            self.q_history.append(q.tolist())
            self.qd_history.append(qd.tolist())

            if self.controller_type in ["baseline", "developed"]:
                self.weight_history.append(np.array(self.theta_hat).flatten().tolist())

            current_cost_integrand = (t * self.q_e * (norm_e ** 2)) + (self.r_u * (norm_u ** 2))
            self.cost_J += (dt / 2.0) * (current_cost_integrand + self.last_cost_integrand)
            self.last_cost_integrand = current_cost_integrand
            
            # Anti-Windup Clamping checks
            self.is_saturated = False
            u_xy = u[0:2]
            norm_uxy = np.linalg.norm(u_xy)
            if norm_uxy > MPC_ACC_HOR_MAX:
                u[0:2] = u_xy * (MPC_ACC_HOR_MAX / norm_uxy)
                self.is_saturated = True
                
            if abs(u[2]) > MPC_ACC_VERT_MAX:
                u[2] = MPC_ACC_VERT_MAX * np.sign(u[2])
                self.is_saturated = True

            # self.publish_offboard_control_mode(position=False, acceleration=True)
            self.publish_trajectory_setpoint_acceleration(u[0], u[1], u[2])

            self.last_t = t

            if t >= self.t_f:
                rms_error = math.sqrt(self.error_sq_integral / self.t_f)
                self.get_logger().info(f"[RESULT] ITAE_COST = {self.cost_J:.4f}")
                self.get_logger().info(f"[RESULT] RMS_ERROR = {rms_error:.4f}")
                
                if self.enable_plotting:
                    import csv
                    csv_filename = f"/home/root/trial_data_cost_{int(self.cost_J)}.csv"
                    try:
                        with open(csv_filename, mode='w', newline='') as file:
                            writer = csv.writer(file)

                            # NEW: Expanded Headers
                            headers = ["Time_s", "Error_Norm_m", "x", "y", "z", "xd", "yd", "zd"]
                            
                            if self.controller_type in ["baseline", "developed"] and self.weight_history:
                                num_weights = len(self.weight_history[0])
                                headers += [f"W{i}" for i in range(num_weights)]
                            
                            writer.writerow(headers)
                            
                            for i in range(len(self.time_history)):
                                row = [
                                    self.time_history[i], 
                                    self.error_norm_history[i],
                                    self.q_history[i][0], self.q_history[i][1], self.q_history[i][2],
                                    self.qd_history[i][0], self.qd_history[i][1], self.qd_history[i][2]
                                ]
                                if self.controller_type in ["baseline", "developed"] and self.weight_history:
                                    row += self.weight_history[i]
                                writer.writerow(row)
                                    
                        self.get_logger().info(f"Telemetry saved to {csv_filename}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to write CSV: {e}")
                sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = AviaryRiseNode()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        pass  
    finally:
        try:
            if rclpy.ok(): node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)
        except Exception: pass 
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
