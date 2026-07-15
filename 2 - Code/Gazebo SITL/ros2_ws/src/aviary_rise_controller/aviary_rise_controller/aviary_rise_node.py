import sys
import os
import math
import time
import csv
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
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from jax_resnet import resnet_network

class ExperimentState:
    STATE_INIT = 0
    STATE_TAKEOFF = 1
    STATE_FOLLOW_TRAJ = 2
    STATE_PAUSED = 3
    STATE_FAILSAFE = 99

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


@jax.jit
def traj1_spatial_derivs(
    tau: float,
    target_z: float,
    traj1_period: float,
    traj1_x_amp: float,
    traj1_y_amp: float,
    traj1_z_amp: float
) -> jax.Array:
    def pos_fn(t):
        w = (2.0 * jnp.pi) / traj1_period
        wx, wy, wz = 2.0 * w, 1.0 * w, 4.0 * w
        # Keeping your exact original orientation mappings
        return jnp.array([
            traj1_x_amp * jnp.sin(wx * t),
            traj1_y_amp * jnp.sin(wy * t),
            traj1_z_amp * jnp.sin(wz * t) + target_z
        ])
    return pos_fn(tau), jax.jacfwd(pos_fn)(tau), jax.jacfwd(jax.jacfwd(pos_fn))(tau)


@jax.jit
def traj2_spatial_derivs(
    theta: float,
    target_z: float,
    traj2_A: float
) -> jax.Array:
    def pos_fn(th):
        r = traj2_A * jnp.cos(2.0 * th)
        return jnp.array([
            r * jnp.cos(th),
            r * jnp.sin(th),
            target_z
        ])
    return pos_fn(theta), jax.jacfwd(pos_fn)(theta), jax.jacfwd(jax.jacfwd(pos_fn))(theta)


class AviaryRiseNode(Node):
    def __init__(self,):
        super().__init__('aviary_rise_node')

        # Experiment Config
        self.declare_parameter('is_gazebo', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('desired_trajectory', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        
        # Global Limits
        self.declare_parameter('mpc_acc_hor_max', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('mpc_acc_vert_max', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('safe_x_max', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('safe_y_max', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('safe_z_max', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('safe_z_min', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('odom_timeout_sec', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('settle_ticks', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('init_tol', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('init_z', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('watchdog_freq', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Trajectory 1
        self.declare_parameter('traj1_period', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj1_alpha_warp', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj1_x_amp', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj1_y_amp', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj1_center_z', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj1_z_amp', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Trajectory 2
        self.declare_parameter('traj2_target_speed', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj2_petal_radius', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('traj2_center_z', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Node / Controller Base
        self.declare_parameter('vehicle_name', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('controller_type', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('control_frequency', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('save_data', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('sim_time', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        
        # Gains & Cost weights
        self.declare_parameter('k_1', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k_2', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k_3', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k_rise', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('q_e', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('r_u', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('w_fail', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Neural Network
        self.declare_parameter('d_in', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('d_out', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('gamma', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('sigma_mod', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('k_0', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('k_i', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('hidden_width', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('num_blocks', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('theta_bar', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('initial_weights', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('h_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('o_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('shortcut_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))

        self.is_gazebo = self.get_parameter('is_gazebo').value
        self.desired_trajectory = self.get_parameter('desired_trajectory').value
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.controller_type = self.get_parameter('controller_type').value
        self.save_data = self.get_parameter('save_data').value
        self.t_f = self.get_parameter('sim_time').value
        
        self.acc_hor_max = self.get_parameter('mpc_acc_hor_max').value
        self.acc_vert_max = self.get_parameter('mpc_acc_vert_max').value
        self.safe_x_max = self.get_parameter('safe_x_max').value
        self.safe_y_max = self.get_parameter('safe_y_max').value
        self.safe_z_max = self.get_parameter('safe_z_max').value
        self.safe_z_min = self.get_parameter('safe_z_min').value
        self.odom_timeout = self.get_parameter('odom_timeout_sec').value
        
        # New Takeoff Parameters
        self.init_z = self.get_parameter('init_z').value
        self.init_tol = self.get_parameter('init_tol').value
        self.watchdog_freq = self.get_parameter('watchdog_freq').value

        self.target_z = self.get_parameter('traj1_center_z').value if self.desired_trajectory == 1 else self.get_parameter('traj2_center_z').value
        self.traj1_period = self.get_parameter('traj1_period').value
        self.traj1_alpha_warp = self.get_parameter('traj1_alpha_warp').value
        self.traj1_x_amp = self.get_parameter('traj1_x_amp').value
        self.traj1_y_amp = self.get_parameter('traj1_y_amp').value
        self.traj1_z_amp = self.get_parameter('traj1_z_amp').value
        self.traj1_warp_c = 1.0 / math.sqrt(1.0 - self.traj1_alpha_warp) if self.traj1_alpha_warp < 1.0 else 1.0
        
        self.traj2_v0 = self.get_parameter('traj2_target_speed').value
        self.traj2_A = self.get_parameter('traj2_petal_radius').value

        self.k_1 = self.get_parameter('k_1').value
        self.k_2 = self.get_parameter('k_2').value
        self.k_3 = self.get_parameter('k_3').value
        if self.controller_type == "pid":
            self.declare_parameter('K_P', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter('K_I', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter('K_D', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        else:
            self.K_P = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
            self.K_I = (self.k_1 * self.k_2 * self.k_3) + self.k_1
            self.K_D = self.k_1 + self.k_2 + self.k_3    

        self.K_RISE = self.get_parameter('k_rise').value
        self.q_e = self.get_parameter('q_e').value
        self.r_u = self.get_parameter('r_u').value
        self.w_fail = self.get_parameter('w_fail').value

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
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
        self.landing_command_sent = False
        
        self.latest_odom = None
        self.last_odom_ros_time = 0.0
        self.initial_position_locked = False
        self.start_x = 0.0
        self.start_y = 0.0

        self.experiment_state = ExperimentState.STATE_INIT
        self.t_0 = 0.0
        self.last_t = 0.0
        
        self.tau = 0.0
        self.theta = math.pi /4
        self.last_traj_time = 0.0
        
        self.d_out = self.get_parameter('d_out').value
        self.integral_term = np.zeros(self.d_out, dtype=np.float64)
        self.last_integrand = np.zeros(self.d_out, dtype=np.float64)
        self.st_integral = np.zeros(self.d_out, dtype=np.float64)
        
        self.cost_J = 0.0
        self.last_cost_integrand = 0.0
        self.cost_started = False
        
        self.is_saturated = False
        self.freeze_int_xy = False
        self.freeze_int_z = False
        
        self.error_sq_integral = 0.0
        self.last_error_sq = 0.0
        self.u_sq_integral = 0.0
        self.last_u_sq = 0.0
        self.time_history = []
        self.error_norm_history = []
        self.weight_history = []
        self.q_history = []
        self.qd_history = []

        self.offboard_timeout_sec = 15.0
        self.init_wait_start = 0.0
        self.last_auto_cmd_time = 0.0

        control_frequency = self.get_parameter('control_frequency').value
        self.control_period = 1 / control_frequency
        self.control_timer = self.create_timer(self.control_period, self.control_timer_callback)

        if self.controller_type in ["resnet", "integrated_resnet"]:
            self.d_in = self.get_parameter('d_in').value
            self.sigma_mod = self.get_parameter('sigma_mod').value
            self.theta_bar = self.get_parameter('theta_bar').value
            
            self.theta_hat = jnp.array(self.get_parameter('initial_weights').value)
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
                theta_next = discrete_projection(theta_hat, theta_dot_unprojected, dt, theta_bar, gamma_diag)
                final_theta = jax.lax.select(saturated, theta_hat, theta_next)
                return final_theta, phi_val
                
            self.compiled_update_step = compiled_update_step
            self.precompile_jax()

        self.get_logger().info(f"Node Booted. Controller: {self.controller_type.upper()} | Trajectory: {self.desired_trajectory} | Gazebo Mode: {self.is_gazebo}")
        self.watchdog_timer = self.create_timer(1.0/self.watchdog_freq, self.watchdog_callback)
        self.ticks_without_odom = 0

    def precompile_jax(self) -> None:
        dummy_x = jnp.zeros(self.d_in)
        dummy_r1 = jnp.zeros(self.d_out)
        self.get_logger().info("[JAX] Compiling XLA Graph on CPU...")
        
        # 1. Warmup Neural Net Update Mechanics
        self.theta_hat, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()
        
        # 2. Warmup Spatial Derivatives (Prevents JIT latency spike on first trajectory tick)
        _ = traj1_spatial_derivs(0.0, self.target_z, self.traj1_period, self.traj1_x_amp, self.traj1_y_amp, self.traj1_z_amp)
        _ = traj2_spatial_derivs(0.0, self.target_z, self.traj2_A)
        
        start_time = time.perf_counter()
        self.theta_hat, _ = self.compiled_update_step(self.theta_hat, dummy_x, dummy_r1, self.control_period, self.theta_bar, self.gamma_diag, self.sigma_mod, False)
        self.theta_hat.block_until_ready()
        hot_time = time.perf_counter() - start_time
        
        # Reset the weights back to true initial conditions
        self.theta_hat = jnp.array(self.get_parameter('initial_weights').value)
        self.theta_hat.block_until_ready()
        self.get_logger().info(f"[JAX] Hot-path latency: {hot_time*1000:.2f} ms")
        if hot_time > self.control_period:
            self.get_logger().fatal(f"[ERROR] Execution time {hot_time}s exceeds {self.control_period}s limit!")
            if self.is_gazebo:
                os._exit(1)
            else:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)

    def status_callback(self, msg: VehicleStatus) -> None:
        self.nav_state = msg.nav_state
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_offboard_mode = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.vehicle_system_id = msg.system_id
        self.vehicle_component_id = msg.component_id

    def odom_callback(self, msg: VehicleOdometry) -> None:
        self.latest_odom = msg
        self.ticks_without_odom = 0
        if not self.is_gazebo:
            self.last_odom_ros_time = self.get_clock().now().nanoseconds / 1e9
            
        if not self.initial_position_locked:
            self.start_x = float(msg.position[0])
            self.start_y = float(msg.position[1])
            self.initial_position_locked = True

    def watchdog_callback(self) -> None:
        self.ticks_without_odom += 1
        
        # At boot, we always use the tick counter
        if not self.initial_position_locked:
            if self.ticks_without_odom >= (self.odom_timeout * self.watchdog_freq):
                self.get_logger().fatal("NO ODOMETRY AT BOOT! EXITING NODE.")
                os._exit(1)
        else:
            if self.is_gazebo:
                # Use tick counter to survive Docker suspension / macOS sleep
                if self.ticks_without_odom >= (self.odom_timeout * self.watchdog_freq):
                    self.get_logger().fatal("YOUR PC IS RUNNING BEHIND SCHEDULE! EXITING SIM.")
                    self.trigger_failsafe_land()
            else:
                # Use original wall clock logic for real vehicle (sim-to-real)
                current_time = self.get_clock().now().nanoseconds / 1e9
                if (current_time - self.last_odom_ros_time) > self.odom_timeout:
                    self.get_logger().fatal("ODOM LOST! TRIGGERING FAILSAFE.")
                    self.trigger_failsafe_land()

    def publish_vehicle_command(self, command: int, param1: float, param2: float) -> None:
        self.get_logger().info(f"[DEBUG] Publishing command {command}")
        msg = VehicleCommand()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.vehicle_system_id
        msg.target_component = self.vehicle_component_id
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)

    def publish_offboard_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = True
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint_acceleration(self, ax: float, ay: float, az: float) -> None:
        msg = TrajectorySetpoint()
        msg.acceleration = [ax, ay, az]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = 0.0
        if self.latest_odom is not None:
            msg.timestamp = self.latest_odom.timestamp
        self.trajectory_setpoint_publisher.publish(msg)

    def log_csv(self):
        traj_name = "figure_eight"
        if self.desired_trajectory == 1:
            pass
        elif self.desired_trajectory == 2:
            traj_name = "rose"
        else:
            self.get_logger().fatal("INVALID DESIRED TRAJECTORY SELECTED.")
            self.trigger_failsafe_land()

        base_dir = f"/home/root/plot_data/{self.controller_type}/{traj_name}"
        os.makedirs(base_dir, exist_ok=True)
        
        existing_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
        iterable = len(existing_files) + 1
        csv_filename = os.path.join(base_dir, f"run_{iterable}.csv")
        try:
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                headers = ["Time_s", "Error_Norm_m", "x", "y", "z", "xd", "yd", "zd"]
                if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                    num_weights = len(self.weight_history[0])
                    headers += [f"W{i}" for i in range(num_weights)]
                writer.writerow(headers)
                for i in range(len(self.time_history)):
                    row = [
                        self.time_history[i], self.error_norm_history[i],
                        self.q_history[i][0], self.q_history[i][1], self.q_history[i][2],
                        self.qd_history[i][0], self.qd_history[i][1], self.qd_history[i][2]
                    ]
                    if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                        row += self.weight_history[i]
                    writer.writerow(row)
            self.get_logger().info(f"Telemetry saved to {csv_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    def trigger_failsafe_land(self) -> None:
        if self.save_data:
            self.log_csv()

        if self.is_gazebo:
            self.get_logger().error("GAZEBO FAILSAFE TRIGGERED. EXITING SIM.")
            os._exit(1)
        else:
            self.get_logger().error("LANDING!")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)
            self.experiment_state = ExperimentState.STATE_FAILSAFE

    def check_safety_boundary(self, q: np.ndarray) -> bool:
        if not (-self.safe_x_max <= q[0] <= self.safe_x_max):
            self.get_logger().fatal(f"X BOUNDARY BREACH: {q[0]} not in [{-self.safe_x_max}, {self.safe_x_max}]")
            return True 
        if not (-self.safe_y_max <= q[1] <= self.safe_y_max):
            self.get_logger().fatal(f"Y BOUNDARY BREACH: {q[1]} not in [{-self.safe_y_max}, {self.safe_y_max}]")
            return True 
        if not (self.safe_z_min <= q[2] <= self.safe_z_max):
            self.get_logger().fatal(f"Z BOUNDARY BREACH: {q[2]} not in [{self.safe_z_min}, {self.safe_z_max}]")
            return True 
        return False


    def get_desired_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.experiment_state == ExperimentState.STATE_TAKEOFF:
            # During takeoff, hold exactly above where it initialized
            return (np.array([self.start_x, self.start_y, self.init_z], dtype=np.float64), 
                    np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
            
        dt = t - self.last_traj_time
        self.last_traj_time = t
        if dt < 0: dt = 0.0

        if self.desired_trajectory == 1:
            w = (2.0 * math.pi) / self.traj1_period
            tau_dot = self.traj1_warp_c * (1.0 - self.traj1_alpha_warp * (math.sin(w * self.tau)**2))
            tau_ddot = -self.traj1_warp_c * self.traj1_alpha_warp * w * math.sin(2.0 * w * self.tau) * tau_dot
            
            self.tau += tau_dot * dt
            
            pos_jnp, dp_dtau, d2p_dtau2 = traj1_spatial_derivs(
                self.tau, self.target_z, self.traj1_period, self.traj1_x_amp, self.traj1_y_amp, self.traj1_z_amp
            )
            
            qd = np.array(pos_jnp, dtype=np.float64)
            qd_dot = np.array(dp_dtau, dtype=np.float64) * tau_dot
            qd_ddot = (np.array(d2p_dtau2, dtype=np.float64) * (tau_dot**2)) + (np.array(dp_dtau, dtype=np.float64) * tau_ddot)

        else:
            f_theta = 1.0 + 3.0 * (math.sin(2.0 * self.theta)**2)
            theta_dot = self.traj2_v0 / (self.traj2_A * math.sqrt(f_theta))
            
            sin_4theta = math.sin(4.0 * self.theta)
            theta_ddot = - (3.0 * (self.traj2_v0**2) * sin_4theta) / ((self.traj2_A**2) * (f_theta**2))
            
            self.theta += theta_dot * dt
            
            pos_jnp, dp_dth, d2p_dth2 = traj2_spatial_derivs(
                self.theta, self.target_z, self.traj2_A
            )
            
            qd = np.array(pos_jnp, dtype=np.float64)
            qd_dot = np.array(dp_dth, dtype=np.float64) * theta_dot
            qd_ddot = (np.array(d2p_dth2, dtype=np.float64) * (theta_dot**2)) + (np.array(dp_dth, dtype=np.float64) * theta_ddot)

        return qd, qd_dot, qd_ddot

    def control_timer_callback(self) -> None:
        if self.latest_odom is None: return
        current_timestamp_s = self.latest_odom.timestamp / 1e6

        match self.experiment_state:
            case ExperimentState.STATE_FAILSAFE:
                if not self.landing_command_sent:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)
                    self.landing_command_sent = True
                return

            case ExperimentState.STATE_INIT:
                self.landing_command_sent = False
                self.cost_started = False
                
                if self.init_wait_start == 0.0:
                    self.init_wait_start = current_timestamp_s
                
                if not self.in_offboard_mode: # haven't started or Joe dropped me out of off-board
                    self.get_logger().info("Waiting for PX4 Offboard Mode switch engagement...", throttle_duration_sec=2.0)
                    self.publish_offboard_heartbeat()
                    self.publish_trajectory_setpoint_acceleration(0.0, 0.0, 0.0)
                    
                    if self.is_gazebo:
                        # Throttle automatic MAVLink commands to 1 Hz to prevent PX4 Commander flood
                        if current_timestamp_s - self.last_auto_cmd_time > 1.0:
                            if not self.in_offboard_mode:
                                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                            if not self.is_armed:
                                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
                            self.last_auto_cmd_time = current_timestamp_s
                    return

                if self.in_offboard_mode and self.is_armed:
                    self.get_logger().info(f"ARMED & OFFBOARD validated. Initializing Takeoff to Z={self.init_z}.")
                    self.integral_term = np.zeros(self.d_out, dtype=np.float64)
                    self.last_integrand = np.zeros(self.d_out, dtype=np.float64)
                    self.st_integral = np.zeros(self.d_out, dtype=np.float64)
                    self.experiment_state = ExperimentState.STATE_TAKEOFF

            case ExperimentState.STATE_PAUSED:
                if self.in_offboard_mode and self.is_armed:
                    # Pilot re-engaged offboard mode. 
                    # We shift t_0 forward by the elapsed paused time so the trajectory completely froze during the dropout
                    time_paused = current_timestamp_s - self.pause_start_time
                    self.t_0 += time_paused  
                    self.experiment_state = self.pre_pause_state
                    self.get_logger().info("Offboard Mode re-engaged! Resuming trajectory seamlessly.")
                else:
                    self.get_logger().info("Trajectory Paused. Waiting for Pilot to re-engage Offboard...", throttle_duration_sec=2.0)
                return

            case ExperimentState.STATE_FOLLOW_TRAJ | ExperimentState.STATE_TAKEOFF:
                self.publish_offboard_heartbeat()

                if not self.in_offboard_mode:
                    if self.is_gazebo:
                        self.get_logger().error("PX4 LEFT OFFBOARD MODE IN SITL! FAILING TRIAL.")
                        self.trigger_failsafe_land()
                        return
                    else:
                        self.get_logger().warn("RC Pilot Intervention Detected! Pausing trajectory.", throttle_duration_sec=1.0)
                        self.pre_pause_state = self.experiment_state
                        self.experiment_state = ExperimentState.STATE_PAUSED
                        self.pause_start_time = current_timestamp_s
                        
                        # Reset memory integrals so they don't explosively un-wind when re-engaged
                        self.integral_term = np.zeros(self.d_out, dtype=np.float64)
                        self.last_integrand = np.zeros(self.d_out, dtype=np.float64)
                        self.st_integral = np.zeros(self.d_out, dtype=np.float64)
                        return
                
                # Check transitions before updating the clock if in TAKEOFF
                q = np.array(self.latest_odom.position, dtype=np.float64)
                if self.experiment_state == ExperimentState.STATE_TAKEOFF:
                    e_takeoff = np.array([self.start_x, self.start_y, self.init_z], dtype=np.float64) - q
                    if np.linalg.norm(e_takeoff) <= self.init_tol:
                        self.experiment_state = ExperimentState.STATE_FOLLOW_TRAJ
                        # Reset t_0 so the trajectory clock starts at exactly 0.0 now
                        self.t_0 = current_timestamp_s
                        self.last_t = 0.0
                        self.last_traj_time = 0.0
                        self.tau = 0.0
                        self.theta = math.pi / 4.0
                        self.get_logger().info(f"TAKEOFF SETTLED. Step Response Triggered: Starting Trajectory {self.desired_trajectory}.")

                # If in TAKEOFF, t_0 hasn't been set to the trajectory clock yet, so t evaluates to arbitrary.
                # However get_desired_state(t) strictly ignores t during TAKEOFF.
                t = current_timestamp_s - self.t_0 if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ else 0.0
                dt = t - self.last_t if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ else self.control_period
                
                q_dot = np.array(self.latest_odom.velocity, dtype=np.float64)
                
                if self.check_safety_boundary(q):
                    if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                        self.cost_J += self.w_fail * ((self.t_f - t) ** 2)
                        self.get_logger().error(f"[RESULT] ITAE_COST = {self.cost_J:.4f} (BOUNDARY FAILURE)")
                    self.trigger_failsafe_land()
                    return

                qd, qd_dot, _ = self.get_desired_state(t)
                e = qd - q
                e_dot = qd_dot - q_dot
                r1 = e_dot + (self.k_1 * e)

                u = np.zeros(self.d_out, dtype=np.float64)
                phi_val = np.zeros(self.d_out, dtype=np.float64)
                
                match self.controller_type:
                    case "baseline" | "pid":
                        current_integrand = (self.K_I * e) + (self.controller_type == "baseline") * (self.K_RISE * np.sign(r1))
                        delta_int = (dt / 2.0) * (current_integrand + self.last_integrand)
                        if not self.freeze_int_xy:
                            self.integral_term[0:2] += delta_int[0:2]
                        if not self.freeze_int_z:
                            self.integral_term[2] += delta_int[2]
                        self.last_integrand = current_integrand
                        u = (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                        
                    case "resnet":
                        if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ: 
                            x_vec = jnp.array(np.concatenate((q, q_dot, qd, qd_dot)))
                            
                            t_start_jax = time.perf_counter()
                            self.theta_hat, phi_out = self.compiled_update_step(
                                self.theta_hat, x_vec, jnp.array(r1), dt, self.theta_bar, self.gamma_diag, self.sigma_mod, self.is_saturated
                            )
                            self.theta_hat.block_until_ready()
                            t_end_jax = time.perf_counter()
                            jax_dt = t_end_jax - t_start_jax
                            if jax_dt > self.control_period:
                                self.get_logger().warn(f"[DEBUG] JAX Execution took {jax_dt*1000:.2f} ms at t={t:.2f}s!")
                                
                            phi_val = np.array(phi_out, dtype=np.float64)
                        
                        current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                        delta_int = (dt / 2.0) * (current_integrand + self.last_integrand)
                        if not self.freeze_int_xy:
                            self.integral_term[0:2] += delta_int[0:2]
                        if not self.freeze_int_z:
                            self.integral_term[2] += delta_int[2]
                        self.last_integrand = current_integrand
                        u = phi_val + (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                        
                    case "integrated_resnet":
                        if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                            u_last =  (self.K_P * e) + (self.K_D * e_dot) + self.integral_term
                            kappa_vec = jnp.array(np.concatenate((q, q_dot, qd, qd_dot, u_last)))
                            
                            t_start_jax = time.perf_counter()
                            self.theta_hat, phi_out = self.compiled_update_step(
                                self.theta_hat, kappa_vec, jnp.array(r1), dt, self.theta_bar, self.gamma_diag, self.sigma_mod, self.is_saturated
                            )
                            self.theta_hat.block_until_ready()
                            t_end_jax = time.perf_counter()
                            jax_dt = t_end_jax - t_start_jax
                            if jax_dt > self.control_period:
                                self.get_logger().warn(f"[CRITICAL] JAX Execution took {jax_dt*1000:.2f} ms at t={t:.2f}s!")
                                
                            phi_val = np.array(phi_out, dtype=np.float64)
                        
                        current_integrand = (self.K_I * e) + (self.K_RISE * np.sign(r1)) + phi_val
                        delta_int = (dt / 2.0) * (current_integrand + self.last_integrand)
                        if not self.freeze_int_xy:
                            self.integral_term[0:2] += delta_int[0:2]
                        if not self.freeze_int_z:
                            self.integral_term[2] += delta_int[2]
                        self.last_integrand = current_integrand
                        u = (self.K_P * e) + (self.K_D * e_dot) + self.integral_term

                    case "supertwisting":
                        norm_r1 = np.linalg.norm(r1)
                        sgn_r1 = np.sign(r1)
                        self.st_integral += sgn_r1 * dt
                        u = self.k_2 * np.sqrt(norm_r1) * sgn_r1 + self.k_3 * self.st_integral + self.k_1 * e_dot
        
                norm_e = float(np.linalg.norm(e))
                norm_u = float(np.linalg.norm(u))
                
                self.time_history.append(t)
                self.error_norm_history.append(norm_e)
                self.q_history.append(q.tolist())
                self.qd_history.append(qd.tolist())

                if self.controller_type in ["resnet", "integrated_resnet"]:
                    self.weight_history.append(np.array(self.theta_hat).flatten().tolist())

                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                    current_error_sq = float(norm_e ** 2)
                    current_u_sq = float(norm_u ** 2)
                    current_cost_integrand = (t * self.q_e * (norm_e ** 2)) + (self.r_u * (norm_u ** 2))

                    if not self.cost_started:
                        # Seed the history at exact start to prevent trapezoidal integration jump
                        self.last_error_sq = current_error_sq
                        self.last_u_sq = current_u_sq
                        self.last_cost_integrand = current_cost_integrand
                        self.cost_started = True
                    
                    self.error_sq_integral += (dt / 2.0) * (current_error_sq + self.last_error_sq)
                    self.last_error_sq = current_error_sq
                    
                    self.u_sq_integral += (dt / 2.0) * (current_u_sq + self.last_u_sq)
                    self.last_u_sq = current_u_sq

                    self.cost_J += (dt / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                    self.last_cost_integrand = current_cost_integrand
                    
                    self.last_t = t
                
                self.is_saturated = False
                self.freeze_int_xy = False
                self.freeze_int_z = False
                
                u_xy = u[0:2]
                norm_uxy = float(np.linalg.norm(u_xy))
                if norm_uxy > self.acc_hor_max:
                    u[0:2] = u_xy * (self.acc_hor_max / norm_uxy)
                    self.is_saturated = True
                    if np.dot(e[0:2], u[0:2]) > 0.0:
                        self.freeze_int_xy = True
                    self.get_logger().debug(f"[DEBUG] XY SATURATION at t={t:.2f}s!")
                    
                if abs(u[2]) > self.acc_vert_max:
                    u[2] = self.acc_vert_max * np.sign(u[2])
                    self.is_saturated = True
                    if np.sign(e[2]) == np.sign(u[2]):
                        self.freeze_int_z = True
                    self.get_logger().debug(f"[DEBUG] Z SATURATION at t={t:.2f}s!")

                self.publish_trajectory_setpoint_acceleration(u[0], u[1], u[2])
                
                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ and t >= self.t_f:
                    rms_error = math.sqrt(self.error_sq_integral / self.t_f) if self.t_f > 0 else 0.0
                    rms_u = math.sqrt(self.u_sq_integral / self.t_f) if self.t_f > 0 else 0.0
                    self.get_logger().info(f"[RESULT] ITAE_COST = {self.cost_J:.2f}")
                    self.get_logger().info(f"[RESULT] RMS_ERROR = {rms_error:.4f}")
                    self.get_logger().info(f"[RESULT] RMS_CONTROL_EFFORT = {rms_u:.3f}")
                    self.trigger_failsafe_land()

def main(args=None):
    rclpy.init(args=args)
    node = AviaryRiseNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass  
    finally:
        try:
            if rclpy.ok() and not node.is_gazebo: 
                node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)
        except Exception: pass 
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()