import os
import math
import time
import csv
from functools import partial
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

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
jax.config.update("jax_platform_name", "cpu") # Force use of CPU since quad has no GPU
jax.config.update("jax_enable_x64", True) # Use 64 bit since all floats to be used are doubles; otherwise XLA recompilation will occur mid-flight
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from jax_resnet import resnet_network
from proj import discrete_projection
from desired_trajectory import TrajectoryGenerator

class ExperimentState:
    STATE_INIT: int = 0
    STATE_TAKEOFF: int = 1
    STATE_FOLLOW_TRAJ: int = 2
    STATE_PAUSED: int = 3
    STATE_FAILSAFE: int = 99

class CriticalHardwareError(Exception):
    pass

class OdomTimeoutError(Exception):
    pass

class FailsafeTriggeredError(Exception):
    pass

class BoundaryBreachError(Exception):
    pass

class AviaryRiseNode(Node):
    def __init__(self) -> None:
        super().__init__(node_name='aviary_rise_node')

        # Basic Parameters of the Experiment
        self.declare_parameter(name='is_gazebo', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='desired_trajectory', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter(name='vehicle_name', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='controller_type', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='control_frequency_hz', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='save_data', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='run_length_s', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='init_tol_m', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='d_out', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.is_gazebo: bool = self.get_parameter(name='is_gazebo').value
        self.desired_trajectory: int = self.get_parameter(name='desired_trajectory').value
        self.vehicle_name: str = self.get_parameter(name='vehicle_name').value
        self.controller_type: str = self.get_parameter(name='controller_type').value
        control_frequency_hz: float = self.get_parameter(name='control_frequency_hz').value
        self.save_data: bool = self.get_parameter(name='save_data').value
        self.run_length_s: float = self.get_parameter(name='run_length_s').value
        self.init_tol_m: float = self.get_parameter(name='init_tol_m').value
        self.d_out: int = self.get_parameter(name='d_out').value

        # Desired Trajectory
        if self.desired_trajectory not in [1,2]: 
            raise ValueError("INVALID DESIRED TRAJECTORY SELECTED.")
        # Fix: Convert parameters to primitive values for config
        self.config: Dict[str, Any] = {k: v.value for k, v in self.get_parameters_by_prefix(prefix='').items()}
        self.traj_gen: TrajectoryGenerator = TrajectoryGenerator(config=self.config)
        
        # Safety
        self.declare_parameter(name='mpc_acc_hor_max_mps2', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='mpc_acc_vert_max_mps2', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='safe_x_min_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='safe_x_max_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='safe_y_min_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='safe_y_max_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='safe_z_min_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='safe_z_max_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='odom_timeout_s', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='init_z_m_ned_aviary', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='odom_watchdog_freq', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.acc_hor_max_mps2: float = self.get_parameter(name='mpc_acc_hor_max_mps2').value
        self.acc_vert_max_mps2: float = self.get_parameter(name='mpc_acc_vert_max_mps2').value
        self.safe_x_min_m: float = self.get_parameter(name='safe_x_min_m_ned_aviary').value
        self.safe_x_max_m: float = self.get_parameter(name='safe_x_max_m_ned_aviary').value
        self.safe_y_min_m: float = self.get_parameter(name='safe_y_min_m_ned_aviary').value
        self.safe_y_max_m: float = self.get_parameter(name='safe_y_max_m_ned_aviary').value
        self.safe_z_min_m: float = self.get_parameter(name='safe_z_min_m_ned_aviary').value
        self.safe_z_max_m: float = self.get_parameter(name='safe_z_max_m_ned_aviary').value
        self.odom_timeout_s: float = self.get_parameter(name='odom_timeout_s').value
        self.init_z_m_ned_aviary: float = self.get_parameter(name='init_z_m_ned_aviary').value
        self.odom_watchdog_freq: float = self.get_parameter(name='odom_watchdog_freq').value
        
        # Cost Function
        self.declare_parameter(name='q_e', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='r_u', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(name='w_fail', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.q_e: float = self.get_parameter(name='q_e').value
        self.r_u: float = self.get_parameter(name='r_u').value
        self.w_fail: float = self.get_parameter(name='w_fail').value

        if self.controller_type == "pid":
            self.declare_parameter(name='K_P', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter(name='K_I', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter(name='K_D', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.K_P: float = self.get_parameter(name='K_P').value
            self.K_I: float = self.get_parameter(name='K_I').value
            self.K_D: float = self.get_parameter(name='K_D').value
    
        elif self.controller_type in ['baseline', 'integrated_resnet', 'resnet', 'supertwisting']:
            self.declare_parameter(name='k_1', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter(name='k_2', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter(name='k_3', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.declare_parameter(name='k_rise', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
            self.k_1: float = self.get_parameter(name='k_1').value
            self.k_2: float = self.get_parameter(name='k_2').value
            self.k_3: float = self.get_parameter(name='k_3').value
            self.K_RISE: float = self.get_parameter(name='k_rise').value
            self.K_P: float = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
            self.K_I: float = (self.k_1 * self.k_2 * self.k_3) + self.k_1
            self.K_D: float = self.k_1 + self.k_2 + self.k_3

            if self.controller_type in ["resnet", "integrated_resnet"]:
                self.declare_parameter(name='d_in', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
                self.declare_parameter(name='gamma', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
                self.declare_parameter(name='sigma_mod', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
                self.declare_parameter(name='theta_bar', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
                self.d_in: int = self.get_parameter(name='d_in').value
                
                self.declare_parameter(name='initial_weights', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
                self.declare_parameter(name='hidden_width', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
                self.declare_parameter(name='k_0', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
                self.declare_parameter(name='k_i', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
                self.declare_parameter(name='num_blocks', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
                self.declare_parameter(name='h_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
                self.declare_parameter(name='o_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
                self.declare_parameter(name='shortcut_act_func', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))                
                self.theta_hat: jax.Array = jnp.array(object=self.get_parameter(name='initial_weights').value)
                
                self.gamma_diag: jax.Array = jnp.ones(shape=self.theta_hat.shape[0]) * self.get_parameter(name='gamma').value
                self.sigma_mod: float = self.get_parameter(name='sigma_mod').value
                self.theta_bar: float = self.get_parameter(name='theta_bar').value

                self.bound_resnet = jax.jit(fun=partial(
                    resnet_network,
                    d_in=self.d_in,
                    hidden_width=self.get_parameter(name='hidden_width').value,
                    d_out=self.d_out,
                    b=self.get_parameter(name='num_blocks').value,
                    k_0=self.get_parameter(name='k_0').value,
                    k_i=self.get_parameter(name='k_i').value,
                    h_act_func=self.get_parameter(name='h_act_func').value,
                    o_act_func=self.get_parameter(name='o_act_func').value,
                    shortcut_act_func=self.get_parameter(name='shortcut_act_func').value,
                ))
            
                @jax.jit
                def compiled_update_step(theta_hat: jax.Array, x_vec: jax.Array, r1_vec: jax.Array, dt: float, theta_bar: float, gamma_diag: jax.Array, s_mod: float, saturated: bool) -> Tuple[jax.Array, jax.Array]:
                    phi_val, vjp_fn = jax.vjp(fun=lambda t: self.bound_resnet(t, x_vec), has_aux=False, *[theta_hat])
                    grad_term = vjp_fn(r1_vec)[0]
                    theta_dot_unprojected = gamma_diag * (grad_term - s_mod * theta_hat)
                    theta_next = discrete_projection(theta=theta_hat, theta_dot=theta_dot_unprojected, dt=dt, theta_bar=theta_bar, gamma_diag=gamma_diag)
                    final_theta = jax.lax.select(pred=saturated, on_true=theta_hat, on_false=theta_next)
                    return final_theta, phi_val
                    
                self.compiled_update_step = compiled_update_step
                self.precompile_jax()
        
        # For VehicleStatus callback
        self.nav_state: int = 0
        self.vehicle_system_id: int = 1
        self.vehicle_component_id: int = 1

        # Init
        self.is_armed: bool = False
        self.in_offboard_mode: bool = False
        self.landing_command_sent: bool = False  
        self.cost_started: bool = False
        self.is_saturated: bool = False
        self.freeze_int_xy: bool = False
        self.freeze_int_z: bool = False
        self.initial_position_locked: bool = False
        self.latest_odom: Optional[VehicleOdometry] = None

        self.last_odom_ros_time: float = 0.0
        self.start_x: float = 0.0
        self.start_y: float = 0.0
        self.experiment_state: int = ExperimentState.STATE_INIT
        self.t_0: float = 0.0

        self.last_t: float = 0.0
        self.last_auto_cmd_time: float = 0.0
        self.pause_start_time: float = 0.0
        self.pre_pause_state: int = ExperimentState.STATE_INIT
        
        self.ticks_without_odom: int = 0
        self.current_integral_control_term: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        self.last_control_integrand: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        self.st_integral: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        self.cost_J: float = 0.0
        self.last_cost_integrand: float = 0.0
        self.error_sq_integral: float = 0.0
        self.last_error_sq: float = 0.0
        self.u_sq_integral: float = 0.0
        self.last_u_sq: float = 0.0
        self.time_history: List[float] = []
        self.control_output_norm_history: List[float] = []
        self.error_norm_history: List[float] = []
        self.weight_history: List[List[float]] = []
        self.q_history: List[List[float]] = []
        self.qd_history: List[List[float]] = []

        qos_profile: QoSProfile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            msg_type=OffboardControlMode, topic=f'/{self.vehicle_name}/fmu/in/offboard_control_mode', qos_profile=qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            msg_type=TrajectorySetpoint, topic=f'/{self.vehicle_name}/fmu/in/trajectory_setpoint', qos_profile=qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            msg_type=VehicleCommand, topic=f'/{self.vehicle_name}/fmu/in/vehicle_command', qos_profile=qos_profile)

        self.status_sub = self.create_subscription(
            msg_type=VehicleStatus, topic=f'/{self.vehicle_name}/fmu/out/vehicle_status', callback=self.vehicle_status_callback, qos_profile=qos_profile)
        self.odom_sub = self.create_subscription(
            msg_type=VehicleOdometry, topic=f'/{self.vehicle_name}/fmu/out/vehicle_odometry', callback=self.odom_callback, qos_profile=qos_profile)
        
        self.control_period: float = 1.0 / control_frequency_hz
        self.control_timer = self.create_timer(timer_period_sec=self.control_period, callback=self.control_timer_callback)

        self.odom_watchdog_timer = self.create_timer(timer_period_sec=1.0/self.odom_watchdog_freq, callback=self.odom_watchdog_callback)
        
        self.get_logger().info(msg=f"Node Booted. Controller: {self.controller_type.upper()} | Trajectory: {self.desired_trajectory} | Gazebo Mode: {self.is_gazebo}")

    def precompile_jax(self) -> None:
        dummy_x: jax.Array = jnp.zeros(shape=self.d_in)
        dummy_r1: jax.Array = jnp.zeros(shape=self.d_out)
        self.get_logger().info(msg="[JAX] Compiling XLA Graph on CPU...")
        
        self.theta_hat, _ = self.compiled_update_step(
            theta_hat=self.theta_hat,
            x_vec=dummy_x,
            r1_vec=dummy_r1,
            dt=self.control_period,
            theta_bar=self.theta_bar,
            gamma_diag=self.gamma_diag,
            s_mod=self.sigma_mod,
            saturated=False # I've never tested True...
        )
        self.theta_hat.block_until_ready()
        
        start_time: float = time.perf_counter()
        self.theta_hat, _ = self.compiled_update_step(
            theta_hat=self.theta_hat,
            x_vec=dummy_x,
            r1_vec=dummy_r1,
            dt=self.control_period,
            theta_bar=self.theta_bar,
            gamma_diag=self.gamma_diag,
            s_mod=self.sigma_mod,
            saturated=False
        )
        self.theta_hat.block_until_ready()
        hot_time: float = time.perf_counter() - start_time
        
        # Reset the weights back to true initial conditions
        self.theta_hat = jnp.array(object=self.get_parameter(name='initial_weights').value)
        self.theta_hat.block_until_ready()
        self.get_logger().info(msg=f"[JAX] Neural Network Latency: {hot_time*1000:.2f} ms")
        if hot_time > self.control_period:
            self.get_logger().fatal(msg=f"[ERROR] Execution time {hot_time}s exceeds {self.control_period}s limit!")
            raise CriticalHardwareError("[JAX] ResNet latency too high for selected control frequency (init).")

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
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

    def odom_watchdog_callback(self) -> None:
        self.ticks_without_odom += 1
        
        if not self.initial_position_locked:
            if self.ticks_without_odom >= (self.odom_timeout_s * self.odom_watchdog_freq):
                raise OdomTimeoutError("NO ODOMETRY AT BOOT! EXITING NODE.")
        else:
            if self.is_gazebo:
                if self.ticks_without_odom >= (self.odom_timeout_s * self.odom_watchdog_freq):
                    raise OdomTimeoutError("YOUR PC IS RUNNING BEHIND SCHEDULE! EXITING SIM.")
            else:
                # Use original wall clock logic for real vehicle (sim-to-real)
                current_time: float = self.get_clock().now().nanoseconds / 1e9
                if (current_time - self.last_odom_ros_time) > self.odom_timeout_s:
                    self.trigger_failsafe_land()
                    raise OdomTimeoutError("ODOM LOST! TRIGGERING FAILSAFE.")

    def publish_vehicle_command(self, command: int, param1: float, param2: float) -> None:
        self.get_logger().info(msg=f"[DEBUG] Publishing command {command}")
        msg: VehicleCommand = VehicleCommand()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.vehicle_system_id
        msg.target_component = self.vehicle_component_id
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg=msg)

    def publish_offboard_heartbeat(self) -> None:
        msg: OffboardControlMode = OffboardControlMode()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = True
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_publisher.publish(msg=msg)

    def publish_trajectory_setpoint_acceleration(self, ax: float, ay: float, az: float) -> None:
        msg: TrajectorySetpoint = TrajectorySetpoint()
        msg.acceleration = [ax, ay, az]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = 0.0  # Command a heading of 0.0 always
        if self.latest_odom is not None:
            msg.timestamp = self.latest_odom.timestamp
        self.trajectory_setpoint_publisher.publish(msg=msg)

    def log_csv(self) -> None:
        traj_name: str = ""
        match self.desired_trajectory:
            case 1:
                traj_name = "figure_eight"
            case 2:
                traj_name = "rose"

        base_dir: str = f"/home/root/plot_data/{self.controller_type}/{traj_name}"
        os.makedirs(name=base_dir, exist_ok=True)
        
        existing_files: List[str] = [f for f in os.listdir(path=base_dir) if os.path.isfile(path=os.path.join(base_dir, f))]
        iterable: int = len(existing_files) + 1
        csv_filename: str = os.path.join(base_dir, f"run_{iterable}.csv")
        try:
            with open(file=csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                headers: List[str] = ["Time_s", "Error_Norm_m", "Control_Output_Norm_mps2", "x", "y", "z", "xd", "yd", "zd"]
                if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                    num_weights: int = len(self.weight_history[0])
                    headers += [f"W{i}" for i in range(num_weights)]
                writer.writerow(headers)
                for i in range(len(self.time_history)):
                    row: List[float] = [
                        self.time_history[i], self.error_norm_history[i], self.control_output_norm_history[i],
                        self.q_history[i][0], self.q_history[i][1], self.q_history[i][2],
                        self.qd_history[i][0], self.qd_history[i][1], self.qd_history[i][2]
                    ]
                    if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                        row += self.weight_history[i]
                    writer.writerow(row)
            self.get_logger().info(msg=f"Telemetry saved to {csv_filename}")
        except Exception as e:
            self.get_logger().error(msg=f"Failed to write CSV: {e}")

    def trigger_failsafe_land(self) -> None:
        if self.save_data:
            self.log_csv()

        if self.is_gazebo:
            raise FailsafeTriggeredError("GAZEBO FAILSAFE TRIGGERED. EXITING SIM.")
        else:
            self.get_logger().error(msg="LANDING!")
            self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_NAV_LAND, param1=0.0, param2=0.0)
            self.experiment_state = ExperimentState.STATE_FAILSAFE

    def check_safety_boundary(self, q: np.ndarray) -> bool:
        if not (self.safe_x_min_m <= q[0] <= self.safe_x_max_m):
            self.get_logger().fatal(msg=f"X BOUNDARY BREACH: {q[0]} not in [{self.safe_x_min_m}, {self.safe_x_max_m}]")
            return True 
        if not (self.safe_y_min_m <= q[1] <= self.safe_y_max_m):
            self.get_logger().fatal(msg=f"Y BOUNDARY BREACH: {q[1]} not in [{self.safe_y_min_m}, {self.safe_y_max_m}]")
            return True 
        if not (self.safe_z_min_m <= q[2] <= self.safe_z_max_m):
            self.get_logger().fatal(msg=f"Z BOUNDARY BREACH: {q[2]} not in [{self.safe_z_min_m}, {self.safe_z_max_m}]")
            return True
        return False

    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.experiment_state == ExperimentState.STATE_TAKEOFF:
            # During takeoff, hold exactly above where it initialized
            return (np.array(object=[self.start_x, self.start_y, self.init_z_m_ned_aviary], dtype=np.float64), 
                    np.zeros(shape=3, dtype=np.float64), np.zeros(shape=3, dtype=np.float64))
            
        return self.traj_gen.get_desired_state(t=t)

    def control_timer_callback(self) -> None:
        if self.latest_odom is None: return
        current_timestamp_s: float = self.latest_odom.timestamp / 1e6

        match self.experiment_state:
            case ExperimentState.STATE_FAILSAFE:
                if not self.landing_command_sent:
                    self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_NAV_LAND, param1=0.0, param2=0.0)
                    self.landing_command_sent = True
                return

            case ExperimentState.STATE_INIT:
                self.landing_command_sent = False
                self.cost_started = False
                
                if not self.in_offboard_mode: # Haven't started or Joe dropped me out of off-board
                    self.get_logger().info(msg="Waiting for PX4 Offboard Mode switch engagement...", throttle_duration_sec=2.0)
                    self.publish_offboard_heartbeat()
                    self.publish_trajectory_setpoint_acceleration(ax=0.0, ay=0.0, az=0.0)
                    
                    if self.is_gazebo:
                        # Throttle automatic MAVLink commands to 1 Hz
                        if current_timestamp_s - self.last_auto_cmd_time > 1.0:
                            if not self.in_offboard_mode:
                                self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0) # 6.0 requests offboard
                            if not self.is_armed:
                                self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
                            self.last_auto_cmd_time = current_timestamp_s
                return

                if self.is_armed:
                    self.get_logger().info(msg=f"ARMED & OFFBOARD validated. Initializing Takeoff to Z={self.init_z_m_ned_aviary}.")
                    self.current_integral_control_term = np.zeros(shape=self.d_out, dtype=np.float64)
                    self.last_control_integrand = np.zeros(shape=self.d_out, dtype=np.float64)
                    self.st_integral = np.zeros(shape=self.d_out, dtype=np.float64)
                    self.experiment_state = ExperimentState.STATE_TAKEOFF

            case ExperimentState.STATE_PAUSED:
                if self.in_offboard_mode and self.is_armed:
                    # Pilot re-engaged offboard mode. 
                    # We shift t_0 forward by the elapsed paused time so the trajectory completely froze during the dropout
                    time_paused: float = current_timestamp_s - self.pause_start_time
                    self.t_0 += time_paused  
                    self.experiment_state = self.pre_pause_state
                    self.get_logger().info(msg="Offboard Mode re-engaged! Resuming trajectory seamlessly.")
                else:
                    self.get_logger().info(msg="Trajectory Paused. Waiting for Pilot to re-engage Offboard...", throttle_duration_sec=2.0)
                return

            case ExperimentState.STATE_FOLLOW_TRAJ | ExperimentState.STATE_TAKEOFF:
                self.publish_offboard_heartbeat()

                if not self.in_offboard_mode:
                    if self.is_gazebo:
                        raise FailsafeTriggeredError("PX4 LEFT OFFBOARD MODE IN SITL! FAILING TRIAL.")
                    else:
                        self.get_logger().warn(msg="RC Pilot Intervention Detected! Pausing trajectory.", throttle_duration_sec=1.0)
                        self.pre_pause_state = self.experiment_state
                        self.experiment_state = ExperimentState.STATE_PAUSED
                        self.pause_start_time = current_timestamp_s
                        
                        # Reset memory integrals so they don't explosively un-wind when re-engaged
                        self.current_integral_control_term = np.zeros(shape=self.d_out, dtype=np.float64)
                        self.last_control_integrand = np.zeros(shape=self.d_out, dtype=np.float64)
                        self.st_integral = np.zeros(shape=self.d_out, dtype=np.float64)
                        return
                
                # Check transitions before updating the clock if in TAKEOFF
                q: np.ndarray = np.array(object=self.latest_odom.position, dtype=np.float64)
                if self.experiment_state == ExperimentState.STATE_TAKEOFF:
                    e_takeoff: np.ndarray = np.array(object=[self.start_x, self.start_y, self.init_z_m_ned_aviary], dtype=np.float64) - q
                    if np.linalg.norm(x=e_takeoff) <= self.init_tol_m:
                        self.experiment_state = ExperimentState.STATE_FOLLOW_TRAJ
                        # Reset t_0 so the trajectory clock starts at exactly 0.0 now
                        self.t_0 = current_timestamp_s
                        self.last_t = 0.0
                        self.get_logger().info(msg=f"TAKEOFF SETTLED. Step Response Triggered: Starting Trajectory {self.desired_trajectory}.")

                # If in TAKEOFF, t_0 hasn't been set to the trajectory clock yet, so t evaluates to arbitrary.
                # However get_desired_state(t) strictly ignores t during TAKEOFF.
                t: float = current_timestamp_s - self.t_0 if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ else 0.0
                dt: float = t - self.last_t if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ else self.control_period
                
                q_dot: np.ndarray = np.array(object=self.latest_odom.velocity, dtype=np.float64)
                
                if self.check_safety_boundary(q=q):
                    if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                        self.cost_J += self.w_fail * ((self.run_length_s - t) ** 2)
                        self.get_logger().error(msg=f"[RESULT] COST = {self.cost_J:.4f} (BOUNDARY FAILURE)")
                    self.trigger_failsafe_land()
                    raise BoundaryBreachError(f"Boundary breached at position: {q}")

                qd: np.ndarray
                qd_dot: np.ndarray
                qd_ddot: np.ndarray
                qd, qd_dot, qd_ddot = self.get_desired_state(t=t)
                e: np.ndarray = qd - q
                e_dot: np.ndarray = qd_dot - q_dot
                r1: np.ndarray = e_dot + (self.k_1 * e)

                u: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
                phi_val: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
                
                match self.controller_type:
                    case "baseline" | "pid":
                        current_integrand: np.ndarray = (self.K_I * e) + (self.controller_type == "baseline") * (self.K_RISE * np.sign(x=r1))
                        delta_int: np.ndarray = (dt / 2.0) * (current_integrand + self.last_control_integrand)
                        if not self.freeze_int_xy:
                            self.current_integral_control_term[0:2] += delta_int[0:2]
                        if not self.freeze_int_z:
                            self.current_integral_control_term[2] += delta_int[2]
                        self.last_control_integrand = current_integrand
                        u = (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term
                        
                    case "resnet":
                        if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ: 
                            x_vec: jax.Array = jnp.array(object=np.concatenate((q, q_dot, qd, qd_dot)))
                            
                            t_start_jax: float = time.perf_counter()
                            self.theta_hat, phi_out = self.compiled_update_step(
                                theta_hat=self.theta_hat,
                                x_vec=x_vec, 
                                r1_vec=jnp.array(object=r1),
                                dt=dt,
                                theta_bar=self.theta_bar,
                                gamma_diag=self.gamma_diag,
                                s_mod=self.sigma_mod,
                                saturated=self.is_saturated
                            )
                            self.theta_hat.block_until_ready()
                            t_end_jax: float = time.perf_counter()
                            jax_dt: float = t_end_jax - t_start_jax
                            if jax_dt > self.control_period:
                                self.get_logger().warn(msg=f"[DEBUG] JAX Execution took {jax_dt*1000:.2f} ms at t={t:.2f}s!")
                                
                            phi_val = np.array(object=phi_out, dtype=np.float64)
                        
                        current_integrand_res: np.ndarray = (self.K_I * e) + (self.K_RISE * np.sign(x=r1))
                        delta_int_res: np.ndarray = (dt / 2.0) * (current_integrand_res + self.last_control_integrand)
                        if not self.freeze_int_xy:
                            self.current_integral_control_term[0:2] += delta_int_res[0:2]
                        if not self.freeze_int_z:
                            self.current_integral_control_term[2] += delta_int_res[2]
                        self.last_control_integrand = current_integrand_res
                        u = phi_val + (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term
                        
                    case "integrated_resnet":
                        if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                            u_last: np.ndarray =  (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term
                            kappa_vec: jax.Array = jnp.array(object=np.concatenate((q, q_dot, qd, qd_dot, u_last)))
                            
                            t_start_jax = time.perf_counter()
                            self.theta_hat, phi_out = self.compiled_update_step(
                                theta_hat=self.theta_hat,
                                x_vec=kappa_vec,
                                r1_vec=jnp.array(object=r1),
                                dt=dt,
                                theta_bar=self.theta_bar,
                                gamma_diag=self.gamma_diag,
                                s_mod=self.sigma_mod,
                                saturated=self.is_saturated
                            )
                            self.theta_hat.block_until_ready()
                            t_end_jax = time.perf_counter()
                            jax_dt = t_end_jax - t_start_jax
                            if jax_dt > self.control_period:
                                self.get_logger().warn(msg=f"[CRITICAL] JAX Execution took {jax_dt*1000:.2f} ms at t={t:.2f}s!")
                                
                            phi_val = np.array(object=phi_out, dtype=np.float64)
                        
                        current_integrand_int: np.ndarray = (self.K_I * e) + (self.K_RISE * np.sign(x=r1)) + phi_val
                        delta_int_int: np.ndarray = (dt / 2.0) * (current_integrand_int + self.last_control_integrand)
                        if not self.freeze_int_xy:
                            self.current_integral_control_term[0:2] += delta_int_int[0:2]
                        if not self.freeze_int_z:
                            self.current_integral_control_term[2] += delta_int_int[2]
                        self.last_control_integrand = current_integrand_int
                        u = (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term

                    case "supertwisting":
                        norm_r1: float = float(np.linalg.norm(x=r1))
                        sgn_r1: np.ndarray = np.sign(x=r1)
                        self.st_integral += sgn_r1 * dt
                        u = qd_ddot + self.k_2 * np.sqrt(x=norm_r1) * sgn_r1 + self.k_3 * self.st_integral + self.k_1 * e_dot
        
                norm_e: float = float(np.linalg.norm(x=e))
                norm_u: float = float(np.linalg.norm(x=u))
                
                self.time_history.append(t)
                self.error_norm_history.append(norm_e)
                self.control_output_norm_history.append(norm_u)
                self.q_history.append(q.tolist())
                self.qd_history.append(qd.tolist())

                if self.controller_type in ["resnet", "integrated_resnet"]:
                    self.weight_history.append(np.array(object=self.theta_hat).flatten().tolist())

                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                    current_error_sq: float = float(norm_e ** 2)
                    current_u_sq: float = float(norm_u ** 2)
                    current_cost_integrand: float = (t * self.q_e * (norm_e ** 2)) + (self.r_u * (norm_u ** 2))

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
                
                u_xy: np.ndarray = u[0:2]
                norm_uxy: float = float(np.linalg.norm(x=u_xy))
                if norm_uxy > self.acc_hor_max_mps2:
                    u[0:2] = u_xy * (self.acc_hor_max_mps2 / norm_uxy)
                    self.is_saturated = True
                    if np.dot(a=e[0:2], b=u[0:2]) > 0.0:
                        self.freeze_int_xy = True
                    self.get_logger().debug(msg=f"[DEBUG] XY SATURATION at t={t:.2f}s!")
                    
                if abs(u[2]) > self.acc_vert_max_mps2:
                    u[2] = self.acc_vert_max_mps2 * np.sign(x=u[2])
                    self.is_saturated = True
                    if np.sign(x=e[2]) == np.sign(x=u[2]):
                        self.freeze_int_z = True
                    self.get_logger().debug(msg=f"[DEBUG] Z SATURATION at t={t:.2f}s!")

                self.publish_trajectory_setpoint_acceleration(ax=u[0], ay=u[1], az=u[2])
                
                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ and t >= self.run_length_s:
                    rms_error: float = math.sqrt(x=self.error_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    rms_u: float = math.sqrt(x=self.u_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    self.get_logger().info(msg=f"[RESULT] COST = {self.cost_J:.2f}")
                    self.get_logger().info(msg=f"[RESULT] RMS_ERROR = {rms_error:.4f}")
                    self.get_logger().info(msg=f"[RESULT] RMS_CONTROL_EFFORT = {rms_u:.3f}")
                    self.trigger_failsafe_land()

def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: AviaryRiseNode = AviaryRiseNode()
    try:
        rclpy.spin(node=node)
    except (SystemExit, KeyboardInterrupt):
        pass
    except ValueError as e:
        print(f"[FATAL] ValueError: {e}")
    except CriticalHardwareError as e:
        print(f"[FATAL] Hardware Error: {e}")
    except OdomTimeoutError as e:
        print(f"[FATAL] Odometry Error: {e}")
    except FailsafeTriggeredError as e:
        print(f"[FATAL] Failsafe Triggered: {e}")
    except BoundaryBreachError as e:
        print(f"[FATAL] Boundary Breach: {e}")
    finally:
        try:
            if node.save_data:
                node.get_logger().info(msg="Interrupt detected. Attempting to save data to CSV...")
                node.log_csv()

            if rclpy.ok(): 
                node.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_NAV_LAND, param1=0.0, param2=0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()
            print("[INFO] Node safely destroyed.")
        else:
            print("[FATAL] Node NOT safely destroyed.")

if __name__ == '__main__':
    main()