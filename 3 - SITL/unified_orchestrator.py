#!/usr/bin/env python3
import os
import subprocess
import time
import re
import argparse
import optuna
import sys
import yaml
from functools import partial

# Ensure JAX stays on CPU
import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ros2_ws", "src", "aviary_rise_controller", "aviary_rise_controller")))
from jax_resnet import get_total_parameters, init_resnet_weights

SEED = 42

FIXED_X = 0.70
FIXED_Y = -2.37
TARGET_Z = -3.0

# Fixed Phase 2 Baseline Gains (Optimized from Phase 1)
FIXED_K1 = 1.31
FIXED_K2 = 0.131
FIXED_K3 = 0.83
FIXED_K_RISE = 0.0287

PATIENCE_TRIALS = 200
NUM_TRIALS = 500
RETRY_ATTEMPTS = 2

class PatienceCallback:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_value = float('inf')
        self.wait_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        current_value = trial.value
        
        # Ignore failed runs (e.g., your 1e6 penalty or boundary failures) 
        # so they don't incorrectly reset our tracking
        if current_value is None or current_value >= 1e5:
            return

        # Check for improvement
        if current_value < self.best_value:
            self.best_value = current_value
            self.wait_count = 0  # Reset the clock!
        else:
            self.wait_count += 1

        # Trigger the early stop
        if self.wait_count >= self.patience:
            print(f"\n==================================")
            print(f"[*] CONVERGENCE REACHED")
            print(f"[*] No improvement in {self.patience} consecutive valid trials.")
            print(f"[*] Terminating optimization early to save compute.")
            print(f"==================================\n")
            study.stop()

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Optuna Orchestrator for Baseline and ResNet Controllers")
    parser.add_argument(
        "--controller_type", 
        type=str, 
        required=True,
        choices=["baseline", "developed", "noresnet"],
        help="Specify which controller profile to optimize."
    )
    parser.add_argument(
        "--desired_trajectory",
        type=int,
        required=True,
        choices=[1, 2],
        help="Specify which trajectory to run (1: Warped Figure-Eight, 2: Rose Curve)."
    )
    parser.add_argument(
        "--wind",
        action='store_true',
        help="Use the windy world (no fans; global only)"
    )
    args = parser.parse_args()
    
    if args.controller_type is None:
        parser.error("--controller_type is required if CONTROLLER_TYPE env var is not set.")
        
    return args

def get_storage_config(controller_type: str, wind: bool):
    if controller_type == "noresnet":
        return "sqlite:///phase1_tuning.db", "phase1_noresnet_baseline_tuning"
    else:
        controller_type_augmented = controller_type
        if wind: controller_type_augmented += "_wind"
        return f"sqlite:///phase2_{controller_type_augmented}.db", f"phase2_{controller_type_augmented}_optimization"

def write_spawn_location(x: float, y: float) -> None:
    env_path = "spawn-locations.env"
    env_content = f"""CONTAINER_NAME="px4-sitl-gz"\nN_TB=0\nTB_SPAWN_LOCATIONS=()\nN_QUAD=1\nQUAD1_LOCATION="{x},{y},0.0"\nQUAD_SPAWN_LOCATIONS=($QUAD1_LOCATION)\n"""
    with open(env_path, "w") as f:
        f.write(env_content)

def cleanup_environment() -> None:
    print("[*] Purging environment...")
    subprocess.run(["./shutdown-background-services.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def wait_for_sim_ready() -> bool:
    print("[*] Polling for PX4 EKF2 and DDS bridge to initialize...")
    for attempt in range(40): 
        cmd = 'docker exec px4-sitl-gz bash -c "source /home/root/ros-sources.sh && timeout 2 ros2 topic echo --once /px4_1/fmu/out/vehicle_odometry"'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            print("[*] EKF2 converged! Odometry is publishing.")
            time.sleep(1) 
            return True
        time.sleep(1)
    return False

def write_ros_params(param_dict: dict) -> str:
    """Dynamically generates the params.yaml file based on the selected controller type."""
    param_dir = os.path.expanduser("~/voxl-px4-sitl/ros2_ws/src/aviary_rise_controller/param")
    os.makedirs(param_dir, exist_ok=True)
    param_file = os.path.join(param_dir, "params.yaml")

    params = {
        'aviary_rise_node': {
            'ros__parameters': param_dict
        }
    }

    with open(param_file, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    return "/home/root/ros2_ws/src/aviary_rise_controller/param/params.yaml"

def execute_single_run(param_dict: dict, desired_trajectory: int, wind: bool):
    cleanup_environment()
    
    container_param_path = write_ros_params(param_dict)
    write_spawn_location(FIXED_X, FIXED_Y)
    
    # 1. Define the target world file with the .sdf extension
    if wind:
        world_file = "mocap_fig8.sdf" if desired_trajectory == 1 else "mocap_rose.sdf"
    else:
        world_file = "default.sdf"

    # 2. Inject it into a copy of the current environment variables
    run_env = os.environ.copy()
    run_env["PX4_GZ_WORLD"] = world_file
    run_env["GZ_SEED"] = str(SEED)
    
    # 3. Pass the modified environment to the spawn script
    subprocess.run(["./spawn-sim-env.sh"], env=run_env, stdout=subprocess.DEVNULL)
    
    if not wait_for_sim_ready():
        print("[!] Polling timeout. Simulator failed to boot properly.")
        cleanup_environment()
        return None
    
    ros_cmd = (
        f"source /home/root/ros-sources.sh && "
        f"cd /home/root/ros2_ws && source install/setup.bash && "
        f"ros2 run aviary_rise_controller aviary_rise_controller "
        f"--ros-args --params-file {container_param_path}"
    )
    
    process = subprocess.Popen(
        ["docker", "exec", "px4-sitl-gz", "bash", "-c", ros_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    final_cost = None
    rms_error = None
    rms_control = None
    for line in process.stdout:
        print(f"  [ROS] {line.strip()}")
        if match := re.search(r"\[RESULT\] ITAE_COST = ([\d\.]+)", line):
            final_cost = float(match.group(1))
        elif match := re.search(r"\[RESULT\] RMS_ERROR = ([\d\.]+)", line):
            rms_error = float(match.group(1))
        elif match := re.search(r"\[RESULT\] RMS_CONTROL_EFFORT = ([\d\.]+)", line):
            rms_control = float(match.group(1))
            
    process.wait()
    cleanup_environment()
    return final_cost, rms_error, rms_control

def run_trial(param_dict: dict, desired_trajectory: int, wind: bool):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        if attempt != 1:
            print(f"\n[*] Execution Attempt {attempt}/{RETRY_ATTEMPTS}")
        
        cost, rms_err, rms_ctrl = execute_single_run(param_dict, desired_trajectory, wind)
        
        if cost is not None:
            return cost, rms_err, rms_ctrl
            
        print("[!] Trial returned None (System/SITL failure). Retrying...")
        time.sleep(5)
        
    print(f"[!] All {RETRY_ATTEMPTS} attempts failed. Assigning ultimate penalty.")
    return 1e6, None, None

def get_base_param_dict(controller_type: str, desired_trajectory: int):
    return {
        'is_gazebo': True,
        'desired_trajectory': desired_trajectory,
        'd_out': 3,
        
        # Global Limits & Timers
        'mpc_acc_hor_max': 6.0,
        'mpc_acc_vert_max': 3.0,
        'safe_x_max': 5.0,
        'safe_y_max': 11.5,
        'safe_z_max': 0.5,
        'safe_z_min': -6.5,
        'odom_timeout_sec': 15.0,
        'settle_ticks': 50,
        'hover_tolerance': 0.2,
        'windup_time_sec': 5.0,
        'watchdog_freq': 10.0,

        # Trajectory 1
        'traj1_period': 30.0,
        'traj1_alpha_warp': 0.5,
        'traj1_x_amp': 1.5,
        'traj1_y_amp': 3.0,
        'traj1_center_z': -0.75,
        'traj1_z_amp': 0.25,

        # Trajectory 2
        'traj2_target_speed': 1.0,
        'traj2_petal_radius': 2.5,
        'traj2_center_z': -0.5,

        'vehicle_name': 'px4_1',
        'controller_type': controller_type,
        'control_frequency': 50.0,
        'plot': False,
        'sim_time': 60.0,
        
        'q_e': 1.0,
        'r_u': 0.2,
        'w_fail': 1000.0,
    }

def objective(trial: optuna.Trial, controller_type: str, desired_trajectory: int, wind: bool) -> float:
    param_dict = get_base_param_dict(controller_type, desired_trajectory)
    
    print(f"\n==============================================")
    print(f"[*] Starting trial")
    print(f"Controller: {controller_type} | Trajectory: {desired_trajectory} | Wind: {wind}")

    if controller_type == "noresnet":
        # Phase 1 Search Space
        param_dict['k1'] = trial.suggest_float("k1", 0.01, 2.0, log=True)
        param_dict['k2'] = trial.suggest_float("k2", 0.01, 5.0, log=True)
        param_dict['k3'] = trial.suggest_float("k3", param_dict['k2'], 5.0) 
        param_dict['k_rise'] = trial.suggest_float("k_rise", 0.01, 2.0, log=True)

        print(f"[*] Suggested Baseline Gains -> k1: {param_dict['k1']:.2f} | k2: {param_dict['k2']:.2f} | k3: {param_dict['k3']:.2f} | krise: {param_dict['k_rise']:.6f}")
        
    else:
        # Phase 2 Search Space (baseline or developed)
        param_dict['k1'] = FIXED_K1
        param_dict['k2'] = FIXED_K2
        param_dict['k3'] = FIXED_K3
        param_dict['k_rise'] = FIXED_K_RISE

        param_dict['h_act_func'] = 'swish'
        param_dict['o_act_func'] = 'tanh'
        param_dict['shortcut_act_func'] = 'swish'
        param_dict['theta_bar'] = 1e6 # Should never come into play
        
        param_dict['d_in'] = 12 if controller_type == "baseline" else 15

        initial_weight_scale_factor =  trial.suggest_categorical("initial_weight_scale_factor", [0.2, 1.0])
        param_dict['num_blocks'] = trial.suggest_categorical("num_blocks", [1, 2, 4])
        param_dict['k_0'] = trial.suggest_categorical("k_0", [1, 2, 4])
        param_dict['k_i'] = trial.suggest_categorical("k_i", [1, 2, 4])
        param_dict['hidden_width'] = trial.suggest_categorical("hidden_width", [4, 8, 16])
        param_dict['gamma'] = float(trial.suggest_float("gamma", 0.1, 10.0, log=True))
        param_dict['sigma_mod'] = float(trial.suggest_float("sigma_mod", 0.5, 5.0, log=True))

        total_params = get_total_parameters(param_dict['d_in'], param_dict['hidden_width'],  param_dict['d_out'], param_dict['num_blocks'], param_dict['k_0'], param_dict['k_i'])
        
        key = jax.random.PRNGKey(SEED)
        initial_weights_jax = initial_weight_scale_factor * init_resnet_weights(key,  param_dict['d_in'], param_dict['hidden_width'],  param_dict['d_out'], param_dict['num_blocks'], param_dict['k_0'], param_dict['k_i'], 'xavier', 'he')
        param_dict['initial_weights'] = [float(w) for w in initial_weights_jax]
        
        print(f"[*] Evaluating Architecture: {total_params} parameters.")
        print(f"[*] Gamma: {param_dict['gamma']:.2f} | Sigma Mod: {param_dict['sigma_mod']:.6f} | hidden_width: {param_dict['hidden_width']} | k_i: {param_dict['k_i']} | k_0: {param_dict['k_0']} | b: {param_dict['num_blocks']} | W_s: {initial_weight_scale_factor}")

    print(f"==============================================")
    
    cost, rms_err, rms_ctrl = run_trial(param_dict, desired_trajectory, wind)
    if rms_err is not None:
        trial.set_user_attr("rms_error", rms_err)
    if rms_ctrl is not None:
        trial.set_user_attr("rms_control_effort", rms_ctrl)
    
    return cost


if __name__ == "__main__":
    args = parse_args()
    db_name, study_name = get_storage_config(args.controller_type, args.wind)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_name,
        load_if_exists=True,
        direction="minimize"
    )
    
    print(f"[*] Starting Tuning for {args.controller_type} (Trajectory {args.desired_trajectory}, Wind {args.wind}). Saving to {db_name}")
    
    early_stopper = PatienceCallback(patience=PATIENCE_TRIALS)
    
    # Inject the new arg parse choice into the objective function
    bound_objective = partial(objective, controller_type=args.controller_type, desired_trajectory=args.desired_trajectory, wind=args.wind)
    
    study.optimize(
        bound_objective, 
        n_trials=NUM_TRIALS,
        callbacks=[early_stopper]
    )
    
    print("\n==================================")
    print("TUNING COMPLETE")