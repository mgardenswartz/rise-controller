#!/usr/bin/env python3
import os
import subprocess
import time
import re
import argparse
import optuna
import sys
import yaml
import random
from functools import partial

# Ensure JAX stays on CPU 
import jax
jax.config.update("jax_platform_name", "cpu")

# Add custom path for ResNet imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ros2_ws", "src", "aviary_rise_controller", "aviary_rise_controller")))
from jax_resnet import get_total_parameters, init_resnet_weights

# --- Fixed Global Configuration ---
SEED = 42
random.seed(SEED)

FIXED_X = 0.70
FIXED_Y = -2.37
TARGET_Z = -3.0

# Fixed Phase 2 Baseline Gains (Optimized from Phase 1)
FIXED_K1 = 1.537
FIXED_K2 = 0.3767
FIXED_K3 = 1.0884
FIXED_K_RISE = 0.006566

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
        "--controller", 
        type=str, 
        required=True, 
        choices=["baseline", "developed", "noresnet"],
        help="Specify which controller profile to optimize."
    )
    return parser.parse_args()

def get_storage_config(controller: str):
    if controller == "noresnet":
        return "sqlite:///phase1_tuning.db", "phase1_noresnet_baseline_tuning"
    return f"sqlite:///phase2_{controller}.db", f"phase2_{controller}_optimization"

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
        cmd = 'docker exec px4-sitl-gz bash -c "source /home/root/ros-sources.sh && timeout 2 ros2 topic echo --once /px4_1/fmu/out/vehicle_status"'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            print("[*] DDS Bridge connected and publishing!")
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

def execute_single_run(param_dict: dict):
    cleanup_environment()
    
    container_param_path = write_ros_params(param_dict)
    write_spawn_location(FIXED_X, FIXED_Y)
    
    subprocess.run(["./spawn-sim-env.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
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
    for line in process.stdout:
        print(f"  [ROS] {line.strip()}")
        match = re.search(r"\[RESULT\] ITAE_COST = ([\d\.]+)", line)
        if match:
            final_cost = float(match.group(1))
            
    process.wait()
    cleanup_environment()
    return final_cost

def run_trial(param_dict: dict) -> float:
    for attempt in range(1, 4):
        if attempt != 1:
            print(f"\n[*] Execution Attempt {attempt}/3")
        
        cost = execute_single_run(param_dict)
        
        if cost is not None:
            return cost
            
        print("[!] Trial returned None (System/SITL failure). Retrying...")
        time.sleep(5)
        
    print("[!] All 3 attempts failed. Assigning ultimate penalty.")
    return 1e6

def objective(trial: optuna.Trial, controller: str) -> float:
    # Base parameters required for all controllers
    param_dict = {
        'vehicle': 'px4_1',
        'z': TARGET_Z,
        'controller': controller
    }
    
    print(f"\n==============================================")
    print(f"[*] Starting trial for controller: {controller}")
    
    if controller == "noresnet":
        # Phase 1 Search Space
        param_dict['k1'] = trial.suggest_float("k1", 1.0, 8.0)
        param_dict['k2'] = trial.suggest_float("k2", 0.1, 5.0)
        param_dict['k3'] = trial.suggest_float("k3", param_dict['k2'], 5.0) # BREAK SYMMETRY: Force k3's lower bound to be k2
        param_dict['k_rise'] = trial.suggest_float("k_rise", 0.0001, 0.05, log=True)
        
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
        
        d_in = 15
        d_out = 3
        hidden_width = 16
        num_blocks = 1
        k_0 = 1
        k_i = 1

        param_dict['hidden_width'] = hidden_width
        param_dict['num_blocks'] = num_blocks
        param_dict['k_0'] = k_0
        param_dict['k_i'] = k_i

        hidden_width = trial.suggest_categorical("hidden_width", [16, 32])
        param_dict['gamma'] = float(trial.suggest_float("gamma", 0.5, 50.0, log=True))
        param_dict['sigma_mod'] = float(trial.suggest_float("sigma_mod", 0.001, 5.0, log=True))

        total_params = get_total_parameters(d_in, hidden_width, d_out, num_blocks, k_0, k_i)
        
        key = jax.random.PRNGKey(42)
        initial_weights_jax = init_resnet_weights(key, d_in, hidden_width, d_out, num_blocks, k_0, k_i, 'xavier', 'he')
        param_dict['initial_weights'] = [float(w) for w in initial_weights_jax]
        
        print(f"[*] Evaluating Architecture: {total_params} parameters.")
        print(f"[*] Gamma: {param_dict['gamma']:.2f} | Sigma Mod: {param_dict['sigma_mod']:.6f} | k_0: {k_0} | k_i: {k_i}")
        print(f"[*] Theta Bar: {param_dict['theta_bar']:.4f} (Scale: {theta_bar_scale:.4f})")

    param_dict['plot'] = False 
    print(f"==============================================")
    
    return run_trial(param_dict)

if __name__ == "__main__":
    args = parse_args()
    db_name, study_name = get_storage_config(args.controller)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_name,
        load_if_exists=True,
        direction="minimize"
    )
    
    print(f"[*] Starting Tuning for {args.controller}. Saving to {db_name}")
    
    # Initialize the early stopping callback with 50 trials of patience
    early_stopper = PatienceCallback(patience=100)
    
    bound_objective = partial(objective, controller=args.controller)
    
    study.optimize(
        bound_objective, 
        n_trials=500,
        callbacks=[early_stopper]
    )
    
    print("\n==================================")
    print("TUNING COMPLETE")