import argparse
import optuna
import numpy as np
import yaml
from typing import Any
import time
from datetime import timedelta
import math
import os
import subprocess

from quadsim import QuadSim
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file
from src.run_sim import SimRun
import signal
import sys

# ==========================================
# Configuration Constants
# ==========================================
PX4_BOOT_SPEED_CAP = 1.0     # Cap sim speed to 1x during boot to prevent flooding uORB queues
PX4_FLY_TIMEOUT_S = 30.0     # Timeout for PX4 returning to the start position
MAX_CRASH_RETRIES = 1        # Number of crashes allowed before failing the trial
MAX_SETUP_RETRIES = 3        # Number of times to retry a failed setup (e.g. fly_to_ned timeout)
# ==========================================

def signal_handler(sig, frame):
    print("\n[!] Ctrl+C detected! Forcefully exiting optimization.py immediately.")
    os._exit(1)
    
signal.signal(signal.SIGINT, signal_handler)

# Globals for PX4 plant
sim = None
runner = None
px4 = None

class ETACallback:
    def __init__(self, target_trials: int):
        self.target_trials = target_trials
        self.start_time = time.time()
        self.trials_completed = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.trials_completed += 1
        elapsed = time.time() - self.start_time
        avg_time_per_trial = elapsed / self.trials_completed
        remaining_trials = self.target_trials - self.trials_completed
        
        if remaining_trials > 0:
            eta_seconds = remaining_trials * avg_time_per_trial
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            print(f"\n[ETA] Trial {self.trials_completed}/{self.target_trials} finished. "
                  f"Avg time/trial: {avg_time_per_trial:.1f}s. Estimated time remaining: {eta_str}\n")


class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.patience <= 0:
            return
            
        try:
            best_trial = study.best_trial
            # If the current trial number is X, and the best trial was found at X - patience, stop!
            if trial.number - best_trial.number >= self.patience:
                print("\n=========================================================================")
                print(f" [Early Stopping] No improvement in the last {self.patience} trials!")
                print(f" Best trial was #{best_trial.number} with cost {best_trial.value:.4f}.")
                print(" Exiting study prematurely.")
                print("=========================================================================\n")
                study.stop()
        except ValueError:
            # Raised if no trial has completed yet
            pass


def restart_px4():
    global runner, px4, sim
    print("\n[!] PX4 crash detected. Initiating recovery sequence...")
    
    if runner:
        try:
            runner.close()
        except Exception as e:
            print(f"[!] Error closing runner: {e}")
            
    print("[*] Killing existing PX4 instance...")
    os.system("pkill -9 -f px4")
    time.sleep(2.0)
    
    print("[*] Resetting Unity drone to safe hover position...")
    try:
        sim.drone(0).reset_pose(0.0, 0.3, 0.0)
    except Exception as e:
        print(f"[!] Warning: Could not reset pose: {e}")
        
    print("[*] Restarting PX4 SITL...")
    px4_log = open("px4_restart.log", "w")
    subprocess.Popen("make px4_sitl none_iris", cwd=os.path.expanduser("~/PX4-Autopilot"), shell=True, stdout=px4_log, stderr=subprocess.STDOUT)
    time.sleep(6.0)
    
    # Reload config
    with open("conf/config.yaml", 'r') as f:
        full_config = yaml.safe_load(f)
    px4_config = full_config['px4']
    aviary_config = full_config['aviary_rise_node']['ros__parameters']

    bridge = HilBridge(
        px4_host=px4_config["host"],
        instance=px4_config["instance"],
        px4_client=px4_config["client"],
    )
    px4 = Px4Link(stream_hz=0, instance=px4_config["instance"])
    runner = LockstepRunner(
        sim,
        bridge,
        px4,
        control_hz=aviary_config["control_frequency_hz"],
    )
    speed = aviary_config["sim_speed"]
    runner.speed_cap = speed if speed > 0.0 else None

    runner.start()
    
    # Cap sim speed to 1x during boot to prevent flooding PX4's uORB queues 
    # which causes EKF data loss and STALE sensors.
    runner.speed_cap = PX4_BOOT_SPEED_CAP
    runner.wait_px4()
    runner.speed_cap = speed if speed > 0.0 else None

    import threading
    def _setup_params():
        param_file = os.path.join("conf", px4_config["param_file"])
        load_param_file(px4, param_file)
        px4.configure_offboard_no_rc()
        
    th = threading.Thread(target=_setup_params)
    th.start()
    while th.is_alive():
        runner.hil_tick()
        time.sleep(0.002)

    if not runner.wait_ekf_ready() or not runner.wait_heading():
        raise RuntimeError("PX4 estimator is not ready after restart")
        
    ground_ned = np.asarray(runner.ground_ned, dtype=float)
    start_position = [ground_ned[0], ground_ned[1], ground_ned[2] - 1.0] 
    
    px4.goto_ned(*start_position, yaw_ned=0.0)
    px4.emit_setpoint_now()
    if not runner.engage_offboard(arm=True):
        raise RuntimeError("PX4 refused OFFBOARD or arm after restart")
    
    print("[*] PX4 restarted and armed successfully.\n")

def evaluate_minibatch(trial: optuna.Trial, param_dict: dict[str, Any]) -> float:
    """Runs a mini-batch of robust domain randomizations and returns the worst-case cost."""
    global runner, px4
    
    # Check if we should stop early due to poor performance
    worst_cost = -1.0
    costs = []
    e_rmses = []
    u_rmses = []
    
    with open("conf/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']

    num_seeds = base_config['num_eval_seeds']
    base_seed = base_config['base_seed']
    xy_range = base_config['xy_rand_range_m']
    z_range = base_config['z_rand_range_m']

    u_rmses = []
    base_desired_traj = base_config['desired_trajectory']
    if base_desired_traj == 1:
        base_x = base_config['traj1_init_x_m_ned']
        base_y = base_config['traj1_init_y_m_ned']
    else:
        base_x = base_config['traj2_init_x_m_ned']
        base_y = base_config['traj2_init_y_m_ned']

    yaw_rad = math.radians(base_config['init_yaw_deg'])

    traj1_fixed = [(1.5, 5.5), (1.5, 2.5), (-1.5, 5.5), (-1.5, 2.5)]
    traj2_fixed = [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5)]

    print(f"\n[Mini-Batch] Evaluating {num_seeds} initial conditions:")
    crash_count = 0
    i = 0
    while i < num_seeds:
        # Deterministic perturbation based on seed index
        np.random.seed(base_seed + i)
        
        batch_params = param_dict.copy()
        
        if i < 4:
            if base_desired_traj == 1:
                target_x, target_y = traj1_fixed[i]
            else:
                target_x, target_y = traj2_fixed[i]
            target_z = base_config['init_z_m_ned']
        else:
            target_x = base_x + np.random.uniform(-xy_range, xy_range)
            target_y = base_y + np.random.uniform(-xy_range, xy_range)
            target_z = base_config['init_z_m_ned'] + np.random.uniform(-z_range, z_range)
            
        batch_params['init_x_m_ned'] = target_x
        batch_params['init_y_m_ned'] = target_y
        batch_params['hover_start_z_m_ned'] = target_z
        
        print(f"[reset] PX4 position control -> ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        
        try:
            if not (px4.is_armed() and px4.in_offboard()):
                raise RuntimeError("vehicle left armed OFFBOARD during a trial")
            
            if not runner.fly_to_ned(
                target_x,
                target_y,
                target_z,
                yaw_ned=yaw_rad,
                tol=base_config['init_tol_m'],
                vel_tol=0.50,
                settle_s=1.0,
                timeout_s=PX4_FLY_TIMEOUT_S,
            ):
                print("[!] Could not recover and settle at hover origin!")
                
                setup_retries += 1
                if setup_retries > MAX_SETUP_RETRIES:
                    raise RuntimeError("Too many setup failures. The simulation environment or PX4 is fundamentally broken.")
                
                print(f"[!] SETUP FAILED (Attempt {setup_retries}/{MAX_SETUP_RETRIES}).")
                print("[!] This is NOT a controller failure. Restarting PX4 and retrying this seed without penalizing the trial...")
                restart_px4()
                continue
                
            print("  -> Arrived at start position. Handing over to RISE controller...")
            
            sim_run = SimRun(batch_params, yaml_config_path="conf/config.yaml", runner=runner, px4=px4)
            cost, e_rms, u_rms = sim_run.run()
            costs.append(cost)
            e_rmses.append(e_rms)
            u_rmses.append(u_rms)
            print(f"  -> Seed #{i+1} | Pos: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) | Cost: {cost:.4f}")
            
            crash_count = 0 # reset crash count on success
            setup_retries = 0 # reset setup retries on success
            i += 1 # advance to next seed
            
        except RuntimeError as e:
            # Check if this was our setup failure abort
            if "Too many setup failures" in str(e):
                raise e
                
            print(f"[W] Trial encountered runtime error: {e}")
            crash_count += 1
            if crash_count >= MAX_CRASH_RETRIES:
                print(f"[!] Trial failed {crash_count} times in a row. Assigning max cost and continuing.")
                worst_cost = 1e6
                worst_e_rms = 1e6
                worst_u_rms = 1e6
                trial.set_user_attr('e_RMS', worst_e_rms)
                trial.set_user_attr('u_RMS', worst_u_rms)
                print(f"[Mini-Batch] Terminated early due to repeated crashes. Cost: {worst_cost:.4f}")
                
                print("[!] Restarting PX4 to ensure a clean state for the next trial...")
                restart_px4()
                
                return worst_cost
                
            restart_px4()
            print(f"[!] Retrying seed #{i+1} (Attempt {crash_count+1}/{MAX_CRASH_RETRIES})...")
            continue
        
    worst_cost = float(np.max(costs))
    worst_e_rms = float(np.max(e_rmses))
    worst_u_rms = float(np.max(u_rmses))
    trial.set_user_attr('e_RMS', worst_e_rms)
    trial.set_user_attr('u_RMS', worst_u_rms)
    print(f"[Mini-Batch] Completed. Worst-Case Cost: {worst_cost:.4f}")
    return worst_cost

def run_stage_1a(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'baseline',
        'k_1': trial.suggest_float("k_1", 0.01, 2.0, log=True),
        'k_rise': trial.suggest_float("k_rise", 0.01, 8.0, log=True)
    }
    param_dict['k_2'] = trial.suggest_float("k_2", 0.01, 8.0, log=True)
    param_dict['k_3'] = trial.suggest_float("k_3", param_dict['k_2'], 8.0)
    return evaluate_minibatch(trial, param_dict)

def run_stage_1b(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'baseline',
        'k_1': trial.suggest_float("k_1", 0.01, 2.0, log=True),
        'k_rise': trial.suggest_float("k_rise", 0.01, 8.0, log=True)
    }
    param_dict['k_2'] = trial.suggest_float("k_2", 0.01, 8.0, log=True)
    param_dict['k_3'] = trial.suggest_float("k_3", param_dict['k_2'], 8.0)
    return evaluate_minibatch(trial, param_dict)

def run_stage_2a(trial: optuna.Trial, db_dir: str) -> float:
    with open("conf/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
    base_stage = base_config['stage2_base_gains']
    
    db_filename = f"stage_{base_stage}.db"
    study_name = f"stage_{base_stage}_study"
    db_file_path = os.path.join(db_dir, db_filename)
    
    best_base_params = {}
    if os.path.exists(db_file_path):
        try:
            stage_db_url = f"sqlite:///{db_file_path}"
            study = optuna.load_study(study_name=study_name, storage=stage_db_url)
            best_base_params = study.best_params
        except Exception as e:
            print(f"Warning: Could not load {db_filename} to seed stage 2! Error: {e}")
    else:
        print(f"Warning: {db_filename} does not exist. Cannot seed Stage 2. Falling back to config.yaml.")

    param_dict = {
        'controller_type': 'resnet', # Stage 2A tunes the Neural Network
        'initial_weight_scale_factor': 0.5, 
        'num_blocks': 4,
        'k_0': 2,
        'k_i': 2,
        'hidden_width': 8,
        'gamma': trial.suggest_float("gamma", 0.0, 100.0),
        'sigma_mod': trial.suggest_float("sigma_mod", 0.01, 20.0, log=True),
        **best_base_params
    }
    return evaluate_minibatch(trial, param_dict)

def run_stage_2b(trial: optuna.Trial, db_dir: str) -> float:
    with open("conf/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
    base_stage = base_config['stage2_base_gains']
    
    db_filename = f"stage_{base_stage}.db"
    study_name = f"stage_{base_stage}_study"
    db_file_path = os.path.join(db_dir, db_filename)
    
    best_base_params = {}
    if os.path.exists(db_file_path):
        try:
            stage_db_url = f"sqlite:///{db_file_path}"
            study = optuna.load_study(study_name=study_name, storage=stage_db_url)
            best_base_params = study.best_params
        except Exception as e:
            print(f"Warning: Could not load {db_filename} to seed stage 2! Error: {e}")
    else:
        print(f"Warning: {db_filename} does not exist. Cannot seed Stage 2. Falling back to config.yaml.")

    param_dict = {
        'controller_type': 'integrated_resnet', # Stage 2B tunes the Integrated Neural Network (INN)
        'initial_weight_scale_factor': 0.1, 
        'num_blocks': 4,
        'k_0': 2,
        'k_i': 2,
        'hidden_width': 8,
        'gamma': trial.suggest_float("gamma", 0.0, 100.0),
        'sigma_mod': trial.suggest_float("sigma_mod", 0.01, 20.0, log=True),
        **best_base_params
    }
    return evaluate_minibatch(trial, param_dict)

def run_stage_3(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'supertwisting',
        'k_1': trial.suggest_float("k_st_1", 0.001, 5.0, log=True),
        'k_2': trial.suggest_float("k_st_2", 0.001, 5.0, log=True),
        'k_3': trial.suggest_float("k_st_3", 0.001, 5.0, log=True)
    }
    return evaluate_minibatch(trial, param_dict)

def run_stage_4(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'pid',
        'K_P': trial.suggest_float("K_P", 0.01, 50.0, log=True),
        'K_I': trial.suggest_float("K_I", 0.01, 50.0, log=True),
        'K_D': trial.suggest_float("K_D", 0.01, 50.0, log=True),
        # Need dummy values for these so sim initialization doesn't fail
        'k_1': 0.0,
        'k_2': 0.0,
        'k_3': 0.0,
        'k_rise': 0.0,
    }
    return evaluate_minibatch(trial, param_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Orchestrator for Quadcopter Adaptive Control")
    parser.add_argument("--stage", type=str, required=True, choices=['1A', '1B', '2A', '2B', '3', '4', 'LHS'], help="Optimization stage to run.")
    parser.add_argument("--num_trials", type=int, required=True, help="Number of trials.")
    parser.add_argument("--db_dir", type=str, required=True, help="Directory for Optuna databases (e.g. output/traj1).")
    parser.add_argument("--patience", type=int, required=True, help="Number of trials to wait for improvement before stopping early. 0 to disable.")
    args = parser.parse_args()

    # Initialize PX4 plant config
    with open("conf/config.yaml", 'r') as f:
        full_config = yaml.safe_load(f)
    px4_config = full_config['px4']
    quadsim_config = full_config['quadsim']
    aviary_config = full_config['aviary_rise_node']['ros__parameters']
    desired_traj = aviary_config['desired_trajectory']

    # Construct the file path and SQLite URL
    os.makedirs(args.db_dir, exist_ok=True)
    db_file_path = os.path.join(args.db_dir, f"stage_{args.stage}.db")
    db_url = f"sqlite:///{db_file_path}"
    
    if os.path.exists(db_file_path):
        print("\n=========================================================================")
        print(f" ⚠️ WARNING: Database '{db_file_path}' already exists.")
        print(" Optuna will RESUME the existing study where it left off.")
        print(" NOTE: This assumes your search space bounds have NOT changed.")
        print(" If you altered the parameter bounds, cancel this run and delete the .db file.")
        print("=========================================================================\n")

    study_name = f"stage_{args.stage}_study"
    study = optuna.create_study(study_name=study_name, storage=db_url, load_if_exists=True, direction="minimize")

    eta_callback = ETACallback(args.num_trials)
    early_stop_callback = EarlyStoppingCallback(args.patience)
    from typing import Callable
    callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = [eta_callback, early_stop_callback]
    
    sim = QuadSim(
        host=quadsim_config["host"],
        command_port=quadsim_config["port"],
        telemetry_port=quadsim_config["port"] + 1,
    )
    sim.connect()
    bridge = HilBridge(
        px4_host=px4_config["host"],
        instance=px4_config["instance"],
        px4_client=px4_config["client"],
    )
    px4 = Px4Link(stream_hz=0, instance=px4_config["instance"])
    runner = LockstepRunner(
        sim,
        bridge,
        px4,
        control_hz=aviary_config["control_frequency_hz"],
    )
    speed = aviary_config["sim_speed"]
    runner.speed_cap = speed if speed > 0.0 else None

    runner.start()
    
    # Cap sim speed to 1x during boot to prevent flooding PX4's uORB queues 
    # which causes EKF data loss and STALE sensors.
    runner.speed_cap = PX4_BOOT_SPEED_CAP
    runner.wait_px4()
    runner.speed_cap = speed if speed > 0.0 else None

    import threading
    def _setup_params_main():
        param_file = os.path.join("conf", px4_config["param_file"])
        load_param_file(px4, param_file)
        px4.configure_offboard_no_rc()
        
    th = threading.Thread(target=_setup_params_main)
    th.start()
    while th.is_alive():
        runner.hil_tick()
        time.sleep(0.002)
    if not runner.wait_ekf_ready() or not runner.wait_heading():
        raise RuntimeError("PX4 estimator is not ready")

    try:
        # Get ground reference
        ground_ned = np.asarray(runner.ground_ned, dtype=float)
        start_position = [ground_ned[0], ground_ned[1], ground_ned[2] - 1.0] 
        
        px4.goto_ned(*start_position, yaw_ned=0.0)
        px4.emit_setpoint_now()
        if not runner.engage_offboard(arm=True):
            raise RuntimeError("PX4 refused OFFBOARD or arm")

        if args.stage == '1A':
            study.optimize(run_stage_1a, n_trials=args.num_trials, callbacks=callbacks)
        elif args.stage == '1B':
            study.optimize(run_stage_1b, n_trials=args.num_trials, callbacks=callbacks)
        elif args.stage == '2A':
            base_stage = aviary_config['stage2_base_gains']
            stage_db_url = f"sqlite:///{os.path.join(args.db_dir, f'stage_{base_stage}.db')}"
            try:
                study_base = optuna.load_study(study_name=f"stage_{base_stage}_study", storage=stage_db_url)
                print("\n=========================================================================")
                print(f" [Sanity Check] Stage {base_stage} Best Gains injected into Stage 2A:")
                for k, v in study_base.best_params.items():
                    print(f"    {k}: {v}")
                print("=========================================================================\n")
            except Exception:
                print(f"\n[!] Could not load Stage {base_stage} gains for sanity check print.\n")
                
            study.optimize(lambda t: run_stage_2a(t, args.db_dir), n_trials=args.num_trials, callbacks=callbacks)
        elif args.stage == '2B':
            base_stage = aviary_config['stage2_base_gains']
            stage_db_url = f"sqlite:///{os.path.join(args.db_dir, f'stage_{base_stage}.db')}"
            try:
                study_base = optuna.load_study(study_name=f"stage_{base_stage}_study", storage=stage_db_url)
                print("\n=========================================================================")
                print(f" [Sanity Check] Stage {base_stage} Best Gains injected into Stage 2B:")
                for k, v in study_base.best_params.items():
                    print(f"    {k}: {v}")
                print("=========================================================================\n")
            except Exception:
                print(f"\n[!] Could not load Stage {base_stage} gains for sanity check print.\n")
                
            study.optimize(lambda t: run_stage_2b(t, args.db_dir), n_trials=args.num_trials, callbacks=callbacks)
        elif args.stage == '3':
            study.optimize(run_stage_3, n_trials=args.num_trials, callbacks=callbacks)
        elif args.stage == '4':
            study.optimize(run_stage_4, n_trials=args.num_trials, callbacks=callbacks)
            
    except KeyboardInterrupt:
        print("\n[study] interrupted")
    except Exception as e:
        print(f"\n[study] stopped: {e}")
    finally:
        # Land and cleanup cleanly
        try:
            if px4 and px4.is_armed():
                print("\nLanding drone before exit...")
                ground = runner.ground_ned
                runner.fly_to_ned(
                    ground[0],
                    ground[1],
                    ground[2] - 0.08,
                    yaw_ned=0.0,
                    tol=0.12,
                    vel_tol=0.25,
                    settle_s=1.5,
                    timeout_s=45.0,
                )
                runner.disarm()
        except KeyboardInterrupt:
            print("\n[!] Force quitting during landing.")
        except Exception as e:
            print(f"\n[!] Error during landing: {e}")
            
        if runner:
            runner.close()
        if sim:
            sim.resume()
            sim.disconnect()