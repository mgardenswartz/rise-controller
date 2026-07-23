import argparse
import yaml
import numpy as np
import time
import math
import os
import sys

from quadsim import QuadSim
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file
from src.run_sim import SimRun

def evaluate_gains(param_dict: dict, base_config: dict, runner: LockstepRunner, px4: Px4Link):
    num_seeds = base_config['num_eval_seeds']
    base_seed = base_config['base_seed']
    xy_range = base_config['xy_rand_range_m']
    z_range = base_config['z_rand_range_m']

    costs = []
    e_rmses = []
    u_rmses = []
    
    base_desired_traj = base_config['desired_trajectory']
    if base_desired_traj == 1:
        base_x = base_config['traj1_init_x_m_ned']
        base_y = base_config['traj1_init_y_m_ned']
    else:
        base_x = base_config['traj2_init_x_m_ned']
        base_y = base_config['traj2_init_y_m_ned']

    yaw_rad = math.radians(base_config['init_yaw_deg'])

    print(f"\n[Test] Evaluating {num_seeds} randomized initial conditions for {param_dict['controller_type']}:")
    for i in range(num_seeds):
        np.random.seed(base_seed + i)
        
        batch_params = param_dict.copy()
        target_x = base_x + np.random.uniform(-xy_range, xy_range)
        target_y = base_y + np.random.uniform(-xy_range, xy_range)
        target_z = base_config['hover_start_z_m_ned'] + np.random.uniform(-z_range, z_range)
        batch_params['init_x_m_ned'] = target_x
        batch_params['init_y_m_ned'] = target_y
        batch_params['hover_start_z_m_ned'] = target_z
        
        print(f"[reset] PX4 position control -> ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        if not (px4.is_armed() and px4.in_offboard()):
            raise RuntimeError("vehicle left armed OFFBOARD during a trial")
        
        if not runner.fly_to_ned(
            target_x,
            target_y,
            target_z,
            yaw_ned=yaw_rad,
            tol=base_config['init_tol_m'],
            vel_tol=0.25,
            settle_s=2.0,
            timeout_s=40.0,
        ):
            print("[!] Could not recover and settle at hover origin!")
            raise RuntimeError("PX4 could not recover to start position.")
            
        print("  -> Arrived at start position. Handing over to RISE controller...")
        
        sim_run = SimRun(batch_params, yaml_config_path="conf/config.yaml", runner=runner, px4=px4)
        cost, e_rms, u_rms = sim_run.run()
        costs.append(cost)
        e_rmses.append(e_rms)
        u_rmses.append(u_rms)
        print(f"  -> Seed #{i+1} | Pos: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) | Cost: {cost:.4f} | e_RMS: {e_rms:.4f} | u_RMS: {u_rms:.4f}")
        
    worst_cost = float(np.max(costs))
    worst_e_rms = float(np.max(e_rmses))
    worst_u_rms = float(np.max(u_rmses))
    print(f"\n[Test Summary] Worst-Case Cost: {worst_cost:.4f} | Worst e_RMS: {worst_e_rms:.4f} | Worst u_RMS: {worst_u_rms:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gains in PX4 SITL")
    parser.add_argument("--desired_trajectory", type=int, required=True, choices=[1, 2], help="Desired trajectory (1 or 2)")
    parser.add_argument("--controller_type", type=str, required=True, help="Controller key in best_gains.yaml (e.g. BEST_PID, BEST_INN)")
    args = parser.parse_args()

    gains_file = os.path.join("output", f"traj{args.desired_trajectory}", "best_gains.yaml")
    if not os.path.exists(gains_file):
        print(f"Error: {gains_file} does not exist!")
        sys.exit(1)

    with open(gains_file, 'r') as f:
        all_gains = yaml.safe_load(f)

    controller_map = {
        "baseline": "BEST_RISE",
        "supertwisting": "BEST_ST",
        "pid": "BEST_PID",
        "integrated_resnet": "BEST_INN",
        "resnet": "BEST_NN",
    }
    
    target_key = controller_map.get(args.controller_type, args.controller_type)

    if target_key not in all_gains:
        print(f"Error: Controller '{args.controller_type}' (key '{target_key}') not found in {gains_file}.")
        print(f"Available controllers: {list(all_gains.keys())}")
        sys.exit(1)
        
    param_dict = all_gains[target_key]

    with open("conf/config.yaml", 'r') as f:
        full_config = yaml.safe_load(f)
    px4_config = full_config['px4']
    quadsim_config = full_config['quadsim']
    aviary_config = full_config['aviary_rise_node']['ros__parameters']

    if aviary_config['desired_trajectory'] != args.desired_trajectory:
        print(f"Warning: config.yaml has desired_trajectory={aviary_config['desired_trajectory']} but you passed --desired_trajectory {args.desired_trajectory}.")
        print(f"Overriding config trajectory to {args.desired_trajectory} for this test.")
        aviary_config['desired_trajectory'] = args.desired_trajectory

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
    runner.wait_px4()
    param_file = os.path.join("conf", px4_config["param_file"])
    load_param_file(px4, param_file)
    px4.configure_offboard_no_rc()
    if not runner.wait_ekf_ready() or not runner.wait_heading():
        raise RuntimeError("PX4 estimator is not ready")

    try:
        ground_ned = np.asarray(runner.ground_ned, dtype=float)
        start_position = [ground_ned[0], ground_ned[1], ground_ned[2] - 1.0] 
        
        px4.goto_ned(*start_position, yaw_ned=0.0)
        px4.emit_setpoint_now()
        if not runner.engage_offboard(arm=True):
            raise RuntimeError("PX4 refused OFFBOARD or arm")

        evaluate_gains(param_dict, aviary_config, runner, px4)
            
    except KeyboardInterrupt:
        print("\n[test] interrupted")
    except Exception as e:
        print(f"\n[test] stopped: {e}")
        import traceback
        traceback.print_exc()
    finally:
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
