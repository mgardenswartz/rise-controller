#!/usr/bin/env python3
import os
import yaml
import argparse
import jax
from jax_resnet import init_resnet_weights

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

def main():
    parser = argparse.ArgumentParser(description="Generate Hardware Param YAML")
    parser.add_argument("--best_gains", type=str, required=True, help="Path to best_gains.yaml file")
    parser.add_argument("--controller_type", type=str, choices=["baseline", "resnet", "integrated_resnet", "supertwisting", "st", "pid", "baseline_no_wind"], required=True)
    parser.add_argument("--desired_trajectory", type=int, choices=[1, 2], required=False, help="Desired trajectory (optional override)")
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="Path to base config.yaml")
    parser.add_argument("--out", type=str, default="hardware_params.yaml", help="Output yaml file path")
    parser.add_argument("--gazebo", type=bool, required=True, help="Output yaml file path")
    args = parser.parse_args()

    # Support 'st' alias for 'supertwisting'
    target_controller_type = "supertwisting" if args.controller_type == "st" else args.controller_type

    if not os.path.exists(args.best_gains):
        raise FileNotFoundError(f"Best gains file not found: {args.best_gains}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Base config file not found: {args.config}")

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
        base_config = full_config['aviary_rise_node']['ros__parameters']

    with open(args.best_gains, 'r') as f:
        best_gains = yaml.safe_load(f)

    # Find the specific controller parameters in best_gains.yaml
    controller_params = None
    for key, params in best_gains.items():
        if params.get('controller_type') == target_controller_type:
            controller_params = params
            break
            
    if controller_params is None:
        raise ValueError(f"Could not find controller_type '{target_controller_type}' in {args.best_gains}")

    # Build the combined parameter dictionary
    param_dict = base_config.copy()
    param_dict.update(controller_params)

    # Allow overriding desired_trajectory via command line
    if args.desired_trajectory is not None:
        param_dict['desired_trajectory'] = args.desired_trajectory

    # Hardware-specific overrides translated from the old script
    param_dict['is_gazebo'] = False
    param_dict['save_data'] = True
    param_dict['mpc_acc_vert_max_mps2'] = 6.0 # updated parameter name
    param_dict['theta_bar'] = 1e6
    param_dict['odom_timeout_s'] = 1.0
    param_dict['odom_watchdog_freq'] = 10.0
    if args.gazebo:
        param_dict['vehicle_name'] = 'px4_1'
    else:
        param_dict['vehicle_name'] = 'sentinel5'

    # TEMP overrides
    param_dict['traj1_z_amp_m_ned_aviary'] = 0.25
    param_dict['init_z_m_ned_aviary'] = -0.75 # updated parameter name

    # Generate initial neural network weights if applicable
    if target_controller_type in ["resnet", "integrated_resnet"]:
        key = jax.random.PRNGKey(param_dict.get('base_seed', 42))
        init_scale = param_dict.get('initial_weight_scale_factor', 0.2)
        
        initial_weights_jax = init_scale * init_resnet_weights(
            key=key,
            d_in=param_dict['d_in'],
            hidden_width=param_dict['hidden_width'],
            d_out=param_dict['d_out'],
            b=param_dict['num_blocks'],
            k_0=param_dict['k_0'],
            k_i=param_dict['k_i'],
            h_method=param_dict['h_method'],
            o_method=param_dict['o_method']
        )
        param_dict['initial_weights'] = [float(w) for w in initial_weights_jax]

    params = {
        'aviary_rise_node': {
            'ros__parameters': param_dict
        }
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    print(f"[*] Generated hardware parameters for {target_controller_type.upper()} running Trajectory {param_dict['desired_trajectory']}.")
    print(f"[*] Saved to {args.out}")

if __name__ == "__main__":
    main()
