#!/usr/bin/env python3
import os
import yaml
import argparse
import jax

# Prevent JAX from gobbling up GPU if any, stay on CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# We can reuse the build_param_dict function from your evaluation pipeline!
from run_best_gains import build_param_dict

def main():
    parser = argparse.ArgumentParser(description="Generate Hardware Param YAML")
    parser.add_argument("--controller_type", type=str, choices=["baseline", "resnet", "integrated_resnet", "supertwisting"], required=True)
    parser.add_argument("--desired_trajectory", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out", type=str, default="hardware_params.yaml", help="Output yaml file path")
    args = parser.parse_args()

    # This securely calls the exact same JAX random seed logic and network architectures
    param_dict = build_param_dict(args.controller_type, args.desired_trajectory)
    
    param_dict['is_gazebo'] = False
    param_dict['save_data'] = True
    param_dict['mpc_acc_vert_max'] = 6.0 #26.0
    param_dict['theta_bar'] = 1e6
    param_dict['odom_timeout_sec'] = 2.0
    param_dict['init_z'] = -0.75
    param_dict['vehicle_name'] = 'sentinel5'

    # TEMP
    param_dict['K_P'] = 6.64
    param_dict['K_I'] = 0.58
    param_dict['K_D'] = 3.51
    param_dict['traj1_period'] = 30.0
    param_dict['traj1_z_amp'] = 0.25
    param_dict['traj2_target_speed'] = 1.5

    # Don't forget about sigma_mod.

    params = {
        'aviary_rise_node': {
            'ros__parameters': param_dict
        }
    }

    with open(args.out, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    print(f"[*] Generated hardware parameters for {args.controller_type.upper()} running Trajectory {args.desired_trajectory}.")
    print(f"[*] Saved to {args.out}")

if __name__ == "__main__":
    main()
