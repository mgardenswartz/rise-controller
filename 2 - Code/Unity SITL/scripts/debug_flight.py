import argparse
import yaml
import os
from typing import Any, Dict

from src.run_sim import SimRun

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single debug flight with best gains.")
    parser.add_argument("--controller_type", type=str, required=True,
                        choices=["baseline", "resnet", "integrated_resnet", "supertwisting"],
                        help="The type of controller to run.")
    parser.add_argument("--config", type=str, default="conf/config.yaml", 
                        help="Path to the base config.yaml")
    parser.add_argument("--db_dir", type=str, required=True, 
                        help="Directory containing the best_gains.yaml file (e.g. output/traj1).")
    args = parser.parse_args()
        
    best_gains_path = os.path.join(args.db_dir, "best_gains.yaml")
    
    if not os.path.exists(best_gains_path):
        print(f"[!] {best_gains_path} not found. Running with base config.yaml parameters instead.")
        params: Dict[str, Any] = {'controller_type': args.controller_type}
    else:
        with open(best_gains_path, 'r') as f:
            best_gains = yaml.safe_load(f)
            
        # Map controller_type to the corresponding BEST_* key
        mapping = {
            "baseline": "BEST_RISE",
            "supertwisting": "BEST_ST",
            "resnet": "BEST_NN",
            "integrated_resnet": "BEST_INN"
        }
        
        target_key = mapping[args.controller_type]
        if target_key in best_gains:
            params = best_gains[target_key]
            print(f"[*] Loaded optimal gains for {args.controller_type} from {target_key}")
        else:
            print(f"[!] {target_key} not found in {best_gains_path}. Running with base config.yaml parameters.")
            params = {'controller_type': args.controller_type}

    print(f"[*] Initializing simulation with {args.controller_type}...")
    sim = SimRun(params, yaml_config_path=args.config)
    
    try:
        cost, e_rms, u_rms = sim.run()
        print(f"\n[*] Flight Complete! Final Cost: {cost:.4f} | E_RMS {e_rms} | u_RMS {u_rms}")
    except Exception as e:
        print(f"\n[!] Flight failed with error: {e}")

if __name__ == "__main__":
    main()
