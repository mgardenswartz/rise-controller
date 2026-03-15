import os
import json
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np

def aggregate_results(base_dir="outputs"):
    # Group by: (sys_id, controller_type, P)
    aggregated_data = defaultdict(lambda: {"rms_e": [], "rms_u": [], "flops": 0})

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found.")
        return

    # Crawl all subdirectories looking for completed runs
    for stats_file in base_path.rglob("statistics.json"):
        run_dir = stats_file.parent
        config_file = run_dir / ".hydra" / "config.yaml"

        if not config_file.exists():
            continue

        with open(stats_file, "r") as f:
            stats = json.load(f)
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        try:
            sys_id = config["simulation"]["sys_id"]
            ctrl_type = config["simulation"]["controller_type"]
            p = stats["total_trainable_parameters"]
            
            key = (sys_id, ctrl_type, p)
            aggregated_data[key]["rms_e"].append(stats["rms_tracking_error_norm"])
            aggregated_data[key]["rms_u"].append(stats["rms_control_input_norm"])
            aggregated_data[key]["flops"] = stats["forward_pass_flops"]
        except KeyError:
            continue

    # Print the Report
    print(f"\n{'='*80}")
    print(f"{'Sys':<5} | {'Controller':<18} | {'Params (P)':<12} | {'Trials':<8} | {'Avg RMS(e)':<12} | {'Avg RMS(u)':<12}")
    print(f"{'-'*80}")

    # Sort by System, then Controller, then Parameter count
    for key in sorted(aggregated_data.keys()):
        sys_id, ctrl_type, p = key
        data = aggregated_data[key]
        
        trials = len(data["rms_e"])
        avg_e = np.mean(data["rms_e"])
        avg_u = np.mean(data["rms_u"])
        
        print(f"{sys_id:<5} | {ctrl_type:<18} | {int(p):<12} | {trials:<8} | {avg_e:<12.4f} | {avg_u:<12.4f}")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    aggregate_results()