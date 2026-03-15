import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Force Python to see the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def aggregate_results(base_dir="outputs/massive_sweep"):
    aggregated_data = defaultdict(lambda: {"rms_e": [], "rms_u": [], "flops": 0})

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found. Ensure you are running from the project root.")
        return

    # Crawl all subdirectories looking for completed runs
    for stats_file in base_path.rglob("statistics.json"):
        run_dir = stats_file.parent

        with open(stats_file, "r") as f:
            stats = json.load(f)
            
        try:
            # Extract metadata directly from the directory path
            # Path looks like: .../sys_3/nn_in_integral/p_200/seed_1009
            ctrl_type = run_dir.parent.parent.name
            sys_str = run_dir.parent.parent.parent.name
            
            # Safety check to ensure we are parsing the right folder level
            if not sys_str.startswith("sys_"):
                continue
                
            sys_id = int(sys_str.replace("sys_", ""))
            p = stats["total_trainable_parameters"]
            
            key = (sys_id, ctrl_type, p)
            aggregated_data[key]["rms_e"].append(stats["rms_tracking_error_norm"])
            aggregated_data[key]["rms_u"].append(stats["rms_control_input_norm"])
            aggregated_data[key]["flops"] = stats["forward_pass_flops"]
        except (IndexError, ValueError, KeyError):
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
    aggregate_results(base_dir="outputs/massive_sweep")