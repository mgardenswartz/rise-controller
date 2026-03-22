import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from master_sweep import MC_TRIALS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def aggregate_results_pivoted(base_dir="outputs/unified_sweep"):
    # Group by: (sys_id, P)
    # Store dictionaries for baseline and nn_in_integral
    aggregated_data = defaultdict(lambda: {
        "baseline": {"rms_e": [], "rms_u": []},
        "integral": {"rms_e": [], "rms_u": []}
    })

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found.")
        return

    for stats_file in base_path.rglob("statistics.json"):
        run_dir = stats_file.parent

        with open(stats_file, "r") as f:
            stats = json.load(f)
            
        try:
            ctrl_type = run_dir.parent.parent.name
            sys_str = run_dir.parent.parent.parent.name
            
            if not sys_str.startswith("sys_"):
                continue
                
            sys_id = int(sys_str.replace("sys_", ""))
            p = stats["total_trainable_parameters"]
            
            target_dict = "baseline" if ctrl_type == "baseline" else "integral"
            
            aggregated_data[(sys_id, p)][target_dict]["rms_e"].append(stats["rms_tracking_error_norm"])
            aggregated_data[(sys_id, p)][target_dict]["rms_u"].append(stats["rms_control_input_norm"])
            
        except (IndexError, ValueError, KeyError):
            continue

    # Print the Pivoted Report
    print(f"\n{'='*105}")
    print(f"{'Sys':<4} | {'Params':<8} | {'Base RMS(e)':<14} | {'Int. RMS(e)':<14} | {'Base RMS(u)':<14} | {'Int. RMS(u)':<14} | {'Status'}")
    print(f"{'-'*105}")

    for key in sorted(aggregated_data.keys()):
        sys_id, p = key
        data = aggregated_data[key]
        
        # Calculate means
        b_e = np.mean(data["baseline"]["rms_e"]) if data["baseline"]["rms_e"] else float('nan')
        b_u = np.mean(data["baseline"]["rms_u"]) if data["baseline"]["rms_u"] else float('nan')
        b_trials = len(data["baseline"]["rms_e"])
        
        i_e = np.mean(data["integral"]["rms_e"]) if data["integral"]["rms_e"] else float('nan')
        i_u = np.mean(data["integral"]["rms_u"]) if data["integral"]["rms_u"] else float('nan')
        i_trials = len(data["integral"]["rms_e"])
        
        # Status check for finite-time escapes
        status = "Complete"
        if b_trials < MC_TRIALS or i_trials < MC_TRIALS:
            status = f"FAIL (B:{b_trials}/{MC_TRIALS}, I:{i_trials}/{MC_TRIALS})"

        print(f"{sys_id:<4} | {int(p):<8} | {b_e:<14.4f} | {i_e:<14.4f} | {b_u:<14.4f} | {i_u:<14.4f} | {status}")
    
    print(f"{'='*105}\n")

if __name__ == "__main__":
    aggregate_results_pivoted()