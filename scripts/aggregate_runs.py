import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    base_dir = Path("outputs/unified_sweep")
    if not base_dir.exists():
        print("No sweep data found in outputs/unified_sweep.")
        return

    # Structure: results[sys_id][size_name][ctrl_name] = {'rms_e': [], 'rms_u': [], 'actual_p': 0}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'rms_e': [], 'rms_u': [], 'actual_p': 0})))

    # 1. Crawl the directories and parse JSONs
    for sys_dir in sorted(base_dir.glob("sys_*")):
        try:
            sys_id = int(sys_dir.name.split("_")[1])
        except ValueError:
            continue
            
        for ctrl_dir in sys_dir.iterdir():
            if not ctrl_dir.is_dir(): continue
            ctrl = ctrl_dir.name
            
            for size_dir in ctrl_dir.iterdir():
                if not size_dir.is_dir(): continue
                size_name = size_dir.name
                
                for seed_dir in size_dir.glob("seed_*"):
                    stat_file = seed_dir / "statistics.json"
                    if stat_file.exists():
                        with open(stat_file, 'r') as f:
                            stats = json.load(f)
                            
                            # Exact JSON Keys
                            e = stats.get('rms_tracking_error_norm', np.nan)
                            u = stats.get('rms_control_input_norm', np.nan)
                            p = stats.get('total_trainable_parameters', 0)
                            
                            results[sys_id][size_name][ctrl]['rms_e'].append(e)
                            results[sys_id][size_name][ctrl]['rms_u'].append(u)
                            results[sys_id][size_name][ctrl]['actual_p'] = int(p)

    # 2. Print the cleanly formatted table
    print("\n" + "="*112)
    print(f"{'Sys':<4} | {'Arch Size':<11} | {'Params (B/I)':<14} | {'Base RMS(e) [Surv]':<18} | {'Int. RMS(e) [Surv]':<18} | {'Base RMS(u)':<11} | {'Int. RMS(u)':<11}")
    print("-" * 112)

    for sys_id in sorted(results.keys()):
        for size_name in ["micro", "small", "medium", "large"]:
            if size_name not in results[sys_id]:
                continue
            
            data = results[sys_id][size_name]
            
            b_data = data.get('baseline', {'rms_e': [], 'rms_u': [], 'actual_p': 0})
            i_data = data.get('nn_in_integral', {'rms_e': [], 'rms_u': [], 'actual_p': 0})
            
            b_e_clean = [x for x in b_data['rms_e'] if not np.isnan(x) and not np.isinf(x)]
            b_u_clean = [x for x in b_data['rms_u'] if not np.isnan(x) and not np.isinf(x)]
            i_e_clean = [x for x in i_data['rms_e'] if not np.isnan(x) and not np.isinf(x)]
            i_u_clean = [x for x in i_data['rms_u'] if not np.isnan(x) and not np.isinf(x)]
            
            b_surv = len(b_e_clean)
            i_surv = len(i_e_clean)
            
            b_e_mean = f"{np.mean(b_e_clean):.4f}" if b_surv > 0 else "FAILED"
            i_e_mean = f"{np.mean(i_e_clean):.4f}" if i_surv > 0 else "FAILED"
            b_u_mean = f"{np.mean(b_u_clean):.2f}" if b_surv > 0 else "FAILED"
            i_u_mean = f"{np.mean(i_u_clean):.2f}" if i_surv > 0 else "FAILED"
            
            p_str = f"{b_data['actual_p']:>4} / {i_data['actual_p']:<4}"
            b_e_str = f"{b_e_mean:>9}  [{b_surv}/10]"
            i_e_str = f"{i_e_mean:>9}  [{i_surv}/10]"
            
            print(f" {sys_id:<3} | {size_name:<11} | {p_str:<14} | {b_e_str:<18} | {i_e_str:<18} | {b_u_mean:>11} | {i_u_mean:>11}")
            
    print("="*112 + "\n")

if __name__ == "__main__":
    main()