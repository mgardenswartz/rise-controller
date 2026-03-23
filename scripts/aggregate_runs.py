import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def format_stats(clean_array):
    """Returns a formatted string 'Median (Max)' and the raw median for math."""
    if not clean_array:
        return "FAILED", np.nan
    
    med = np.median(clean_array)
    mx = np.max(clean_array)
    
    mx_str = f"{mx:.2e}" if mx > 9999 else f"{mx:.4f}"
    return f"{med:.4f} ({mx_str})", med

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
    print("\n" + "="*162)
    print(f"{'Sys':<3} | {'Arch Size':<10} | {'Params (B/I)':<12} | {'Base RMS(e): Med (Max) [Surv]':<32} | {'Int. RMS(e): Med (Max) [Surv]':<32} | {'% Imp (e)':<10} | {'Base RMS(u): Med (Max)':<22} | {'Int. RMS(u): Med (Max)':<22}")
    print("-" * 162)

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
            
            b_surv, i_surv = len(b_e_clean), len(i_e_clean)
            
            # Format Strings and Extract Medians
            b_e_str, b_e_med = format_stats(b_e_clean)
            i_e_str, i_e_med = format_stats(i_e_clean)
            b_u_str, _ = format_stats(b_u_clean)
            i_u_str, _ = format_stats(i_u_clean)
            
            # Calculate Percentage Improvement (Positive = Integral is better)
            if b_surv > 0 and i_surv > 0:
                improvement = ((b_e_med - i_e_med) / b_e_med) * 100.0
                imp_str = f"{improvement:+.1f}%"
            else:
                imp_str = "N/A"
            
            p_str = f"{b_data['actual_p']:>4} / {i_data['actual_p']:<4}"
            
            # Combine Error with Survival Counts
            b_e_full = f"{b_e_str:>19}  [{b_surv:2d}/10]" if b_surv > 0 else f"{'FAILED':>19}  [ 0/10]"
            i_e_full = f"{i_e_str:>19}  [{i_surv:2d}/10]" if i_surv > 0 else f"{'FAILED':>19}  [ 0/10]"
            b_u_full = f"{b_u_str:>22}" if b_surv > 0 else f"{'FAILED':>22}"
            i_u_full = f"{i_u_str:>22}" if i_surv > 0 else f"{'FAILED':>22}"
            
            print(f" {sys_id:<3} | {size_name:<10} | {p_str:<12} | {b_e_full:<32} | {i_e_full:<32} | {imp_str:>10} | {b_u_full:<22} | {i_u_full:<22}")
            
    print("="*162 + "\n")

if __name__ == "__main__":
    main()