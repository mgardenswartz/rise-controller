import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

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

    # Structure: results[sys_id][detune_val][size_name][ctrl] = {'rms_e': [], 'rms_u': [], 'actual_p': 0}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'rms_e': [], 'rms_u': [], 'actual_p': 0}))))

    # Regex to extract the sys_id and detune multiplier from the directory name
    dir_pattern = re.compile(r"sys_(\d+)_detune_([0-9.]+)")

    # 1. Crawl the directories and parse JSONs
    for sys_dir in base_dir.iterdir():
        if not sys_dir.is_dir(): continue
        
        match = dir_pattern.match(sys_dir.name)
        if not match:
            continue
            
        sys_id = int(match.group(1))
        detune_val = float(match.group(2))
        
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
                            
                            e = stats.get('rms_tracking_error_norm', np.nan)
                            u = stats.get('rms_control_input_norm', np.nan)
                            p = stats.get('total_trainable_parameters', 0)
                            
                            results[sys_id][detune_val][size_name][ctrl]['rms_e'].append(e)
                            results[sys_id][detune_val][size_name][ctrl]['rms_u'].append(u)
                            results[sys_id][detune_val][size_name][ctrl]['actual_p'] = int(p)

    if not results:
        print("No valid detune directories parsed. Check your outputs folder.")
        return

    # 2. Print the cleanly formatted table
    print("\n" + "="*175)
    print(f"{'Sys':<3} | {'Detune':<6} | {'Arch Size':<9} | {'Params(B/I)':<11} | {'Base RMS(e): Med (Max) [Surv]':<31} | {'Int. RMS(e): Med (Max) [Surv]':<31} | {'% Imp(e)':<8} | {'Base RMS(u): Med(Max)':<22} | {'Int. RMS(u): Med(Max)':<22}")
    print("-" * 175)

    # Sort Systems ascending, then Detune descending (1.0 -> 0.1)
    for sys_id in sorted(results.keys()):
        for detune_val in sorted(results[sys_id].keys(), reverse=True):
            for size_name in ["small", "medium", "large"]:
                if size_name not in results[sys_id][detune_val]:
                    continue
                
                data = results[sys_id][detune_val][size_name]
                
                b_data = data.get('baseline', {'rms_e': [], 'rms_u': [], 'actual_p': 0})
                i_data = data.get('nn_in_integral', {'rms_e': [], 'rms_u': [], 'actual_p': 0})
                
                b_e_clean = [x for x in b_data['rms_e'] if not np.isnan(x) and not np.isinf(x)]
                b_u_clean = [x for x in b_data['rms_u'] if not np.isnan(x) and not np.isinf(x)]
                i_e_clean = [x for x in i_data['rms_e'] if not np.isnan(x) and not np.isinf(x)]
                i_u_clean = [x for x in i_data['rms_u'] if not np.isnan(x) and not np.isinf(x)]
                
                b_surv, i_surv = len(b_e_clean), len(i_e_clean)
                
                b_e_str, b_e_med = format_stats(b_e_clean)
                i_e_str, i_e_med = format_stats(i_e_clean)
                b_u_str, _ = format_stats(b_u_clean)
                i_u_str, _ = format_stats(i_u_clean)
                
                if b_surv > 0 and i_surv > 0:
                    improvement = ((b_e_med - i_e_med) / b_e_med) * 100.0
                    imp_str = f"{improvement:+.1f}%"
                else:
                    imp_str = "N/A"
                
                p_str = f"{b_data['actual_p']} / {i_data['actual_p']}"
                detune_str = f"{detune_val*100:.0f}%"
                
                b_e_full = f"{b_e_str:>18} [{b_surv:2d}/10]" if b_surv > 0 else f"{'FAILED':>18} [ 0/10]"
                i_e_full = f"{i_e_str:>18} [{i_surv:2d}/10]" if i_surv > 0 else f"{'FAILED':>18} [ 0/10]"
                b_u_full = f"{b_u_str:>22}" if b_surv > 0 else f"{'FAILED':>22}"
                i_u_full = f"{i_u_str:>22}" if i_surv > 0 else f"{'FAILED':>22}"
                
                print(f" {sys_id:<2} | {detune_str:<6} | {size_name:<9} | {p_str:<11} | {b_e_full:<31} | {i_e_full:<31} | {imp_str:>8} | {b_u_full:<22} | {i_u_full:<22}")
                
    print("="*175 + "\n")

if __name__ == "__main__":
    main()