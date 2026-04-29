import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

CONTROLLERS = ["direct_e", "direct_r", "integral_e", "integral_r"]
CTRL_ABBR   = {"direct_e": "dir_e", "direct_r": "dir_r", "integral_e": "int_e", "integral_r": "int_r"}
REFERENCE   = "direct_e"


def fmt_cell(med, surv, total):
    if surv == 0:
        return f"{'FAILED':>8} [--/{total:2d}]"
    return f"{med:>8.4f} [{surv:2d}/{total:2d}]"


def main():
    base_dir = Path("outputs/unified_sweep")
    if not base_dir.exists():
        print("No sweep data found in outputs/unified_sweep.")
        return

    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {"rms_e": [], "rms_u": [], "actual_p": 0}
                )
            )
        )
    )

    dir_pattern = re.compile(r"sys_(\d+)_detune_([0-9.]+)")

    for sys_dir in base_dir.iterdir():
        if not sys_dir.is_dir():
            continue
        match = dir_pattern.match(sys_dir.name)
        if not match:
            continue
        sys_id    = int(match.group(1))
        detune_val = float(match.group(2))

        for ctrl_dir in sys_dir.iterdir():
            if not ctrl_dir.is_dir():
                continue
            ctrl = ctrl_dir.name
            if ctrl not in CONTROLLERS:
                continue

            for size_dir in ctrl_dir.iterdir():
                if not size_dir.is_dir():
                    continue
                size_name = size_dir.name

                for seed_dir in size_dir.glob("seed_*"):
                    stat_file = seed_dir / "statistics.json"
                    if not stat_file.exists():
                        continue
                    with open(stat_file, "r") as f:
                        stats = json.load(f)
                    results[sys_id][detune_val][size_name][ctrl]["rms_e"].append(
                        stats.get("rms_tracking_error_norm", np.nan)
                    )
                    results[sys_id][detune_val][size_name][ctrl]["rms_u"].append(
                        stats.get("rms_control_input_norm", np.nan)
                    )
                    results[sys_id][detune_val][size_name][ctrl]["actual_p"] = int(
                        stats.get("total_trainable_parameters", 0)
                    )

    if not results:
        print("No valid detune directories parsed. Check your outputs folder.")
        return

    # Build header
    cell_w = 16  # "  0.1234 [10/10]" = 16 chars
    ctrl_header = " | ".join(f"{CTRL_ABBR[c]:^{cell_w}}" for c in CONTROLLERS)
    header = f"{'Sys':<3} | {'Detune':<6} | {'Arch':<9} | {'P(dir/int)':<11} | {ctrl_header} | {'Best%Imp':>8}"
    sep    = "=" * len(header)

    print("\n" + sep)
    print(header)
    print("-" * len(header))

    for sys_id in sorted(results.keys()):
        for detune_val in sorted(results[sys_id].keys(), reverse=True):
            for size_name in ["small", "medium", "large"]:
                size_data = results[sys_id][detune_val].get(size_name)
                if size_data is None:
                    continue

                medians, survs, cells = {}, {}, {}
                for ctrl in CONTROLLERS:
                    cd    = size_data.get(ctrl, {"rms_e": [], "actual_p": 0})
                    total = len(cd["rms_e"])
                    clean = [x for x in cd["rms_e"] if np.isfinite(x)]
                    med   = float(np.median(clean)) if clean else np.nan
                    medians[ctrl] = med
                    survs[ctrl]   = len(clean)
                    cells[ctrl]   = fmt_cell(med, len(clean), total)

                # Best % improvement over REFERENCE
                ref_med = medians[REFERENCE]
                best_imp = np.nan
                if np.isfinite(ref_med) and survs[REFERENCE] > 0:
                    for ctrl in CONTROLLERS:
                        if ctrl == REFERENCE:
                            continue
                        if np.isfinite(medians[ctrl]) and survs[ctrl] > 0:
                            imp = (ref_med - medians[ctrl]) / ref_med * 100.0
                            if np.isnan(best_imp) or imp > best_imp:
                                best_imp = imp
                imp_str = f"{best_imp:+.1f}%" if np.isfinite(best_imp) else "N/A"

                # Param counts: direct and integral groups share counts within group
                d_p = size_data.get("direct_e", size_data.get("direct_r", {})).get("actual_p", 0)
                i_p = size_data.get("integral_e", size_data.get("integral_r", {})).get("actual_p", 0)
                p_str      = f"{d_p}/{i_p}"
                detune_str = f"{detune_val*100:.0f}%"

                ctrl_cells = " | ".join(f"{cells[c]:>{cell_w}}" for c in CONTROLLERS)
                print(f" {sys_id:<2} | {detune_str:<6} | {size_name:<9} | {p_str:<11} | {ctrl_cells} | {imp_str:>8}")

    print(sep + "\n")


if __name__ == "__main__":
    main()
