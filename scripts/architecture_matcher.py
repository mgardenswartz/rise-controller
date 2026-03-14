import argparse
import numpy as np

def get_block_parameters(d_in: int, hidden_width: int, d_out: int, k: int) -> int:
    if k == 0:
        return (d_in + 1) * d_out
    p_in = (d_in + 1) * hidden_width
    p_hidden = (k - 1) * (hidden_width + 1) * hidden_width if k > 0 else 0
    p_out = (hidden_width + 1) * d_out
    return p_in + p_hidden + p_out

def get_total_parameters(d_in: int, hidden_width: int, d_out: int, b: int, k_0: int, k_i: int) -> int:
    total = get_block_parameters(d_in, hidden_width, d_out, k_0)
    total += b * get_block_parameters(d_out, hidden_width, d_out, k_i)
    return total

def find_best_architecture(target_p: int, d_in: int, d_out: int) -> dict:
    best_diff = float('inf')
    best_config = None
    
    # Grid search over reasonable ResNet topological bounds
    for w in range(2, 64):
        for b in range(0, 10):
            for k_0 in range(1, 5):
                for k_i in range(1, 5):
                    p = get_total_parameters(d_in, w, d_out, b, k_0, k_i)
                    diff = abs(p - target_p)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_config = {
                            "d_in": d_in, "d_out": d_out, "hidden_width": w,
                            "b": b, "k_0": k_0, "k_i": k_i, "actual_p": p
                        }
                    
                    # Exact match short-circuit
                    if diff == 0:
                        return best_config
                        
    return best_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_p", type=int, required=True)
    parser.add_argument("--d_out", type=int, default=2)
    args = parser.parse_args()

    # Assuming x is R^2, and kappa = [x, x_d] in R^4
    config_outside = find_best_architecture(args.target_p, 2, args.d_out)
    config_inside = find_best_architecture(args.target_p, 4, args.d_out)

    print(f"\n--- TARGET PARAMETERS: {args.target_p} ---")
    print(f"OUTSIDE INTEGRAL (d_in=2) -> Actual P: {config_outside['actual_p']}")
    print(f"YAML: b={config_outside['b']}, k_0={config_outside['k_0']}, k_i={config_outside['k_i']}, hidden_width={config_outside['hidden_width']}")
    
    print(f"\nINSIDE INTEGRAL (d_in=4)  -> Actual P: {config_inside['actual_p']}")
    print(f"YAML: b={config_inside['b']}, k_0={config_inside['k_0']}, k_i={config_inside['k_i']}, hidden_width={config_inside['hidden_width']}")