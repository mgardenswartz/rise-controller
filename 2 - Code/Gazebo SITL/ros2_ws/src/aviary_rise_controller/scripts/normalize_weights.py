#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Strict imports from your non-editable library
from jax_resnet import resnet_network, get_total_parameters, init_resnet_weights

def sweep_and_normalize(
    dev_hidden_width: int,
    dev_b: int,
    dev_k0: int,
    dev_ki: int,
    tolerance: int
):
    """
    Sweeps the baseline network configuration space to match the total 
    trainable parameters of a target developed network configuration.
    """
    # System Dimensions (3D Acceleration Commands)
    d_out = 3 
    
    # Input Dimensions
    d_in_baseline = 12  # State vector (x)
    d_in_developed = 15 # Composite state vector (kappa)
    
    # Calculate target parameters using your exact library function
    target_params = get_total_parameters(
        d_in_developed, 
        dev_hidden_width, 
        d_out, 
        dev_b, 
        dev_k0, 
        dev_ki
    )
    
    print(f"Target Configuration (Developed Network - Input Dim: {d_in_developed}):")
    print(f"  hidden_width: {dev_hidden_width}, b: {dev_b}, k_0: {dev_k0}, k_i: {dev_ki}")
    print(f"  Total Parameter Elements: {target_params}\n")
    print(f"Searching for Baseline Network Matches (Input Dim: {d_in_baseline}) within tolerance ±{tolerance}...")
    print("-" * 90)
    print(f"{'Width':<12}{'b (blocks)':<15}{'k_0':<10}{'k_i':<10}{'Total Params':<18}{'Delta (Base - Dev)':<15}")
    print("-" * 90)
    
    # Grid search across reasonable bounds for physical deployment
    for hw in range(16, 128):
        for b in range(1, 5):
            for k_0 in range(1, 4):
                for k_i in range(1, 4):
                    # Compute baseline parameters with distinct input dimensionality
                    base_params = get_total_parameters(
                        d_in_baseline, 
                        hw, 
                        d_out, 
                        b, 
                        k_0, 
                        k_i
                    )
                    delta = base_params - target_params
                    
                    if abs(delta) <= tolerance:
                        print(f"{hw:<12}{b:<15}{k_0:<10}{k_i:<10}{base_params:<18}{delta:<15}")

if __name__ == "__main__":
    # Define your exact target Developed architecture dimensions explicitly
    TARGET_HIDDEN_WIDTH = 32
    TARGET_B = 2
    TARGET_K0 = 2
    TARGET_KI = 2
    PARAM_TOLERANCE = 20
    
    sweep_and_normalize(
        TARGET_HIDDEN_WIDTH,
        TARGET_B,
        TARGET_K0,
        TARGET_KI,
        PARAM_TOLERANCE
    )