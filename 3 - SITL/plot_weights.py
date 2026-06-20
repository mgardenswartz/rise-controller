import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax_resnet import init_resnet_weights

def plot_weight_spread():
    key = jax.random.PRNGKey(42)
    
    # Common parameters
    hidden_width = 16
    d_out = 3
    num_blocks = 1
    k_0 = 1
    k_i = 1
    h_method = 'xavier'
    o_method = 'he'
    
    # 1. Generate Baseline Weights (d_in = 12)
    w_base = 0.2 * init_resnet_weights(key, 12, hidden_width, d_out, num_blocks, k_0, k_i, h_method, o_method)
    w_base_np = np.array(w_base)
    
    # 2. Generate Developed Weights (d_in = 15)
    w_dev = 0.2 * init_resnet_weights(key, 15, hidden_width, d_out, num_blocks, k_0, k_i, h_method, o_method)
    w_dev_np = np.array(w_dev)
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    fig.suptitle(f"ResNet Initial Weight Distribution (Width={hidden_width}, Blocks={num_blocks})", fontsize=16, fontweight='bold')
    
    bins = np.linspace(-1.5, 1.5, 50)
    
    # Baseline Plot
    axes[0].hist(w_base_np, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title(f"Baseline Controller (d_in=12)\nTotal Params: {len(w_base_np)}")
    axes[0].set_xlabel("Weight Value")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)
    
    # Add text box with stats
    stats_base = f"Mean: {w_base_np.mean():.4f}\nStd: {w_base_np.std():.4f}\nMax: {w_base_np.max():.4f}\nL2 Norm: {np.linalg.norm(w_base_np):.2f}"
    axes[0].text(0.65, 0.85, stats_base, transform=axes[0].transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Developed Plot
    axes[1].hist(w_dev_np, bins=bins, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title(f"Developed Controller (d_in=15)\nTotal Params: {len(w_dev_np)}")
    axes[1].set_xlabel("Weight Value")
    axes[1].grid(True, alpha=0.3)
    
    # Add text box with stats
    stats_dev = f"Mean: {w_dev_np.mean():.4f}\nStd: {w_dev_np.std():.4f}\nMax: {w_dev_np.max():.4f}\nL2 Norm: {np.linalg.norm(w_dev_np):.2f}"
    axes[1].text(0.65, 0.85, stats_dev, transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("weight_initialization_spread.png", dpi=300)
    print("[*] Saved visualization to weight_initialization_spread.png")

if __name__ == "__main__":
    plot_weight_spread()