# DNN Adaptive Controller

## Overview
This repository contains a high-performance, JAX-accelerated simulation framework for Lyapunov-based adaptive control of nonlinear systems. It leverages Deep Neural Networks (DNNs) to approximate unknown continuous dynamics online. The framework is designed for rigorous comparative analysis of neural architectures (deep vs. shallow) and features built-in hyperparameter optimization via Hydra and Optuna. It is built to validate control strategies before deploying them to physical hardware, such as quadcopter fleets or ground robots.

## Repository Structure
```text
├── conf/
│   └── config.yaml                 # Master configuration (Hydra)
├── scripts/
│   ├── animate_runs_3d.py          # Animates the NN's learning reconstruction magnitude over time
│   ├── animate_runs.py             # Decoupled 3D/2D trajectory animator
│   ├── architecture_matcher.py     # Utility to map depth/width to parameter count
│   ├── rank_runs.py                # Parses and ranks multirun Optuna statistics
│   └── robustness_mc.py            # Monte Carlo initial condition testing
├── src/
│   ├── core/                       # Config schemas and dataclasses
│   ├── io/                         # Data exporting, plotting, and statistics
│   ├── math/                       # Dynamics, NN architectures, and update laws
│   └── simulation/                 # Diffrax ODE runner and JAX execution
├── main.py                         # Primary entry point / Objective function
└── pyproject.toml                  # Single source of truth for dependencies
```

## Quickstart Guide

### Prerequisites
* **Python 3.12+** (Developed and tested on 3.12.8)
* All dependencies are strictly managed via `pyproject.toml`.

### Installation
For a clean installation, build a virtual environment and install the package locally. The Hydra-Optuna sweeper is included in the project dependencies.


# Optional: Teardown existing corrupted environments
```bash
deactivate && rm -rf venv
```

# 1. Set specific Python version
```bash
pyenv local 3.12.8
```

# 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

# 3. Upgrade pip and install project
```bash
pip install --upgrade pip
pip install -e .
```

If using a CUDA platform, use
```bash
pip install --upgrade pip
pip install -e ".[cuda12]"
```
Likewise for CUDA 13:
```bash
pip install --upgrade pip
pip install -e ".[cuda13]"
```
or Apple Silicon (not tested)
```bash
pip install --upgrade pip
pip install -e ".[apple]"
```

---

## Usage Workflows

This framework relies on **Hydra** for configuration management. You do not need to edit Python files to change simulation parameters; simply override them from the command line.

### 1. Standard Simulation
Run a single 60-second trial using the default parameters defined in `conf/config.yaml`. Data and figures will be saved to a timestamped directory in `outputs/`.

```bash
python main.py
```

### 2. Debugging (JAX JIT Disabled)
If you are developing new mathematical update laws and need to trace matrix dimensions or NaN values step-by-step, disable the XLA compiler to force eager execution:

```bash
JAX_DISABLE_JIT=1 python main.py
```

### 3. Hyperparameter Sweeps (Optuna)
To run Bayesian optimization using the Tree-structured Parzen Estimator (TPE), use the `-m` (multirun) flag. Optuna will automatically minimize the post-excitation tracking error.

**Sweep Control Gains (Continuous Intervals):**
```bash
python main.py -m math_constants.k_e="interval(7.5, 15.0)" math_constants.k_theta_hat="interval(0.0001, 0.05)"
```

**Sweep Neural Architectures (Discrete Choices):**
*Note: Always use `choice()` for structural network parameters to prevent type errors.*
```bash
python main.py -m neural_network.num_layers="choice(1,2,3,4)" neural_network.hidden_width="choice(4,8)"
```

If you want to disable the default 90% allocation of RAM and do parallelization, use:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python main.py -m ... [your sweep args] ... hydra.sweeper.n_jobs=4
```

---

## Post-Processing & Analysis

To maintain maximum JAX compilation speeds, visualizations and metric aggregations are decoupled from the main simulation loop. 

### Ranking Optuna Sweeps
After executing a multirun, you can parse the generated `statistics.json` files to view a ranked leaderboard of all trials based on tracking error, control effort, and FLOPs.

```bash
python scripts/rank_runs.py --dir multirun/2026-02-21/17-36-54
```

### Animating Trajectories
Generate a real-time `.mp4` video mapping the system's actual trajectory against the desired path.

```bash
python scripts/animate_runs.py --dir outputs/2026-02-21/17-41-12/
```

### Architecture Parameter Matching
When comparing deep vs. shallow networks, use this utility to mathematically generate depths and widths that result in an equivalent total trainable parameter count (P):

```bash
python scripts/architecture_matcher.py --p 100 --d_in 2 --d_out 2 --max_depth 5
```
