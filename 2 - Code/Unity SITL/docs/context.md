# Quadcopter Adaptive Control Architecture

## Overview
This repository contains a high-fidelity Simulation-in-the-Loop (SITL) environment designed to test robust nonlinear control strategies for a quadcopter experiencing unknown disturbances (e.g., wind, slung loads).

The core contribution is the evaluation of an **Integrated Neural Network (INN)** embedded directly within the integral action of a robust RISE controller, in comparison to a standard Feedforward Neural Network, a baseline RISE, and a Super-Twisting (ST) sliding mode controller.

## Target Tracking vs. Trajectory Tracking
A critical architectural choice in this repository is the focus on **Target Tracking** rather than **Trajectory Tracking**.

While the `desired_trajectory.py` module computes `qd`, `qd_dot`, and `qd_ddot`, **the controllers in this framework do not receive or use `qd_ddot` (the desired acceleration).** 
- In standard trajectory tracking, `qd_ddot` is fed into the controller to preemptively align the quadcopter's dynamics with the planned path.
- By omitting `qd_ddot`, the quadcopter treats the moving target as an exogenous signal. It must robustly respond to the instantaneous position and velocity errors (`e` and `e_dot`) without prior knowledge of the target's acceleration profile, simulating real-world target tracking (e.g., following a moving vehicle).

## Core Technologies
1. **Unity (Headless SITL)**: The physics engine (`QuadSim`) computes real-time dynamics, aerodynamics, and collisions.
2. **JAX**: Used for the Neural Network compilation and execution. JAX allows real-time, just-in-time (JIT) compiled updates to the neural network parameters during the flight control loop (50 Hz).
3. **Optuna**: A hyperparameter optimization framework used to tune control gains and network architectures via a Tree-structured Parzen Estimator (TPE).

## Flight State Machine
Every flight simulation (`SimRun`) operates in two phases:
1. **Takeoff & Settle**: The quadcopter navigates from its randomized takeoff position to a static `hover_start_z`. During this phase, integral terms are carefully managed (and optionally reset) to prevent wind-up before the tracking experiment begins.
2. **Target Tracking**: Once within `init_tol` of the hover point, the target begins moving along the desired trajectory (Figure-8 or 4-Petal Rose). The controller attempts to track the target while subject to dynamic limits and anti-windup clamping. The evaluation metric (ITAE cost) is only integrated during this phase.

## Neural Network Adaptation
For the `resnet` (Feedforward NN) and `integrated_resnet` (Integrated NN) controllers, the weights $\hat{\theta}$ are updated in real-time according to a Lyapunov-derived adaptation law. To ensure stability, the `discrete_projection` algorithm strictly confines the weights to a predefined hypersphere ($\bar{\theta}$), preventing unbounded growth.

# Script Documentation

This document explains the purpose, inputs, and outputs of the executable scripts located in the `scripts/` directory.

---

## 1. `run_all.sh`
**Purpose**: An automated bash script that executes the complete optimization pipeline across all stages.
**Usage**: `./scripts/run_all.sh`
**Details**:
- Calls `optimization.py` sequentially for Stages 1A, 1B, 2, and 3.
- Calls `extract_gains.py` at the end to compile the optimal parameters.
- Outputs all SQLite database tracking to `output/optimization.db`.

---

## 2. `optimization.py`
**Purpose**: Uses Optuna to tune hyperparameter gains and neural network architectures using mini-batch domain randomization.
**Usage**: 
```bash
python scripts/optimization.py --stage <1A|1B|2|3> --num_trials <N> --db <sqlite_url>
```
**Details**:
- **Stage 1A**: Tunes baseline RISE controller (No Wind).
- **Stage 1B**: Tunes baseline RISE controller (With Wind).
- **Stage 2**: Tunes Neural Network architecture (size, learning rate, leakage).
- **Stage 3**: Tunes Super-Twisting baseline.
- **Stage 4**: Tunes a PID baseline.
- **Output**: Writes trials to the specified Optuna SQLite database. Can be interrupted and safely resumed.

---

## 3. `extract_gains.py`
**Purpose**: Reads the Optuna SQLite database, extracts the absolute best hyperparameters from each study, and compiles them into a single YAML file.
**Usage**:
```bash
python scripts/extract_gains.py --db <sqlite_url> --config conf/config.yaml
```
**Details**:
- Automatically merges the best RISE gains with the best Neural Network architecture to form the complete parameter sets for `BEST_NN` and `BEST_INN`.
- **Output**: Generates `conf/best_gains.yaml`.

---

## 4. `evaluate_robustness.py`
**Purpose**: The final statistical validation script. Proves whether the Integrated Neural Network (INN) provides a statistically significant improvement over the baselines.
**Usage**:
```bash
python scripts/evaluate_robustness.py --num_trials 50 --config conf/config.yaml
```
**Details**:
- Loads the optimized gains from `conf/best_gains.yaml`.
- Generates a fixed array of `N` randomized takeoff conditions (varying X, Y, Z).
- Forces all controllers to fly the exact same `N` conditions to create a paired dataset.
- Runs a Shapiro-Wilk test for normality, followed by a Paired t-test or Wilcoxon Signed-Rank test.
- **Output**: Prints statistical conclusions to the terminal and saves raw paired data to `output/robustness_results.csv`.
