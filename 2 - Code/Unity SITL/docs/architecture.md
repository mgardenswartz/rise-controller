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
For the `baseline` (Feedforward NN) and `developed` (Integrated NN) controllers, the weights $\hat{\theta}$ are updated in real-time according to a Lyapunov-derived adaptation law. To ensure stability, the `discrete_projection` algorithm strictly confines the weights to a predefined hypersphere ($\bar{\theta}$), preventing unbounded growth.
