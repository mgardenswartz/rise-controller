#!/bin/bash

# Configuration
# 500 takes about 50 minutes with 4 mini-batches
TRIALS_PHASE_1=50 
TRIALS_PHASE_2=50
TRIALS_PHASE_3=50
PATIENCE=0
TRAJ_NUM=2
DB_DIR="output/traj${TRAJ_NUM}"

echo "=========================================="
echo " Starting Quadcopter Optimization Pipeline"
echo "=========================================="

# Ensure output directory exists
mkdir -p $DB_DIR

echo "=========================================="
echo " Starting Optimization Sweep"
echo "=========================================="
echo "Ensure you have started Unity in HEADLESS mode:"
echo "  ~/Desktop/aviary_${TRAJ_NUM}.app/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556"
echo "AND you have started PX4 SITL lockstep:"
echo "  make px4_sitl none_iris"
echo "=========================================="
read -p "Press Enter to continue once both Unity and PX4 are running..."


# Comment out any lines below to skip a specific stage

# No wind
# ~/Desktop/aviary_0.app/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556

# Trajectory 1 Fan setup
# ~/Desktop/aviary_1.app/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556

# Trajectory 2 Fan setup
# ~/Desktop/aviary_2.app/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556

# echo "[*] Running Stage 1A (RISE No Wind - Ensure wind is OFF in Unity/config)"
# python scripts/optimization.py --stage 1A --num_trials $TRIALS_PHASE_1 --db_dir $DB_DIR --patience $PATIENCE
echo "[*] WARNING: Skipping Stage 1A (No Wind) for this test."

echo "[*] Running Stage 1B (RISE With Wind - Ensure wind is ON in Unity/config)"
python scripts/optimization.py --stage 1B --num_trials $TRIALS_PHASE_1 --db_dir $DB_DIR --patience $PATIENCE

echo "[*] Running Stage 2A (Neural Network Baseline Adaptation)"
# Note: Stage 2A automatically queries stage_1B.db for its base gains!
python scripts/optimization.py --stage 2A --num_trials $TRIALS_PHASE_2 --db_dir $DB_DIR --patience $PATIENCE

echo "[*] Running Stage 2B (Integrated Neural Network Adaptation)"
# Note: Stage 2B automatically queries stage_1B.db for its base gains!
python scripts/optimization.py --stage 2B --num_trials $TRIALS_PHASE_2 --db_dir $DB_DIR --patience $PATIENCE

echo "[*] Running Stage 3 (Super-Twisting Baseline)"
python scripts/optimization.py --stage 3 --num_trials $TRIALS_PHASE_3 --db_dir $DB_DIR --patience $PATIENCE

echo "[*] Running Stage 4 (PID Controller Baseline)"
python scripts/optimization.py --stage 4 --num_trials $TRIALS_PHASE_1 --db_dir $DB_DIR --patience $PATIENCE

echo "=========================================="
echo " Extracting Best Gains..."
echo "=========================================="
python scripts/extract_gains.py --db_dir $DB_DIR --config conf/config.yaml

echo "=========================================="
echo " Pipeline Complete."
echo "=========================================="