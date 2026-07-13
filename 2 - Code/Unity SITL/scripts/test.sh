#!/bin/bash

# Configuration - Reduced trials for a test run
TRIALS_PHASE_1=5
TRIALS_PHASE_2=5
TRIALS_PHASE_3=5

# Dynamically route the output based on trajectory
TRAJ_NUM=1
DB_DIR="output/traj${TRAJ_NUM}"

echo "=========================================="
echo " Starting Quadcopter Test Pipeline"
echo "=========================================="
echo "Ensure you have started Unity:"
echo "GUI Mode (Recommended for testing):"
echo "  ~/Desktop/aviary_${TRAJ_NUM}.app/Contents/MacOS/Unity_QuadSim -rpcPort 5555 -telemetryPort 5556"
echo "Headless Mode (For fast optimization):"
echo "  ~/Desktop/aviary_${TRAJ_NUM}.app/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556"
echo "=========================================="
read -p "Press Enter to continue once Unity is running..."

# Ensure output directory exists
mkdir -p $DB_DIR

echo "[*] Skipping Stage 1A (No Wind) for this test."

echo "[*] Running Stage 1B (RISE With Wind)"
python scripts/run_optimization.py --stage 1B --num_trials $TRIALS_PHASE_1 --db_dir $DB_DIR --patience 0

echo "[*] Running Stage 2A (Neural Network Baseline Adaptation)"
python scripts/run_optimization.py --stage 2A --num_trials $TRIALS_PHASE_2 --db_dir $DB_DIR --patience 0

echo "[*] Running Stage 2B (Integrated Neural Network Adaptation)"
python scripts/run_optimization.py --stage 2B --num_trials $TRIALS_PHASE_2 --db_dir $DB_DIR --patience 0

echo "[*] Running Stage 3 (Super-Twisting Baseline)"
python scripts/run_optimization.py --stage 3 --num_trials $TRIALS_PHASE_3 --db_dir $DB_DIR --patience 0

echo "=========================================="
echo " Extracting Best Gains..."
echo "=========================================="
# NOTE: If your extract_gains script reads conf/config.yaml for best_gains_path, 
# you might want to save the test gains to a different file, or let it overwrite.
python scripts/extract_gains.py --db_dir $DB_DIR --config conf/config.yaml

echo "=========================================="
echo " Test Pipeline Complete."
echo "=========================================="
