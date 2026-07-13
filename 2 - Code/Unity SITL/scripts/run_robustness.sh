#!/bin/bash

# Configuration
NUM_TRIALS=20
TRAJ_NUM=1
DB_DIR="output/traj${TRAJ_NUM}"

echo "=========================================="
echo " Starting Monte Carlo Robustness Sweep"
echo "=========================================="
echo "Ensure you have started Unity in HEADLESS mode:"
echo "  ~/Desktop/aviary_${TRAJ_NUM}.app/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556"
echo "=========================================="
read -p "Press Enter to continue once Unity is running..."

echo "[*] Running Robustness Sweep on $DB_DIR/best_gains.yaml..."
python scripts/evaluate_robustness.py --num_trials $NUM_TRIALS --config conf/config.yaml --db_dir $DB_DIR

echo "=========================================="
echo " Robustness Evaluation Complete."
echo "=========================================="
