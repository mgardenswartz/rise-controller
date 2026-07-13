#!/bin/bash

# Configuration
TRIALS_PHASE_1=400
TRIALS_PHASE_2=500
TRIALS_PHASE_3=400
PATIENCE=150
TRAJ_NUM=1
DB_DIR="output/traj${TRAJ_NUM}"

echo "=========================================="
echo " Starting Quadcopter Optimization Pipeline"
echo "=========================================="

# Ensure output directory exists
mkdir -p $DB_DIR

# Comment out any lines below to skip a specific stage

# echo "[*] Running Stage 1A (RISE No Wind - Ensure wind is OFF in Unity/config)"
# python scripts/run_optimization.py --stage 1A --num_trials $TRIALS_PHASE_1 --db_dir $DB_DIR --patience $PATIENCE
echo "[*] WARNING: Skipping Stage 1A (No Wind) for this test."

echo "[*] Running Stage 1B (RISE With Wind - Ensure wind is ON in Unity/config)"
python scripts/run_optimization.py --stage 1B --num_trials $TRIALS_PHASE_1 --db_dir $DB_DIR --patience $PATIENCE

echo "[*] Running Stage 2 (Neural Network Adaptation)"
# Note: Stage 2 now automatically queries stage_1B.db for its base gains!
python scripts/run_optimization.py --stage 2 --num_trials $TRIALS_PHASE_2 --db_dir $DB_DIR --patience $PATIENCE

echo "[*] Running Stage 3 (Super-Twisting Baseline)"
python scripts/run_optimization.py --stage 3 --num_trials $TRIALS_PHASE_3 --db_dir $DB_DIR --patience $PATIENCE

echo "=========================================="
echo " Extracting Best Gains..."
echo "=========================================="
python scripts/extract_gains.py --db_dir $DB_DIR --config conf/config.yaml

echo "=========================================="
echo " Pipeline Complete."
echo "=========================================="