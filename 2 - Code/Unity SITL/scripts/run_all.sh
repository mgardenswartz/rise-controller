#!/bin/bash

# Configuration
TRIALS_PHASE_1=300
TRIALS_PHASE_2=750
TRIALS_PHASE_3=300
DB_DIR="output"

echo "=========================================="
echo " Starting Quadcopter Optimization Pipeline"
echo "=========================================="

# Ensure output directory exists
mkdir -p $DB_DIR

# Comment out any lines below to skip a specific stage

echo "[*] Running Stage 1A (RISE No Wind - Ensure wind is OFF in Unity/config)"
python scripts/run_optimization.py --stage 1A --num_trials $TRIALS_PHASE_1 --db sqlite:///$DB_DIR/stage_1A.db

echo "[*] Running Stage 1B (RISE With Wind - Ensure wind is ON in Unity/config)"
python scripts/run_optimization.py --stage 1B --num_trials $TRIALS_PHASE_1 --db sqlite:///$DB_DIR/stage_1B.db

echo "[*] Running Stage 2 (Neural Network Adaptation)"
# Note: Manually update config.yaml with Phase 1B gains and set controller_type before this runs!
python scripts/run_optimization.py --stage 2 --num_trials $TRIALS_PHASE_2 --db sqlite:///$DB_DIR/stage_2.db

echo "[*] Running Stage 3 (Super-Twisting Baseline)"
python scripts/run_optimization.py --stage 3 --num_trials $TRIALS_PHASE_3 --db sqlite:///$DB_DIR/stage_3.db

echo "=========================================="
echo " Extracting Best Gains..."
echo "=========================================="
python scripts/extract_gains.py --db_dir $DB_DIR --config conf/config.yaml

echo "=========================================="
echo " Pipeline Complete."
echo "=========================================="