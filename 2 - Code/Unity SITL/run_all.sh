#!/bin/bash

# Configuration
TRIALS_PHASE_1=300
TRIALS_PHASE_2=750
TRIALS_PHASE_3=300
DB_URL="sqlite:///optimization.db"

echo "=========================================="
echo " Starting Quadcopter Optimization Pipeline"
echo "=========================================="

# Comment out any lines below to skip a specific stage

echo "[*] Running Stage 1A (RISE No Wind - Ensure wind is OFF in Unity/config)"
python3 run_optimization.py --stage 1A --trials $TRIALS_PHASE_1 --db $DB_URL

echo "[*] Running Stage 1B (RISE With Wind - Ensure wind is ON in Unity/config)"
python3 run_optimization.py --stage 1B --trials $TRIALS_PHASE_1 --db $DB_URL

echo "[*] Running Stage 2 (Neural Network Adaptation)"
# Note: Manually update config.yaml with Phase 1B gains and set controller_type before this runs!
python3 run_optimization.py --stage 2 --trials $TRIALS_PHASE_2 --db $DB_URL

echo "[*] Running Stage 3 (Super-Twisting Baseline)"
python3 run_optimization.py --stage 3 --trials $TRIALS_PHASE_3 --db $DB_URL

echo "=========================================="
echo " Pipeline Complete."
echo "=========================================="