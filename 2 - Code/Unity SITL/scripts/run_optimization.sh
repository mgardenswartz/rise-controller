#!/bin/bash

# Configuration
TRIALS_PHASE_1=75
TRIALS_PHASE_2=50
TRIALS_PHASE_3=100
TRIALS_PHASE_4=50
PATIENCE=0

echo "=========================================="
echo " Starting Quadcopter Optimization Pipeline"
echo "=========================================="

# Prevent Mac from sleeping
caffeinate -i &
CAFF_PID=$!

cleanup() {
    echo "=========================================="
    echo " Cleaning up background processes..."
    echo "=========================================="
    pkill -f Unity_QuadSim
    pkill -f px4
    kill -9 $CAFF_PID 2>/dev/null
}

# Trap EXIT and SIGINT to ensure cleanup runs
trap cleanup EXIT SIGINT SIGTERM

kill_simulators() {
    pkill -f Unity_QuadSim
    pkill -f px4
    sleep 2 # wait for ports to be freed
}

start_simulators() {
    local aviary_app=$1
    echo "[*] Starting Unity: ${aviary_app}"
    ~/Desktop/${aviary_app}/Contents/MacOS/Unity_QuadSim -batchmode -nographics -rpcPort 5555 -telemetryPort 5556 &
    
    echo "[*] Starting PX4 SITL lockstep"
    pushd ~/PX4-Autopilot >/dev/null
    make px4_sitl none_iris &
    popd >/dev/null
    
    echo "[*] Waiting 10 seconds for ports to bind..."
    sleep 10
}

run_stage() {
    local stage=$1
    local trials=$2
    local db_dir=$3
    
    echo "[*] Running Stage ${stage}"
    python scripts/optimization.py --stage ${stage} --num_trials ${trials} --db_dir ${db_dir} --patience ${PATIENCE}
    if [ $? -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo " ❌ ERROR DETECTED"
        echo "=========================================="
        echo "Stage ${stage} failed during optimization for Trajectory ${TRAJ_NUM}."
        echo "The drone may have crashed and PX4 failed to recover."
        echo "Check the console output above to see exactly which gains were being tested when the failure occurred."
        echo "=========================================="
        exit 1
    fi
}

# Source the python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment 'venv' not found!"
    exit 1
fi

for TRAJ_NUM in 1 2; do
    echo "=========================================="
    echo " BEGINNING OPTIMIZATION FOR TRAJECTORY ${TRAJ_NUM}"
    echo "=========================================="
    
    DB_DIR="output/traj${TRAJ_NUM}"
    mkdir -p $DB_DIR
    
    # Update config.yaml with correct trajectory
    sed -i '' "s/desired_trajectory: [0-9]/desired_trajectory: ${TRAJ_NUM}/g" conf/config.yaml
    
    # ==========================
    # Phase 1: No Wind (Stage 1A)
    # ==========================
    echo "=========================================="
    echo " Trajectory ${TRAJ_NUM} - NO WIND ENVIRONMENT"
    echo "=========================================="
    kill_simulators
    start_simulators "aviary_0.app"
    
    run_stage "1A" $TRIALS_PHASE_1 $DB_DIR
    
    # ==========================
    # Phase 2: Wind Environment
    # ==========================
    echo "=========================================="
    echo " Trajectory ${TRAJ_NUM} - WIND ENVIRONMENT"
    echo "=========================================="
    kill_simulators
    start_simulators "aviary_${TRAJ_NUM}.app"
    
    run_stage "1B" $TRIALS_PHASE_1 $DB_DIR
    
    echo "[*] Running Stage 2A (Neural Network Baseline Adaptation)"
    run_stage "2A" $TRIALS_PHASE_2 $DB_DIR
    
    echo "[*] Running Stage 2B (Integrated Neural Network Adaptation)"
    run_stage "2B" $TRIALS_PHASE_2 $DB_DIR
    
    echo "[*] Running Stage 3 (Super-Twisting Baseline)"
    run_stage "3" $TRIALS_PHASE_3 $DB_DIR
    
    echo "[*] Running Stage 4 (PID Controller Baseline)"
    run_stage "4" $TRIALS_PHASE_4 $DB_DIR
    
    echo "=========================================="
    echo " Extracting Best Gains for Trajectory ${TRAJ_NUM}..."
    echo "=========================================="
    python scripts/extract_gains.py --db_dir $DB_DIR --config conf/config.yaml
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: extract_gains.py failed for Trajectory ${TRAJ_NUM}."
        exit 1
    fi

done

echo "=========================================="
echo " ✅ Pipeline Complete for all trajectories!"
echo "=========================================="