#!/bin/bash
PX4_CMD="px4_sitl_default"
PX4_PID=$(pgrep -f "$PX4_CMD" | grep -v "pgrep")
if [ -z "$PX4_PID" ]; then 
    echo "PX4 SITL is not running "
else
    echo "Found following PIDs for command $PX4_CMD: $PX4_PID"
    for PID in $PX4_PID; do 
        echo "Terminating process $PID..."
        kill "$PID"
    done
    echo "PX4 SITL has been terminated."
fi 
