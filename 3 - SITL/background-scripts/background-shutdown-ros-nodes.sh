#!/bin/bash
ROS_PIDS=$(pgrep -f "/opt/ros/humble" | grep -v "pgrep")
if [ -z "$ROS_PIDS" ]; then 
    echo "No ros processes running"
else
    echo "Found following PIDs for ros processes: $ROS_PIDS"
    for PID in $ROS_PIDS; do 
        echo "Terminating process $PID..."
        kill "$PID"
    done
    echo "All ros processes have been terminated."
fi 
source /opt/ros/humble/setup.bash
ros2 daemon stop
