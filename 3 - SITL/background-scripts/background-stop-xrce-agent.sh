#!/bin/bash
XRCE_CMD="MicroXRCEAgent"
XRCE_PID=$(pgrep -f "$XRCE_CMD" | grep -v "pgrep")
if [ -z "$XRCE_PID" ]; then 
    echo "MicroXRCEAgent is not running "
else
    echo "Found following PIDs for command $XRCE_CMD: $XRCE_PID"
    for PID in $XRCE_PID; do 
        echo "Terminating process $PID..."
        kill "$PID"
    done
    echo "MicroXRCEAgent has been terminated."
fi 