#!/bin/bash
# Note - This script is intended to be run outside of the docker container
CONTAINER_NAME=px4-sitl-gz 

# make sure the container is running before we attempt to connect
if [ ! "$(docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. Gazebo set up failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Preparing to run gazebo"
    echo 
    # only log the first 100 lines of output so we don't end up with massive log files
	docker exec -e PX4_GZ_WORLD -e GZ_SEED -d $CONTAINER_NAME bash -c "./run-gz-sim.sh 2>&1 | tee >(head -n 200 > /tmp/gz-sim-output.log) > /dev/null "
	#docker exec -d $CONTAINER_NAME bash -c "./run-gz-sim.sh 2>&1 | tee >(head -n 200 > /tmp/gz-sim-output.log) > /dev/null "
	sleep 5
	# docker exec -i $CONTAINER_NAME bash -c "cat /tmp/gz-sim-output.log"
	# docker exec -i $CONTAINER_NAME bash -c "cat /tmp/gz-sim-output.log"
fi