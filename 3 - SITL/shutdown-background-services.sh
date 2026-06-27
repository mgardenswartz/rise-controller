#!/bin/bash

# Note - This script is intended to be run outside of the docker container
CONTAINER_NAME=px4-sitl-gz 

# make sure the container is running before we attempt to connect
if [ ! "$(docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. Cleanup process failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Preparing to shutdown background processes"
    echo "Shutting down XRCE Agent..."
	docker exec -i $CONTAINER_NAME bash -c "./background-scripts/background-stop-xrce-agent.sh"
	echo "Shutting down PX4 SITL instances..."
    docker exec -i $CONTAINER_NAME bash -c "./background-scripts/background-shutdown-px4-sitl.sh"
    echo "Shutting down Gazebo..."
    docker exec -i $CONTAINER_NAME bash -c "./background-scripts/background-shutdown-gz.sh"
    echo "Shutting down ROS nodes..."
    docker exec -i $CONTAINER_NAME bash -c "./background-scripts/background-shutdown-ros-nodes.sh"
fi