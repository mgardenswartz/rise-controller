#!/bin/bash 

WORLD_FILE="${PX4_GZ_WORLD:-default.sdf}"

# Export paths to gazebo worlds, models, AND our custom compiled plugins
export GZ_SIM_RESOURCE_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models:/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/worlds"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/home/root/ros2_ws/install/aviary_wind_plugin/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_VERSION=harmonic

gz sim -r -s -v 4 --seed ${GZ_SEED:-42} "$WORLD_FILE"