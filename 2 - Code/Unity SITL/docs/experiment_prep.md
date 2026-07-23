# PREPARING FOR A REAL-WORLD EXPERIMENT

Generate params for the quadcopter.

``
GAZEBO=false
python scripts/generate_hardware_params.py --best_gains output/traj1/best_gains.yaml --controller_type pid --out output/hardware_params/pid_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains output/traj1/best_gains.yaml --controller_type integrated_resnet --out output/hardware_params/integrated_resnet_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains output/traj1/best_gains.yaml --controller_type resnet --out output/hardware_params/resnet_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains output/traj1/best_gains.yaml --controller_type baseline --out output/hardware_params/baseline_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains output/traj1/best_gains.yaml --controller_type st --out output/hardware_params/st_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1

python scripts/generate_hardware_params.py --best_gains output/traj2/best_gains.yaml --controller_type pid --out output/hardware_params/pid_params_2.yaml --gazebo $GAZEBO --desired_trajectory 2
python scripts/generate_hardware_params.py --best_gains output/traj2/best_gains.yaml --controller_type integrated_resnet --out output/hardware_params/integrated_resnet_params_2.yaml --gazebo $GAZEBO --desired_trajectory 2
python scripts/generate_hardware_params.py --best_gains output/traj2/best_gains.yaml --controller_type resnet --out output/hardware_params/resnet_params_2.yaml --gazebo $GAZEBO --desired_trajectory 2
python scripts/generate_hardware_params.py --best_gains output/traj2/best_gains.yaml --controller_type baseline --out output/hardware_params/baseline_params_2.yaml --gazebo $GAZEBO --desired_trajectory 2
python scripts/generate_hardware_params.py --best_gains output/traj2/best_gains.yaml --controller_type st --out output/hardware_params/st_params_2.yaml --gazebo $GAZEBO --desired_trajectory 2
``

Push latest node to the quad.
``
PATH_TO_MY_REPO="$HOME/Documents/GitHub/rise-controller"
adb push "$PATH_TO_MY_REPO/2 - Code/Gazebo SITL/ros2_ws/src/aviary_rise_controller/aviary_rise_controller/aviary_rise_node.py" /home/root/humble_ws/src/aviary_rise_controller/aviary_rise_controller/aviary_rise_node.py
adb push "$PATH_TO_MY_REPO/2 - Code/Gazebo SITL/ros2_ws/src/aviary_rise_controller/aviary_rise_controller/proj.py" /home/root/humble_ws/src/aviary_rise_controller/aviary_rise_controller/proj.py
adb push "$PATH_TO_MY_REPO/2 - Code/Gazebo SITL/ros2_ws/src/aviary_rise_controller/aviary_rise_controller/desired_trajectory.py" /home/root/humble_ws/src/aviary_rise_controller/aviary_rise_controller/desired_trajectory.py
``

Push latest params to the quad.
``
PATH_TO_MY_REPO="$HOME/Documents/GitHub/rise-controller"
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/baseline_params_1.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/baseline_params_1.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/resnet_params_1.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/resnet_params_1.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/integrated_resnet_params_1.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/integrated_resnet_params_1.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/pid_params_1.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/pid_params_1.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/st_params_1.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/st_params_1.yaml

adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/baseline_params_2.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/baseline_params_2.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/resnet_params_2.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/resnet_params_2.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/integrated_resnet_params_2.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/integrated_resnet_params_2.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/pid_params_2.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/pid_params_2.yaml
adb push "$PATH_TO_MY_REPO/2 - Code/Unity SITL/output/hardware_params/st_params_2.yaml" /home/root/humble_ws/src/aviary_rise_controller/param/st_params_2.yaml
``

Connect the quad to your computer via USB and run the following.

First, get the latest time. shell in and run Docker.
``
adb shell
systemctl restart chrony
``

Second, start and connect to Docker.
``
cd /home/root
bash docker-scripts/run-ros-docker.sh
``

Then setup the necessary Python environment. WiFi required.
``
git clone https://github.com/mgardenswartz/resnet.git
apt update && apt install -y python3.10-venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
source /home/root/hyrl/hyrl_ros_sources.sh
python3 -m pip install --upgrade pip
python3 -m pip install "setuptools==58.2.0"
python3 -m pip install jax pandas ./resnet
``

Build the node
``
cd /home/root/humble_ws
rm -rf build/aviary_rise_controller
colcon build --symlink-install --packages-select aviary_rise_controller --cmake-args -DPython3_EXECUTABLE=/home/root/venv/bin/python3
sed -i '1s|^.*$|#!/home/root/venv/bin/python3|' install/aviary_rise_controller/lib/aviary_rise_controller/aviary_rise_controller
source /home/root/hyrl/hyrl_ros_sources.sh
``

To run the experiment:
``
DESIRED_TRAJECTORY=1
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/humble_ws/src/aviary_rise_controller/param/baseline_params_${DESIRED_TRAJECTORY}.yaml"
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/humble_ws/src/aviary_rise_controller/param/resnet_params_${DESIRED_TRAJECTORY}.yaml"
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/humble_ws/src/aviary_rise_controller/param/integrated_resnet_params_${DESIRED_TRAJECTORY}.yaml"
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/humble_ws/src/aviary_rise_controller/param/pid_params_${DESIRED_TRAJECTORY}.yaml"
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/humble_ws/src/aviary_rise_controller/param/st_params_${DESIRED_TRAJECTORY}.yaml"
``