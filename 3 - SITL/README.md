INSIDE DOCKER:
cd /home/root
git clone https://github.com/mgardenswartz/resnet.git
python3 -m pip install --upgrade pip
python3 -m pip install "setuptools==58.2.0"
python3 -m pip install jax ./resnet
cd /home/root/ros2_ws
rm -rf build/ install/ log/
colcon build --symlink-install
source /home/root/ros-sources.sh

MACOS DOCKER CONTAINER (ghcr.io/ufl-autonomy-park/apark-ros2:mac) NEEDS:
apt-get update
apt-get install -y ros-humble-ros-gz

OUTSIDE DOCKER:
pyenv local 3.10.12
pip install jax pandas matplotlib
pip install git+https://github.com/mgardenswartz/resnet.git

THE MONTE CARLO SCRIPT (OUTSIDE DOCKER)
python3 unified_orchestrator.py --controller_type noresnet --active_trajectory 1 --wind

EXPERIMENT TIME
python3 generate_hardware_params.py --controller developed --desired_trajectory 1 --out baseline_params.yaml
python3 generate_hardware_params.py --controller baseline --desired_trajectory 1 --out developed_params.yaml
python3 generate_hardware_params.py --controller noresnet --desired_trajectory 1 --out noresnet_params.yaml

adb push ~/voxl-px4-sitl/ros2_ws/src/aviary_rise_controller/aviary_rise_controller/aviary_rise_node.py /home/root/humble_ws/src/aviary_rise_controller/aviary_rise_controller/aviary_rise_node.py
adb push ~/voxl-px4-sitl/noresnet_params.yaml /home/root/humble_ws/src/aviary_rise_controller/param/noresnet_params.yaml
adb push ~/voxl-px4-sitl/baseline_params.yaml /home/root/humble_ws/src/aviary_rise_controller/param/baseline_params.yaml
adb push ~/voxl-px4-sitl/developed_params.yaml /home/root/humble_ws/src/aviary_rise_controller/param/developed_params.yaml

ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file /home/root/ros2_ws/src/aviary_rise_controller/param/noresnet_params.yaml
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file /home/root/ros2_ws/src/aviary_rise_controller/param/baseline_params.yaml
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file /home/root/ros2_ws/src/aviary_rise_controller/param/developed_params.yaml