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

TO RUN THE NODE ONCE (INSIDE DOCKER; FOR REAL-WORLD EXPERIMENTS ONLY; NO GAZEBO)
ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file /home/root/ros2_ws/src/aviary_rise_controller/param/params.yaml

OUTSIDE DOCKER:
pyenv local 3.10.12
pip install jax
pip install git+https://github.com/mgardenswartz/resnet.git

THE MONTE CARLO SCRIPT (OUTSIDE DOCKER)
python3 unified_orchestrator.py --controller_type noresnet --active_trajectory 1 --wind
