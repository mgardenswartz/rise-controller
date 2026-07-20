START DOCKER:
./run-px4-sim-container.sh

INSIDE DOCKER (GAZEBO):
cd /home/root
git clone https://github.com/mgardenswartz/resnet.git
apt update && apt install -y python3.10-venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
source /home/root/ros-sources.sh
python3 -m pip install --upgrade pip
python3 -m pip install "setuptools==58.2.0"
python3 -m pip install jax pandas ./resnet
cd /home/root/ros2_ws
apt update
apt-get install -y ros-humble-ros-gz
apt upgrade -y
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/home/root/venv/bin/python3
sed -i '1s|^.*$|#!/home/root/venv/bin/python3|' install/aviary_rise_controller/lib/aviary_rise_controller/aviary_rise_controller

In some cases, you need to run the following:
cd /home/root/ros2_ws
rm -rf build/ install/ log/

OUTSIDE DOCKER:
pyenv local 3.10.12
pip install jax pandas matplotlib
pip install git+https://github.com/mgardenswartz/resnet.git

THE MONTE CARLO SCRIPT (OUTSIDE DOCKER)
python3 unified_orchestrator.py --controller_type baseline --desired_trajectory 1 --wind

BEST RUNS (OUTSIDE DOCKER)
python3 run_best_gains.py --desired_trajectory 1 --wind