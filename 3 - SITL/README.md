INSIDE DOCKER:

cd /home/root
git clone https://github.com/mgardenswartz/resnet.git
apt update && apt install -y python3.10-venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install --upgrade pip pyyaml packaging "setuptools==69.0.3"
python3 -m pip install ./resnet
source ros-sources.sh
cd /home/root/ros2_ws
rm -rf build/ install/ log/
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/home/root/venv/bin/python3
sed -i '1s|^.*$|#!/home/root/venv/bin/python3|' install/aviary_rise_controller/lib/aviary_rise_controller/aviary_rise_controller
source install/setup.bash


ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file /home/root/ros2_ws/src/aviary_rise_controller/param/params.yaml

OUTSIDE DOCKER
pyenv local 3.12.13
pip install jax
pip install git+https://github.com/mgardenswartz/resnet.git@v1.2.0

