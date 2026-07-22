START DOCKER:
./run-px4-sim-container.sh
or connect to an existing running docker:
./connect-to-sim-container.sh

INSIDE DOCKER to build node:
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
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/home/root/venv/bin/python3 --packages-select aviary_rise_controller
sed -i '1s|^.*$|#!/home/root/venv/bin/python3|' install/aviary_rise_controller/lib/aviary_rise_controller/aviary_rise_controller

In some cases, you need to run the following (avoid in general since building px4_msgs take 8 minutes)
cd /home/root/ros2_ws
rm -rf build/ install/ log/

OUTSIDE DOCKER (to run everything)
pyenv local 3.10.14
pip install -e .
python3 scripts/unified_orchestrator.py --desired_trajectory 1 --wind
python3 scripts/unified_orchestrator.py --desired_trajectory 2 --wind

BEST RUNS (OUTSIDE DOCKER) 
python3 scripts/run_best_gains.py --controller_type baseline --desired_trajectory 1 --wind --db_dir output/traj1
python3 scripts/run_best_gains.py --controller_type resnet --desired_trajectory 1 --wind --db_dir output/traj1
python3 scripts/run_best_gains.py --controller_type integrated_resnet --desired_trajectory 1 --wind --db_dir output/traj1
python3 scripts/run_best_gains.py --controller_type pid --desired_trajectory 1 --wind --db_dir output/traj1
python3 scripts/run_best_gains.py --controller_type supertwisting --desired_trajectory 1 --wind --db_dir output/traj1

python3 scripts/run_best_gains.py --controller_type baseline --desired_trajectory 2 --wind --db_dir output/traj2
python3 scripts/run_best_gains.py --controller_type resnet --desired_trajectory 2 --wind --db_dir output/traj2
python3 scripts/run_best_gains.py --controller_type integrated_resnet --desired_trajectory 2 --wind --db_dir output/traj2
python3 scripts/run_best_gains.py --controller_type pid --desired_trajectory 2 --wind --db_dir output/traj2
python3 scripts/run_best_gains.py --controller_type supertwisting --desired_trajectory 2 --wind --db_dir output/traj2