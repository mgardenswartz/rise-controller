#!/usr/bin/env python3
"""Position PID driving PX4 acceleration setpoints in QuadSim lockstep."""

import math
from pathlib import Path

import numpy as np
import yaml

from quadsim import QuadSim
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file

from desired_trajectory import TrajectoryGenerator
from pid_controller import AccelerationPid


RISEMAX_DIR = Path(__file__).resolve().parent
CONFIG_FILE = RISEMAX_DIR / "config.yaml"


def make_trajectory(config, center_down):
    trajectory_config = config["trajectory"]
    figure8_config = trajectory_config["figure8"]
    rose_config = trajectory_config["rose"]
    trajectory_id = (
        TrajectoryGenerator.FIGURE_EIGHT
        if trajectory_config["type"] == "figure8"
        else TrajectoryGenerator.ROSE
    )
    return TrajectoryGenerator(
        {
            "desired_trajectory": trajectory_id,
            "run_length_s": trajectory_config["duration_s"],
            "traj1_center_z_m_ned_aviary": center_down,
            "traj1_period_s": figure8_config["period_s"],
            "traj1_x_amp_m_ned_aviary": figure8_config["north_amplitude_m"],
            "traj1_y_amp_m_ned_aviary": figure8_config["east_amplitude_m"],
            "traj1_z_amp_m_ned_aviary": figure8_config["down_amplitude_m"],
            "traj1_alpha_warp": figure8_config["warp"],
            "traj2_center_z_m_ned_aviary": center_down,
            "traj2_petal_radius_m": rose_config["radius_m"],
            "traj2_target_speed_mps": rose_config["speed_mps"],
        }
    )


def load_config():
    with CONFIG_FILE.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def run(config):
    quadsim_config = config["quadsim"]
    px4_config = config["px4"]
    control_config = config["control"]
    trajectory_config = config["trajectory"]
    yaw_ned = control_config["yaw_ned_rad"]

    with QuadSim(
        host=quadsim_config["host"],
        command_port=quadsim_config["port"],
        telemetry_port=quadsim_config["port"] + 1,
    ) as sim:
        bridge = HilBridge(
            px4_host=px4_config["host"],
            instance=px4_config["instance"],
            px4_client=px4_config["client"],
        )
        px4 = Px4Link(stream_hz=0, instance=px4_config["instance"])
        runner = LockstepRunner(
            sim, bridge, px4, control_hz=control_config["frequency_hz"]
        )
        speed = control_config["simulation_speed"]
        runner.speed_cap = speed if speed > 0.0 else None

        runner.start()
        runner.wait_px4()
        try:
            param_file = RISEMAX_DIR / px4_config["param_file"]
            load_param_file(px4, str(param_file))
            px4.configure_offboard_no_rc()
            if not runner.wait_ekf_ready() or not runner.wait_heading():
                raise RuntimeError("PX4 estimator is not ready")

            ground_ned = np.asarray(runner.ground_ned, dtype=float)
            trajectory = make_trajectory(
                config,
                center_down=(
                    ground_ned[2] - trajectory_config["altitude_m"]
                ),
            )
            trajectory_origin = np.array(
                [ground_ned[0], ground_ned[1], 0.0]
            )

            start_position = (
                trajectory.get_desired_state(0.0)[0] + trajectory_origin
            )
            px4.goto_ned(*start_position, yaw_ned=yaw_ned)
            px4.emit_setpoint_now()
            if not runner.engage_offboard(arm=True):
                raise RuntimeError("PX4 refused OFFBOARD or arm")
            if not runner.fly_to_ned(
                *start_position,
                yaw_ned=yaw_ned,
                tol=0.30,
                settle_s=1.0,
                timeout_s=40.0,
                vel_tol=0.35,
            ):
                raise RuntimeError("PX4 could not reach the trajectory start")

            controller = AccelerationPid(
                kp=control_config["kp"],
                ki=control_config["ki"],
                kd=control_config["kd"],
                max_horizontal_accel=control_config[
                    "max_horizontal_accel_mps2"
                ],
                max_vertical_accel=control_config[
                    "max_vertical_accel_mps2"
                ],
            )
            dt = runner.control_dt
            start_time = runner.sim_time
            error_squared_sum = 0.0
            acceleration_squared_sum = 0.0
            steps = int(math.ceil(trajectory_config["duration_s"] / dt))

            print("[handover] position PID -> PX4 acceleration NED")
            for step in range(steps):
                position_ned = np.asarray(px4.state.pos_ned, dtype=float)
                velocity_ned = np.asarray(px4.state.vel_ned, dtype=float)
                elapsed = runner.sim_time - start_time
                desired_position, desired_velocity, _ = (
                    trajectory.get_desired_state(elapsed)
                )
                desired_position += trajectory_origin

                output = controller.update(
                    position_ned,
                    velocity_ned,
                    desired_position,
                    desired_velocity,
                    dt,
                )
                error_norm = np.linalg.norm(output.position_error_ned)
                acceleration_norm = np.linalg.norm(output.acceleration_ned)
                if error_norm > control_config["abort_position_error_m"]:
                    raise RuntimeError(
                        "tracking error exceeded "
                        f"{control_config['abort_position_error_m']} m"
                    )

                runner.step_with_acceleration_ned(
                    *output.acceleration_ned,
                    yaw_ned=yaw_ned,
                )
                error_squared_sum += error_norm**2
                acceleration_squared_sum += acceleration_norm**2

                if step % max(1, round(1.0 / dt)) == 0:
                    roll_pitch_deg = np.degrees(
                        px4.state.attitude_rpy_ned_frd[:2]
                    )
                    print(
                        f"[t={elapsed:5.1f}] error={error_norm:.2f} m "
                        f"target={np.round(desired_position, 2)} "
                        f"px4={np.round(position_ned, 2)} "
                        f"roll/pitch={np.round(roll_pitch_deg, 1)} deg "
                        f"accel={np.round(output.acceleration_ned, 2)}"
                    )

            print(
                f"[result] RMS error: "
                f"{math.sqrt(error_squared_sum / steps):.4f} m"
            )
            print(
                f"[result] RMS acceleration: "
                f"{math.sqrt(acceleration_squared_sum / steps):.4f} m/s^2"
            )
        finally:
            if px4.is_armed():
                ground_target = (
                    runner.ground_ned[0],
                    runner.ground_ned[1],
                    runner.ground_ned[2] - 0.08,
                )
                px4.goto_ned(*ground_target, yaw_ned=yaw_ned)
                runner.fly_to_ned(
                    *ground_target,
                    yaw_ned=yaw_ned,
                    tol=0.12,
                    settle_s=1.5,
                    timeout_s=45.0,
                    vel_tol=0.25,
                )
                runner.disarm()
            runner.close()
            sim.resume()

if __name__ == "__main__":
    run(load_config())
