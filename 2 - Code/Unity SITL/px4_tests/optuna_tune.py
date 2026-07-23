#!/usr/bin/env python3
"""Optuna gain tuning with one persistent QuadSim/PX4 HIL plant."""

import math

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from quadsim import QuadSim
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file

from pid_controller import AccelerationPid
from px4_pid_node import RISEMAX_DIR, load_config, make_trajectory


class PlantRecoveryError(RuntimeError):
    pass


def suggest_controller(trial, config):
    """Controller hook: replace this function for the coworker's RISE node."""
    control = config["control"]
    bounds = config["optuna"]["gain_bounds"]
    kp = trial.suggest_float("kp", **bounds["kp"])
    ki = trial.suggest_float("ki", **bounds["ki"])
    kd = trial.suggest_float("kd", **bounds["kd"])
    return AccelerationPid(
        kp=kp,
        ki=ki,
        kd=kd,
        max_horizontal_accel=control["max_horizontal_accel_mps2"],
        max_vertical_accel=control["max_vertical_accel_mps2"],
    )


def controller_step(
    controller,
    position_ned,
    velocity_ned,
    desired_position_ned,
    desired_velocity_ned,
    desired_acceleration_ned,
    dt,
):
    """Controller hook returning ``acceleration_ned`` and position error.

    The PID intentionally ignores desired acceleration, matching the original
    Aviary RISE PID branch. A replacement RISE controller can use it here.
    """
    return controller.update(
        position_ned,
        velocity_ned,
        desired_position_ned,
        desired_velocity_ned,
        dt,
    )


def make_storage(config):
    journal_path = RISEMAX_DIR / config["optuna"]["journal_file"]
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    path = str(journal_path)
    backend = JournalFileBackend(
        path,
        lock_obj=JournalFileOpenLock(path),
    )
    return JournalStorage(backend)


def make_plant(config):
    quadsim_config = config["quadsim"]
    px4_config = config["px4"]
    control_config = config["control"]

    sim = QuadSim(
        host=quadsim_config["host"],
        command_port=quadsim_config["port"],
        telemetry_port=quadsim_config["port"] + 1,
    )
    sim.connect()
    bridge = HilBridge(
        px4_host=px4_config["host"],
        instance=px4_config["instance"],
        px4_client=px4_config["client"],
    )
    px4 = Px4Link(stream_hz=0, instance=px4_config["instance"])
    runner = LockstepRunner(
        sim,
        bridge,
        px4,
        control_hz=control_config["frequency_hz"],
    )
    speed = config["optuna"]["simulation_speed"]
    runner.speed_cap = speed if speed > 0.0 else None

    runner.start()
    runner.wait_px4()
    param_file = RISEMAX_DIR / px4_config["param_file"]
    load_param_file(px4, str(param_file))
    px4.configure_offboard_no_rc()
    if not runner.wait_ekf_ready() or not runner.wait_heading():
        raise RuntimeError("PX4 estimator is not ready")
    return sim, runner, px4


def hover_origin(config, runner):
    reset = config["optuna"]["reset"]
    ground = runner.ground_ned
    return np.array(
        [
            ground[0] + reset["north_m"],
            ground[1] + reset["east_m"],
            ground[2] - config["trajectory"]["altitude_m"],
        ],
        dtype=float,
    )


def return_to_hover(config, runner, px4, origin):
    reset = config["optuna"]["reset"]
    yaw_ned = config["control"]["yaw_ned_rad"]
    print("[reset] PX4 position control -> hover origin")
    if not (px4.is_armed() and px4.in_offboard()):
        raise PlantRecoveryError("vehicle left armed OFFBOARD during a trial")
    if not runner.fly_to_ned(
        *origin,
        yaw_ned=yaw_ned,
        tol=reset["position_tolerance_m"],
        vel_tol=reset["velocity_tolerance_mps"],
        settle_s=reset["settle_s"],
        timeout_s=reset["timeout_s"],
    ):
        raise PlantRecoveryError("could not recover and settle at hover origin")
    print("[reset] settled; next trial may begin")


def run_trial(trial, controller, config, runner, px4, trajectory, origin):
    control = config["control"]
    trajectory_config = config["trajectory"]
    objective_config = config["optuna"]["objective"]
    yaw_ned = control["yaw_ned_rad"]
    dt = runner.control_dt
    steps = int(math.ceil(trajectory_config["duration_s"] / dt))
    trajectory_origin = np.array([origin[0], origin[1], 0.0])

    cost = 0.0
    last_integrand = None
    error_squared_sum = 0.0
    max_error = 0.0

    for step in range(steps):
        elapsed = step * dt
        position_ned = np.asarray(px4.state.pos_ned, dtype=float)
        velocity_ned = np.asarray(px4.state.vel_ned, dtype=float)
        desired_position, desired_velocity, desired_acceleration = (
            trajectory.get_desired_state(elapsed)
        )
        desired_position += trajectory_origin

        output = controller_step(
            controller,
            position_ned,
            velocity_ned,
            desired_position,
            desired_velocity,
            desired_acceleration,
            dt,
        )
        error = float(np.linalg.norm(output.position_error_ned))
        acceleration_squared = float(
            output.acceleration_ned @ output.acceleration_ned
        )
        error_squared = error**2
        max_error = max(max_error, error)
        error_squared_sum += error_squared

        integrand = (
            elapsed * objective_config["error_weight"] * error_squared
            + objective_config["acceleration_weight"] * acceleration_squared
        )
        if last_integrand is None:
            last_integrand = integrand
        cost += 0.5 * dt * (integrand + last_integrand)
        last_integrand = integrand

        if not np.all(np.isfinite(output.acceleration_ned)):
            fail_reason = "nonfinite_acceleration"
        elif error > control["abort_position_error_m"]:
            fail_reason = f"tracking_error_{error:.2f}m"
        else:
            fail_reason = None

        if fail_reason is not None:
            remaining = max(0.0, trajectory_config["duration_s"] - elapsed)
            cost += objective_config["failure_weight"] * remaining**2
            trial.set_user_attr("fail_reason", fail_reason)
            trial.set_user_attr("fail_time_s", elapsed)
            print(f"[trial {trial.number}] failed: {fail_reason}")
            return cost

        runner.step_with_acceleration_ned(
            *output.acceleration_ned,
            yaw_ned=yaw_ned,
        )

        if step % max(1, round(1.0 / dt)) == 0:
            trial.report(cost, step)
            print(
                f"[trial {trial.number} t={elapsed:5.1f}] "
                f"error={error:.2f}m cost={cost:.2f}"
            )

    rms_error = math.sqrt(error_squared_sum / steps)
    trial.set_user_attr("rms_error_m", rms_error)
    trial.set_user_attr("max_error_m", max_error)
    print(
        f"[trial {trial.number}] complete: cost={cost:.2f} "
        f"RMS={rms_error:.3f}m max={max_error:.3f}m"
    )
    return cost


def objective(trial, config, runner, px4, trajectory, origin):
    controller = suggest_controller(trial, config)
    controller.reset()
    parameters = " ".join(
        f"{name}={value:.4g}" for name, value in trial.params.items()
    )
    print(f"\n--- Trial {trial.number}: {parameters} ---")
    try:
        return float(
            run_trial(
                trial,
                controller,
                config,
                runner,
                px4,
                trajectory,
                origin,
            )
        )
    finally:
        return_to_hover(config, runner, px4, origin)


def land_and_close(config, sim, runner, px4):
    yaw_ned = config["control"]["yaw_ned_rad"]
    if px4.is_armed():
        ground = runner.ground_ned
        runner.fly_to_ned(
            ground[0],
            ground[1],
            ground[2] - 0.08,
            yaw_ned=yaw_ned,
            tol=0.12,
            vel_tol=0.25,
            settle_s=1.5,
            timeout_s=45.0,
        )
        runner.disarm()
    runner.close()
    sim.resume()
    sim.disconnect()


def main():
    config = load_config()
    optuna_config = config["optuna"]
    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        storage=make_storage(config),
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=optuna_config["sampler_seed"]
        ),
    )
    if (
        optuna_config["enqueue_current_gains"]
        and len(study.trials) == 0
    ):
        control = config["control"]
        study.enqueue_trial(
            {"kp": control["kp"], "ki": control["ki"], "kd": control["kd"]}
        )

    sim, runner, px4 = make_plant(config)
    origin = hover_origin(config, runner)
    trajectory = make_trajectory(config, center_down=origin[2])
    yaw_ned = config["control"]["yaw_ned_rad"]

    try:
        px4.goto_ned(*origin, yaw_ned=yaw_ned)
        px4.emit_setpoint_now()
        if not runner.engage_offboard(arm=True):
            raise RuntimeError("PX4 refused OFFBOARD or arm")
        return_to_hover(config, runner, px4, origin)
        study.optimize(
            lambda trial: objective(
                trial,
                config,
                runner,
                px4,
                trajectory,
                origin,
            ),
            n_trials=optuna_config["n_trials"],
        )
    except KeyboardInterrupt:
        print("\n[study] interrupted")
    except PlantRecoveryError as error:
        print(f"\n[study] stopped: {error}")
    finally:
        land_and_close(config, sim, runner, px4)

    best = study.best_trial
    print(f"\nBest trial {best.number}: cost={best.value:.3f}")
    for name, value in best.params.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
