#!/usr/bin/env python3
"""Minimal Optuna + PX4 NED-acceleration lockstep skeleton.

Fill in only ``build_controller`` and ``run_trial``. This intentionally
contains no controller, trajectory, errors, thresholds, or objective policy.
"""

from pathlib import Path

import optuna

from quadsim import QuadSim
from quadsim.px4 import Px4Link
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file


# Handoff configuration -- change paths/ports for the recipient's machine.
ROOT = Path(__file__).resolve().parents[2]
QUADSIM_HOST = "localhost"
QUADSIM_PORT = 5555
PX4_HOST = "127.0.0.1"
PX4_INSTANCE = 0
PX4_CLIENT = False
PX4_PARAM_FILE = ROOT / "config/px4/quadsim_none_iris_mc.params"
PX4_PARAM_PREFIXES = ("MC_",)
CONTROL_HZ = 50.0
SPEED_CAP = 10.0                 # 0 = uncapped faster-than-realtime
TAKEOFF_ALTITUDE_M = 3.0
START_NORTH_M = 0.0
START_EAST_M = 0.0
START_YAW_NED_RAD = 0.0
N_TRIALS = 30
STUDY_NAME = "controller_accel_ned_sitl"
JOURNAL = ROOT / "optuna_journals/controller_accel_ned_sitl.log"


def build_controller(trial, dt):
    """TODO: suggest the recipient's gains and return their controller."""
    # Example shape:
    # gains = MyGains(k1=trial.suggest_float("k1", low, high))
    # return MyController(gains=gains, dt=dt)
    raise NotImplementedError


def run_trial(trial, controller, runner, px4):
    """TODO: recipient-owned control loop and Optuna objective.

    Native PX4 estimator state is available at::

        px4.state.pos_ned                 # north, east, down [m]
        px4.state.vel_ned                 # vN, vE, vD [m/s]
        px4.state.attitude_rpy_ned_frd    # roll, pitch, yaw [rad]
        px4.state.attitude_q_ned_frd      # Hamilton w, x, y, z
        px4.state.rates_frd               # body p, q, r [rad/s]

    Send acceleration and advance one synchronous control period with::

        runner.step_with_acceleration_ned(
            accel_north, accel_east, accel_down, yaw_ned)

    Define references, duration, errors, thresholds, pruning, and the returned
    scalar objective here.
    """
    raise NotImplementedError


def make_plant():
    sim = QuadSim(host=QUADSIM_HOST, command_port=QUADSIM_PORT,
                  telemetry_port=QUADSIM_PORT + 1)
    sim.connect()
    bridge = HilBridge(px4_host=PX4_HOST, instance=PX4_INSTANCE,
                       px4_client=PX4_CLIENT)
    px4 = Px4Link(stream_hz=0, instance=PX4_INSTANCE)
    runner = LockstepRunner(sim, bridge, px4, control_hz=CONTROL_HZ)
    if SPEED_CAP > 0.0:
        runner.speed_cap = SPEED_CAP

    runner.start()
    runner.wait_px4()
    load_param_file(px4, str(PX4_PARAM_FILE), PX4_PARAM_PREFIXES)
    px4.configure_offboard_no_rc()
    if not runner.wait_ekf_ready() or not runner.wait_heading():
        raise RuntimeError("PX4 EKF/heading did not become ready")
    return sim, runner, px4


def takeoff_and_reset(runner, px4, start_ned):
    """PX4 position-control takeoff/fly-to-start before each trial."""
    px4.goto_ned(*start_ned, yaw_ned=START_YAW_NED_RAD)
    px4.emit_setpoint_now()
    if not (px4.in_offboard() and px4.is_armed()):
        if not runner.engage_offboard(arm=True):
            raise RuntimeError("PX4 refused OFFBOARD/arm")
    if not runner.fly_to_ned(
            *start_ned, yaw_ned=START_YAW_NED_RAD, tol=0.35,
            settle_s=1.0, timeout_s=30.0, vel_tol=0.4):
        raise RuntimeError("PX4 could not reach the trial start")


def recover_land_disarm(runner, px4, start_ned):
    """PX4 takeover, recovery, landing, and disarm after each trial."""
    if not (px4.in_offboard() and px4.is_armed()):
        raise RuntimeError("vehicle left armed OFFBOARD; restart the plant")

    if not runner.fly_to_ned(
            *start_ned, yaw_ned=START_YAW_NED_RAD, tol=0.6,
            settle_s=0.5, timeout_s=30.0, vel_tol=0.5):
        raise RuntimeError("PX4 recovery failed")

    ground = list(runner.ground_ned)
    ground[2] -= 0.08
    if not runner.fly_to_ned(
            *ground, yaw_ned=START_YAW_NED_RAD, tol=0.12,
            settle_s=1.5, timeout_s=30.0, vel_tol=0.25):
        raise RuntimeError("PX4 landing failed")
    if not runner.disarm():
        raise RuntimeError("PX4 disarm failed")


def objective(trial, runner, px4, start_ned):
    controller = build_controller(trial, runner.control_dt)
    takeoff_and_reset(runner, px4, start_ned)
    try:
        controller.reset()
        return float(run_trial(trial, controller, runner, px4))
    finally:
        recover_land_disarm(runner, px4, start_ned)


def storage():
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    path = str(JOURNAL)
    return JournalStorage(JournalFileBackend(
        path, lock_obj=JournalFileOpenLock(path)))


def main():
    sim, runner, px4 = make_plant()
    ground = runner.ground_ned
    start_ned = (
        ground[0] + START_NORTH_M,
        ground[1] + START_EAST_M,
        ground[2] - TAKEOFF_ALTITUDE_M,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage(), direction="minimize",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42))
    try:
        study.optimize(lambda t: objective(t, runner, px4, start_ned),
                       n_trials=N_TRIALS)
    finally:
        if px4.is_armed():
            runner.disarm()
        runner.close()
        sim.disconnect()


if __name__ == "__main__":
    main()
