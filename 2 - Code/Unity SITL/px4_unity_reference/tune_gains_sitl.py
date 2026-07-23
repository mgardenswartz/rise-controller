#!/usr/bin/env python3


import math
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import numpy as np
import optuna

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from GeometricController import (DroneParameters, GeometricControllerLee,
                                 LeeGainParameters)
from project_config import load_px4_tuning_config
from trajectories import Figure8Trajectory
from utils import sensors_to_state

from quadsim import QuadSim, UdpViz
from quadsim.px4 import Px4Link
from quadsim.px4_thrust import ThrustCalibration
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file


CONTROL_HZ = 50.0
DEFAULT_TRIAL_DURATION = 40.0  # two laps for screening; validate winners at 150 s

GRACE_S = 3.0
HARD_ERR_M = 10.0
SOFT_RMSE_M = 4.0
SOFT_WINDOW_S = 1.0
MIN_Z_M = 0.35


SAFETY_MIN_Z_M = 1.25
SAFETY_MAX_DESCENT_MPS = 3.0
SAFETY_MAX_SPEED_MPS = 8.0
SAFETY_MAX_TILT_DEG = 50.0
SAFETY_MAX_ERR_M = 3.0


Q_E = 2.0
FIXED_Q_YAW = 0.5
FIXED_R_OMEGA = 0.02
VELOCITY_Q_YAW = 1.0
VELOCITY_R_OMEGA = 0.01
R_DTHR = 0.003
W_FAIL  = 1000.0

# The original broad logarithmic space remains available for the fixed-heading
# baseline.  Tangent yaw exposed a stable basin around the original Sentinel
# gains, so the v2 velocity-yaw study samples that neighborhood linearly.
FIXED_BOUNDS = {
    "kx": (8.0, 30.0),
    "kv": (2.5, 10.0),
    "kR_rate": (2.0, 8.0),
}
VELOCITY_BOUNDS = {
    "kx": (14.0, 22.0),
    "kv": (3.5, 6.5),
    "kR_rate": (4.0, 7.5),
}
STABLE_SENTINEL_GAINS = {
    "kx": 17.98802006499531,
    "kv": 4.929956884645732,
    "kR_rate": 6.188531042643287,
}


class PlantRecoveryError(RuntimeError):
    """Infrastructure/reset failure: stop the study instead of blaming gains."""


def _wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class SitlPlant:
    sim: QuadSim
    runner: LockstepRunner
    px4: Px4Link
    cal: ThrustCalibration
    params: DroneParameters
    viz: UdpViz
    trial_duration: float
    fig8_heading: str
    needs_recovery: bool = False


def make_plant(args) -> SitlPlant:

    sim = QuadSim(host=args.quadsim_host, command_port=args.quadsim_port,
                  telemetry_port=args.quadsim_port + 1)
    sim.connect()
    if args.wind != "leave":
        print(f"[wind] {sim.set_wind(enabled=(args.wind == 'on'), wind_speed=args.wind_speed)}")
    params = DroneParameters(args.vehicle)
    cal = ThrustCalibration.load(args.calib)
    bridge = HilBridge(px4_host=args.px4_host, instance=args.instance,
                       px4_client=args.px4_client)
    px4 = Px4Link(stream_hz=0, instance=args.instance, thrust_cal=cal)
    runner = LockstepRunner(sim, bridge, px4, control_hz=CONTROL_HZ)
    if args.speed and args.speed > 0:
        runner.speed_cap = args.speed
        print(f"[speed] requested cap={args.speed:.1f}x; achieved FTRT is "
              "reported for every trial")
    runner.start()
    runner.wait_px4()
    if args.param_file:
        prefixes = (tuple(args.param_prefixes.split(","))
                    if args.param_prefixes else None)
        load_param_file(px4, args.param_file, prefixes)
    px4.configure_offboard_no_rc()
    if not runner.wait_ekf_ready():
        raise SystemExit("EKF never set home")
    if not runner.wait_heading():
        raise SystemExit("heading chain failed — do not tune")
    if px4.state.pos_enu is None:
        raise SystemExit("PX4 local position unavailable after EKF ready")
    px4.goto(*px4.state.pos_enu, 0.0)
    if not runner.engage_offboard(arm=True):
        raise SystemExit("offboard/arm refused — run quadsim-px4-lockstep --check")
    viz = UdpViz(addr=(args.viz_host, args.viz_port), enabled=args.viz,
                 source=f"px4-optuna-{args.instance}")
    if args.viz:
        print(f"[viz] streaming worker {args.instance} to "
              f"udp://{args.viz_host}:{args.viz_port}")
    return SitlPlant(sim=sim, runner=runner, px4=px4, cal=cal, params=params,
                     viz=viz, trial_duration=float(args.trial_duration),
                     fig8_heading=args.fig8_heading)


def reset_trial(plant: SitlPlant, start, heading, tol=0.35) -> bool:
    r, px4 = plant.runner, plant.px4
    pos = px4.state.pos_enu
    if (plant.needs_recovery or pos is None
            or not (px4.is_armed() and px4.in_offboard())):
        print("[reset] FAIL: plant/EKF requires a clean PX4 restart")
        return False

    # Normal trial completion: let MPC fly back without perturbing the EKF.
    if r.fly_to(float(start[0]), float(start[1]), float(start[2]),
                yaw=float(heading), tol=tol, settle_s=1.0,
                timeout_s=25.0, vel_tol=0.4):
        return True

    # A failed MPC recovery is infrastructure state, not evidence against the
    # next candidate. Stop instead of teleporting truth under a live EKF.
    print("[reset] FAIL: MPC fly-back failed; clean PX4 restart required")
    plant.needs_recovery = True
    return False


def _record_timing(trial, runner, sim_t0, wall_t0, reset_wall_s):
    sim_span = max(0.0, runner.sim_time - sim_t0)
    wall_span = max(1e-9, time.perf_counter() - wall_t0)
    ftrt = sim_span / wall_span
    trial.set_user_attr("reset_wall_s", float(reset_wall_s))
    trial.set_user_attr("flight_wall_s", float(wall_span))
    trial.set_user_attr("achieved_ftrt", float(ftrt))
    print(f"      timing: reset={reset_wall_s:.2f}s wall; "
          f"flight={sim_span:.1f}s sim/{wall_span:.2f}s wall = {ftrt:.2f}x")
    return sim_span, wall_span, ftrt


# ============================================================================
# Objective
# ============================================================================
def objective(trial, plant: SitlPlant):
    velocity_yaw = plant.fig8_heading == "velocity"
    bounds = VELOCITY_BOUNDS if velocity_yaw else FIXED_BOUNDS
    use_log_scale = not velocity_yaw
    kx = trial.suggest_float("kx", *bounds["kx"], log=use_log_scale)
    kv = trial.suggest_float("kv", *bounds["kv"], log=use_log_scale)
    kR_rate = trial.suggest_float(
        "kR_rate", *bounds["kR_rate"], log=use_log_scale
    )
    q_yaw = VELOCITY_Q_YAW if velocity_yaw else FIXED_Q_YAW
    r_omega = VELOCITY_R_OMEGA if velocity_yaw else FIXED_R_OMEGA

    trial.set_user_attr("objective_version", "velocity_yaw_v2" if velocity_yaw
                        else "fixed_v1")
    trial.set_user_attr("q_yaw", q_yaw)
    trial.set_user_attr("r_omega", r_omega)

    print(f"\n--- Trial {trial.number}: kx={kx:.2f}, kv={kv:.2f}, "
          f"kR_rate={kR_rate:.2f} ---")

    params = plant.params
    gains = LeeGainParameters(params.profile)
    gains.kx, gains.kv, gains.kR_rate = kx, kv, kR_rate
    controller = GeometricControllerLee(params, gains)

    r = plant.runner
    dt = r.control_dt
    t_max = plant.trial_duration
    fail_cost = W_FAIL * t_max ** 2
    controller.set_timestep(dt)

    traj_gen = Figure8Trajectory(heading_mode=plant.fig8_heading)
    trial.set_user_attr("fig8_heading", plant.fig8_heading)
    # One lap is enough for the static ghost; the 150 s objective repeats it.
    # Keep this packet comfortably below the UDP datagram size limit.
    plant.viz.path([
        traj_gen.sample(t)["pos"].flatten()
        for t in np.linspace(0.0, 20.0, 201)
    ])
    s0 = traj_gen.sample(0.0)
    reset_wall_t0 = time.perf_counter()
    if not reset_trial(plant, s0["pos"].flatten(), s0["heading"]):
        trial.set_user_attr("fail_reason", "plant_recovery_failed")
        raise PlantRecoveryError(
            "trial reset/recovery failed; aborting study so gains are not "
            "charged for broken plant state")
    reset_wall_s = time.perf_counter() - reset_wall_t0

    controller.reset()
    hover_thrust = params.mass * params.gravity

    cost_error = 0.0
    cost_heading = 0.0
    cost_rate = 0.0
    cost_thrust_slew = 0.0
    last_terms = (0.0, 0.0, 0.0, 0.0)
    thrust_prev = hover_thrust
    min_z = float("inf")
    max_err = 0.0
    err_sq_sum = 0.0
    yaw_err_sq_sum = 0.0
    max_yaw_err = 0.0
    scored_ticks = 0
    max_omega_cmd = 0.0
    min_thrust = float("inf")
    max_thrust = 0.0
    max_ticks = int(t_max / dt)
    err_sq_window = []
    soft_window_ticks = max(1, int(round(SOFT_WINDOW_S / dt)))
    last_cmd = (0.0, 0.0, 0.0, plant.cal.normalized(hover_thrust))
    sensors = r.sensors
    t0 = r.sim_time
    flight_wall_t0 = time.perf_counter()

    try:
        for tick in range(max_ticks):
            state = sensors_to_state(sensors, px4_state=plant.px4.state)
            if state is None:
                sensors = r.step_with_rate_thrust(*last_cmd)
                continue

            min_z = min(min_z, float(state['pos'][2, 0]))
            t = tick * dt
            trajectory = traj_gen.sample(t)

            result = controller.compute_rate_thrust_setpoint(state, trajectory)
            omega_cmd = np.asarray(result['omega_flu'], dtype=float)
            thrust = float(result['thrust'])

            e = (state['pos'] - trajectory['pos']).flatten()
            norm_e_sq = float(e @ e)
            err = math.sqrt(norm_e_sq)
            yaw = math.atan2(float(state['R'][1, 0]),
                             float(state['R'][0, 0]))
            yaw_target = float(trajectory['heading'])
            yaw_err = _wrap_pi(yaw - yaw_target)
            max_err = max(max_err, err)
            err_sq_sum += norm_e_sq
            yaw_err_sq_sum += yaw_err ** 2
            max_yaw_err = max(max_yaw_err, abs(yaw_err))
            scored_ticks += 1
            max_omega_cmd = max(max_omega_cmd, float(np.linalg.norm(omega_cmd)))
            min_thrust = min(min_thrust, thrust)
            max_thrust = max(max_thrust, thrust)

            terms = (t * Q_E * norm_e_sq,
                     q_yaw * yaw_err ** 2,
                     r_omega * float(omega_cmd @ omega_cmd),
                     R_DTHR * (thrust - thrust_prev) ** 2)
            cost_error += (dt / 2.0) * (terms[0] + last_terms[0])
            cost_heading += (dt / 2.0) * (terms[1] + last_terms[1])
            cost_rate += (dt / 2.0) * (terms[2] + last_terms[2])
            cost_thrust_slew += (dt / 2.0) * (terms[3] + last_terms[3])
            last_terms = terms
            thrust_prev = thrust

            # --- validity gates (identical structure to tune_gains.py) ---
            pos_f = state['pos'].flatten()
            vel_f = state['vel'].flatten()
            err_sq_window.append(norm_e_sq)
            if len(err_sq_window) > soft_window_ticks:
                err_sq_window.pop(0)
            rmse_window = math.sqrt(sum(err_sq_window) / len(err_sq_window))
            z_now = float(pos_f[2])
            speed_now = float(np.linalg.norm(vel_f))
            descent_mps = max(0.0, -float(vel_f[2]))
            tilt_deg = math.degrees(math.acos(float(np.clip(
                state['R'][2, 2], -1.0, 1.0))))

            # Optional headless visualization. UDP is fire-and-forget and
            # deliberately decimated so visualization cannot pace lockstep.
            if tick % 5 == 0:
                plant.viz.sample(
                    t, pos_f, target=trajectory['pos'].flatten(), err=err,
                    speed=speed_now, trial=trial.number,
                    yaw_deg=math.degrees(yaw),
                    yaw_target_deg=math.degrees(yaw_target),
                    yaw_error_deg=math.degrees(yaw_err))

            fail_reason = None
            if not (np.all(np.isfinite(pos_f)) and np.all(np.isfinite(vel_f))):
                fail_reason = "nonfinite_state"
            elif z_now < SAFETY_MIN_Z_M:
                fail_reason = f"safety_low_altitude z={z_now:.2f}m"
            elif descent_mps > SAFETY_MAX_DESCENT_MPS:
                fail_reason = f"safety_descent_rate vz={vel_f[2]:.2f}m/s"
            elif speed_now > SAFETY_MAX_SPEED_MPS:
                fail_reason = f"safety_speed speed={speed_now:.2f}m/s"
            elif tilt_deg > SAFETY_MAX_TILT_DEG:
                fail_reason = f"safety_tilt tilt={tilt_deg:.1f}deg"
            elif err > SAFETY_MAX_ERR_M:
                fail_reason = f"safety_tracking_radius err={err:.2f}m"
            elif t >= GRACE_S:
                if z_now < MIN_Z_M:
                    fail_reason = f"ground_contact z={z_now:.2f}m"
                elif err > HARD_ERR_M:
                    fail_reason = f"hard_tracking_radius err={err:.2f}m"
                elif (len(err_sq_window) >= soft_window_ticks
                      and rmse_window > SOFT_RMSE_M):
                    fail_reason = (f"sustained_tracking_rmse "
                                   f"rmse={rmse_window:.2f}m")

            if fail_reason is not None:
                ground_z = r.ground_enu[2]
                hard_contact = (not np.all(np.isfinite(pos_f))
                                or z_now <= ground_z + 0.35
                                or not (plant.px4.is_armed()
                                        and plant.px4.in_offboard()))
                if hard_contact:
                    plant.needs_recovery = True
                    print("   -> vehicle/estimator no longer recoverable; "
                          "clean PX4 restart required")
                else:
                    # Change SET_ATTITUDE_TARGET to a position setpoint NOW;
                    # PX4 MPC gets the vehicle before the next reset wait.
                    plant.px4.goto(0.0, 0.0, 3.0, 0.0)
                    plant.px4.emit_setpoint_now()
                    plant.needs_recovery = False
                    print("   -> safety takeover: handed control to PX4 MPC")
                cost = cost_error + cost_heading + cost_rate + cost_thrust_slew
                remaining = max(0.0, t_max - t)
                cost += W_FAIL * remaining ** 2
                cost += 100.0 * err ** 2
                cost += 50.0 * rmse_window ** 2
                trial.set_user_attr("fail_reason", fail_reason)
                trial.set_user_attr("fail_t", float(t))
                trial.set_user_attr("min_z", float(min_z))
                print(f"   -> [FAILED] {fail_reason} at t={t:.1f}s | "
                      f"Cost={cost:.0f}")
                _record_timing(trial, r, t0, flight_wall_t0, reset_wall_s)
                return cost

            last_cmd = (float(omega_cmd[0]), float(omega_cmd[1]),
                        float(omega_cmd[2]), plant.cal.normalized(thrust))
            sensors = r.step_with_rate_thrust(*last_cmd)

    except Exception as e:
        plant.needs_recovery = True
        print(f"   -> Simulation error: {e}")
        _record_timing(trial, r, t0, flight_wall_t0, reset_wall_s)
        return fail_cost

    cost = cost_error + cost_heading + cost_rate + cost_thrust_slew
    tracking_rms = math.sqrt(err_sq_sum / max(1, scored_ticks))
    yaw_rms = math.sqrt(yaw_err_sq_sum / max(1, scored_ticks))
    trial.set_user_attr("min_z", min_z)
    trial.set_user_attr("sim_t_span", r.sim_time - t0)
    trial.set_user_attr("tracking_rms", tracking_rms)
    trial.set_user_attr("max_err", max_err)
    trial.set_user_attr("cost_error", cost_error)
    trial.set_user_attr("cost_heading", cost_heading)
    trial.set_user_attr("cost_rate", cost_rate)
    trial.set_user_attr("cost_thrust_slew", cost_thrust_slew)
    trial.set_user_attr("max_omega_cmd", max_omega_cmd)
    trial.set_user_attr("min_thrust", min_thrust)
    trial.set_user_attr("max_thrust", max_thrust)
    trial.set_user_attr("yaw_rms_deg", math.degrees(yaw_rms))
    trial.set_user_attr("max_yaw_err_deg", math.degrees(max_yaw_err))
    print(f"   -> Trial {trial.number} survived. Cost={cost:.3f} "
          f"[error={cost_error:.3f}, heading={cost_heading:.3f}, "
          f"rate={cost_rate:.3f}, "
          f"dthrust={cost_thrust_slew:.3f}]")
    print(f"      RMS={tracking_rms:.3f}m max_err={max_err:.3f}m "
          f"yaw_RMS={math.degrees(yaw_rms):.2f}deg "
          f"yaw_max={math.degrees(max_yaw_err):.2f}deg "
          f"min_z={min_z:.2f}m max|omega_cmd|={max_omega_cmd:.2f}rad/s "
          f"thrust=[{min_thrust:.1f},{max_thrust:.1f}]N")
    _record_timing(trial, r, t0, flight_wall_t0, reset_wall_s)
    return cost


def main():
    config = load_px4_tuning_config()
    try:
        worker_index = int(os.environ.get("PX4_TUNING_WORKER_INDEX", "0"))
    except ValueError as error:
        raise SystemExit("PX4_TUNING_WORKER_INDEX must be an integer") from error
    args = config.worker(worker_index)
    print(f"[config] {config.source} (worker {worker_index})")

    if args.journal:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import (JournalFileBackend,
                                             JournalFileOpenLock)
        storage = JournalStorage(JournalFileBackend(
            args.journal, lock_obj=JournalFileOpenLock(args.journal)))
    else:
        storage = None

    study = optuna.create_study(
        study_name=args.study, storage=storage, direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed))

    if not args.no_seed_trial and len(study.trials) == 0:
        if args.fig8_heading == "velocity":
            seed = STABLE_SENTINEL_GAINS
            seed_name = "original stable Sentinel gains"
        else:
            current = LeeGainParameters(args.vehicle)
            seed = {
                "kx": current.kx,
                "kv": current.kv,
                "kR_rate": current.kR_rate,
            }
            seed_name = "current vehicle profile"
        study.enqueue_trial(seed, user_attrs={"seed_source": seed_name})
        print(f"  seeded trial 0 ({seed_name}): "
              f"kx={seed['kx']:.2f} kv={seed['kv']:.2f} "
              f"kR_rate={seed['kR_rate']:.2f}")

    plant = make_plant(args)
    try:
        study.optimize(lambda tr: objective(tr, plant),
                       n_trials=args.n_trials)
    except PlantRecoveryError as exc:
        print(f"\n[ABORT] {exc}")
    except KeyboardInterrupt:
        print("\n[Exit] interrupted — best so far below")
    finally:
        try:
            r = plant.runner
            if (not plant.needs_recovery and plant.px4.is_armed()
                    and plant.px4.in_offboard()):
                # Hand the final rate-controlled state back to MPC and land
                # against the EKF ground reference before disarming.
                r.fly_to(0.0, 0.0, 3.0, tol=0.6, settle_s=0.5,
                         timeout_s=30.0, vel_tol=0.5)
                gx, gy, gz = r.ground_enu
                r.fly_to(gx, gy, gz + 0.08, tol=0.12, settle_s=1.5,
                         timeout_s=30.0, vel_tol=0.25)
            elif plant.needs_recovery:
                print("[teardown] last trial failed; disarming without a "
                      "pointless MPC landing timeout")
            plant.runner.disarm()
        except Exception:
            pass
        plant.viz.close()
        plant.runner.close()
        plant.sim.disconnect()

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial {best.number}: cost={best.value:.0f}")
    for k, v in best.params.items():
        print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
