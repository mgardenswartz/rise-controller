#!/usr/bin/env python3


import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from numpy.linalg import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from GeometricController import (DroneParameters, GeometricControllerLee,
                                 LeeGainParameters)
from project_config import load_px4_geometric_config
from resnet_compensator import AdaptiveConfig, ResNetCompensator
from trajectories import Figure8Trajectory
from utils import PerformanceMonitor, sensors_to_state

from quadsim import QuadSim, UdpViz
from quadsim.px4 import Px4Link
from quadsim.px4_thrust import ThrustCalibration
from quadsim.tools.px4_bridge import HilBridge
from quadsim.tools.px4_lockstep_runner import LockstepRunner, load_param_file

CONTROL_HZ = 50.0
VIZ_HZ = 10.0


def _wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _stream_viz(viz, t, state, trajectory, phi=None):
    """Publish one ENU/FLU sample for the headless live viewer."""
    pos = state["pos"].flatten()
    target = trajectory["pos"].flatten()
    vel = state["vel"].flatten()
    yaw = math.atan2(float(state["R"][1, 0]), float(state["R"][0, 0]))
    yaw_target = float(trajectory["heading"])
    extra = {
        "yaw_deg": math.degrees(yaw),
        "yaw_target_deg": math.degrees(yaw_target),
        "yaw_error_deg": math.degrees(_wrap_pi(yaw - yaw_target)),
    }
    if phi is not None:
        extra["phi_norm"] = float(norm(phi))
    viz.sample(t, pos, target=target,
               err=float(norm(pos - target)),
               speed=float(norm(vel)), trial=0, **extra)


def _stream_viz_path(viz, trajectory, duration):
    """Send a compact planned-path ghost that stays below one UDP datagram."""
    preview_s = min(float(duration), 40.0)
    viz.path([
        trajectory.sample(t)["pos"].flatten()
        for t in np.linspace(0.0, preview_s, 201)
    ])


# ============================================================================
# Stage trajectories (FLU/ENU, same dict interface as trajectories.py)
# ============================================================================
class HoverTrajectory:
    """Stage 3: hold one point. RMS here is the noise floor of the whole
    stack (EKF + rate loop + our attitude law)."""

    def __init__(self, pos=(0.0, 0.0, 3.0), heading=0.0):
        self.p = np.asarray(pos, dtype=float).reshape(3, 1)
        self.h = float(heading)

    def sample(self, t):
        return {"pos": self.p.copy(), "vel": np.zeros((3, 1)),
                "accel": np.zeros((3, 1)), "heading": self.h}


class YawSweepTrajectory:
    """
    Stage 4: the frame-error detector. Per yaw in (0, 90, 180) deg:
    hover -> +East leg -> back, all with smoothstep position profiles.
    The TRANSLATION is identical in the world frame for every yaw; only the
    nose direction changes. A sign/axis error at the FLU->FRD or ENU->NED
    boundary makes the world-frame response depend on yaw — compare per-leg
    RMS (the CSV carries a `leg` marker via heading).
    """
    YAWS = (0.0, math.pi / 2.0, math.pi)

    def __init__(self, base=(0.0, 0.0, 3.0), leg_east=3.0,
                 hover_s=3.0, leg_s=4.0):
        self.base = np.asarray(base, dtype=float).reshape(3, 1)
        self.leg = float(leg_east)
        self.hover_s = float(hover_s)
        self.leg_s = float(leg_s)
        self.seg_s = hover_s + 2.0 * leg_s          # hover, out, back
        self.total_s = len(self.YAWS) * self.seg_s

    @staticmethod
    def _smooth(s):
        s = max(0.0, min(1.0, s))
        S = 6 * s**5 - 15 * s**4 + 10 * s**3
        Sd = 30 * s**4 - 60 * s**3 + 30 * s**2
        Sdd = 120 * s**3 - 180 * s**2 + 60 * s
        return S, Sd, Sdd

    def sample(self, t):
        if t >= self.total_s:
            # ``make_trajectory`` intentionally adds a short final hover to
            # the scored sweep. Clamp to the completed base pose instead of
            # allowing tau to create a nonexistent third translation leg.
            return {"pos": self.base.copy(),
                    "vel": np.zeros((3, 1)),
                    "accel": np.zeros((3, 1)),
                    "heading": self.YAWS[-1]}
        i = min(len(self.YAWS) - 1, int(t // self.seg_s))
        yaw = self.YAWS[i]
        tau = t - i * self.seg_s
        pos = self.base.copy()
        vel = np.zeros((3, 1))
        acc = np.zeros((3, 1))
        if tau >= self.hover_s:                      # out or back leg
            tau -= self.hover_s
            leg_i = int(tau // self.leg_s)           # 0 = out, 1 = back
            s = (tau - leg_i * self.leg_s) / self.leg_s
            S, Sd, Sdd = self._smooth(s)
            if leg_i == 0:
                x, sgn = S, 1.0
            else:
                x, sgn = 1.0 - S, -1.0
            pos[0, 0] += self.leg * x
            vel[0, 0] = sgn * self.leg * Sd / self.leg_s
            acc[0, 0] = sgn * self.leg * Sdd / self.leg_s**2
        return {"pos": pos, "vel": vel, "accel": acc, "heading": yaw}


def make_trajectory(name, fig8_heading="fixed"):
    if name == "hover":
        return HoverTrajectory(), 30.0
    if name == "yawsweep":
        tr = YawSweepTrajectory()
        return tr, tr.total_s + 2.0
    if name == "fig8":
        # Fixed heading preserves the gain-tuning baseline. Velocity heading
        # points the nose along the figure-eight tangent for a combined
        # translation + yaw demonstration.
        return Figure8Trajectory(heading_mode=fig8_heading), 140.0
    raise SystemExit(f"unknown --traj {name}")


def _port_is_open(host, port, timeout_s=0.25):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _launch_owned_sim(args):
    """Launch one Unity build that this process owns and may terminate."""
    if not args.launch_sim:
        return None

    build = Path(args.sim_build).expanduser().resolve()
    if not build.exists():
        raise SystemExit(f"QuadSim build not found: {build}")
    if _port_is_open(args.quadsim_host, args.quadsim_port):
        raise SystemExit(
            f"QuadSim RPC port {args.quadsim_port} is already in use. "
            "Refusing to launch/kill an unowned build. Close it once, then "
            "use --launch-sim so future cleanup is automatic."
        )

    log_path = Path(args.sim_log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(build)]
    if not args.sim_graphics:
        cmd += ["-batchmode", "-nographics"]
    cmd += [
        "-rpcPort", str(args.quadsim_port),
        "-telemetryPort", str(args.quadsim_port + 1),
        "-logFile", str(log_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    print(f"[sim] launched owned QuadSim pid={proc.pid} "
          f"rpc={args.quadsim_port} -> {log_path}")

    try:
        deadline = time.monotonic() + args.sim_start_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise SystemExit(
                    f"QuadSim exited during startup (rc={proc.returncode}); "
                    f"check {log_path}"
                )
            if _port_is_open(args.quadsim_host, args.quadsim_port):
                print("[sim] QuadSim RPC ready")
                return proc
            time.sleep(0.25)
        raise SystemExit(
            f"QuadSim did not bind RPC port {args.quadsim_port}; "
            f"check {log_path}")
    except BaseException:
        _stop_owned_sim(proc)
        raise


def _stop_owned_sim(proc):
    if proc is None or proc.poll() is not None:
        return
    print(f"[sim] stopping owned QuadSim pid={proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"[sim] QuadSim pid={proc.pid} did not exit; killing it")
        proc.kill()
        proc.wait(timeout=5)


# ============================================================================
# Shared controller tick (both modes call exactly this)
# ============================================================================
def controller_tick(t, sensors, px4_state, controller, traj_gen, compensator,
                    mass, dt):
    """PX4 estimate + sensors -> (omega_flu, thrust_N) + logging bundle.
    Returns None on degenerate sensors (caller holds last command)."""
    state = sensors_to_state(sensors, px4_state=px4_state)
    if state is None:
        return None
    trajectory = traj_gen.sample(t)
    phi = None
    traj_cmd = trajectory
    if compensator is not None:
        phi = compensator.step(state, trajectory, dt)
        traj_cmd = dict(trajectory)
        # Command-level (hardware-deployable) injection: the estimated
        # disturbance force folds into a_des BEFORE R_des is built, so both
        # thrust magnitude and tilt compensate. Same convention as the
        # wrench path: Phi estimates +d, injected as a_des - phi/m.
        traj_cmd["accel"] = trajectory["accel"] - phi / mass
    result = controller.compute_rate_thrust_setpoint(state, traj_cmd)
    return state, trajectory, result, phi


# ============================================================================
# Lockstep entry point
# ============================================================================
def run_lockstep(args, controller, compensator, params, cal, monitor, viz):
    dt = 1.0 / CONTROL_HZ
    with QuadSim(host=args.quadsim_host, command_port=args.quadsim_port,
                 telemetry_port=args.quadsim_port + 1) as sim:
        if args.wind != "leave":
            print(f"[wind] {sim.set_wind(enabled=(args.wind == 'on'), wind_speed=args.wind_speed)}")
        bridge = HilBridge(px4_host=args.px4_host, instance=args.instance,
                           px4_client=args.px4_client)
        px4 = Px4Link(stream_hz=0, instance=args.instance, thrust_cal=cal)
        runner = LockstepRunner(sim, bridge, px4, control_hz=CONTROL_HZ)
        if args.speed and args.speed > 0:
            runner.speed_cap = args.speed
        controller.set_timestep(runner.control_dt)
        dt = runner.control_dt

        runner.start()
        runner.wait_px4()
        try:
            if args.param_file:
                prefixes = (tuple(args.param_prefixes.split(","))
                            if args.param_prefixes else None)
                load_param_file(px4, args.param_file, prefixes)
            px4.configure_offboard_no_rc()
            if not runner.wait_ekf_ready():
                raise SystemExit("EKF never set home — check bridge/GPS")
            if not runner.wait_heading():
                raise SystemExit("heading chain failed — do not arm")

            # Approach the trajectory start with PX4's own MPC (position
            # offboard), then hand over to our rate law. This mirrors the
            # hardware flow and is why teleport resets are never needed here.
            traj_gen, duration = make_trajectory(
                args.traj, fig8_heading=args.fig8_heading)
            if args.duration:
                duration = args.duration
            _stream_viz_path(viz, traj_gen, duration)
            s0 = traj_gen.sample(0.0)
            start = s0["pos"].flatten()
            s = runner.sensors
            px4.goto(s.gps_position[0], s.gps_position[1], s.gps_position[2], 0.0)
            if not runner.engage_offboard(arm=True):
                raise SystemExit("offboard/arm refused — run "
                                 "quadsim-px4-lockstep --check first")
            if not runner.fly_to(start[0], start[1], start[2],
                                 yaw=float(s0["heading"])):
                raise SystemExit("could not reach trajectory start")
            handover_state = sensors_to_state(runner.sensors,
                                              px4_state=px4.state)
            if handover_state is None:
                raise SystemExit("PX4 estimator state incomplete at handover")
            handover_error = norm(handover_state["pos"] - s0["pos"])
            handover_speed = norm(handover_state["vel"])
            print(f"[handover] PX4 EKF error={handover_error:.3f} m, "
                  f"speed={handover_speed:.3f} m/s")
            print(f"[handover] rate+thrust control at sim t={runner.sim_time:.1f}s")

            t0 = runner.sim_time
            sensors = runner.sensors
            last = (0.0, 0.0, 0.0, cal.normalized(params.mass * params.gravity))
            n_ticks = int(duration / dt)
            log_every = int(CONTROL_HZ)
            abort_reason = None
            for k in range(n_ticks):
                t = runner.sim_time - t0
                out = controller_tick(t, sensors, px4.state, controller, traj_gen,
                                      compensator, params.mass, dt)
                if out is None:
                    sensors = runner.step_with_rate_thrust(*last)
                    continue
                state, trajectory, result, phi = out
                position_error = norm(state["pos"] - trajectory["pos"])
                tilt_deg = math.degrees(math.acos(
                    float(np.clip(state["R"][2, 2], -1.0, 1.0))))
                if position_error > args.abort_position_error:
                    abort_reason = (f"tracking error {position_error:.2f} m "
                                    f"> {args.abort_position_error:.2f} m")
                elif tilt_deg > args.abort_tilt:
                    abort_reason = (f"tilt {tilt_deg:.1f} deg > "
                                    f"{args.abort_tilt:.1f} deg")
                if abort_reason:
                    print(f"[safety] ABORT: {abort_reason}; handing control "
                          "back to PX4 MPC")
                    break
                w = result["omega_flu"]
                u = cal.normalized(result["thrust"])
                last = (w[0], w[1], w[2], u)
                sensors = runner.step_with_rate_thrust(*last)
                monitor.update(state=state, trajectory=trajectory,
                               result=result, phi=phi,
                               theta_norm=(compensator.theta_norm
                                           if compensator else None),
                               theta_vector=None, dt=dt)
                if k % max(1, int(round(CONTROL_HZ / VIZ_HZ))) == 0:
                    _stream_viz(viz, t, state, trajectory, phi)
                if k % log_every == 0:
                    e = norm(state["pos"] - trajectory["pos"])
                    yaw_actual_deg = math.degrees(math.atan2(
                        float(state["R"][1, 0]), float(state["R"][0, 0])))
                    yaw_target_deg = math.degrees(float(trajectory["heading"]))
                    print(f"[t={t:6.1f}s] err={e:5.2f} m  thrust={result['thrust']:5.1f} N"
                          f"  u={u:.2f}  yaw={yaw_actual_deg:6.1f}/"
                          f"{yaw_target_deg:6.1f}°"
                          f"  |eR|={norm(result['e_R']):.3f}"
                          + (f"  |phi|={norm(phi):.2f} N" if phi is not None else ""))

            # Recover with MPC, then settle at the measured ground height.
            # A fixed 0.6 m target could disarm in the air when the scene's
            # spawn/ground Z differs and could manufacture estimator warnings.
            px4.goto(start[0], start[1], start[2], 0.0)
            if abort_reason:
                runner.fly_to(start[0], start[1], start[2], tol=0.75,
                              timeout_s=60.0, vel_tol=0.5)
            gx, gy, gz = runner.ground_enu
            if not runner.fly_to(gx, gy, gz + 0.08, tol=0.12,
                                 settle_s=1.5, timeout_s=60.0,
                                 vel_tol=0.25):
                print("[landing] WARN: controlled landing did not settle "
                      "before disarm")
            runner.disarm()
            if abort_reason:
                raise SystemExit(f"geometric flight aborted: {abort_reason}")
        finally:
            if px4.is_armed():
                try:
                    runner.disarm()
                except Exception:
                    pass
            runner.close()
            try:
                sim.resume()   # leave the editor live, not frozen-paused
            except Exception:
                pass


# ============================================================================
# Realtime entry point (bridge runs separately at --speed 1.0)
# ============================================================================
def run_realtime(args, controller, compensator, params, cal, monitor, viz):
    dt = 1.0 / CONTROL_HZ
    controller.set_timestep(dt)
    # 100 Hz wall stream: the rx thread re-emits the latest setpoint between
    # our 50 Hz updates — margin for PX4's offboard-loss timer without
    # changing the control rate.
    px4 = Px4Link(stream_hz=100.0, instance=args.instance, thrust_cal=cal)
    with QuadSim(host=args.quadsim_host, command_port=args.quadsim_port,
                 telemetry_port=args.quadsim_port + 1) as sim:
        sim.set_frame("flu")
        if args.wind != "leave":
            print(f"[wind] {sim.set_wind(enabled=(args.wind == 'on'), wind_speed=args.wind_speed)}")
        drone = sim.drone()
        px4.connect()
        try:
            if args.param_file:
                prefixes = (tuple(args.param_prefixes.split(","))
                            if args.param_prefixes else None)
                load_param_file(px4, args.param_file, prefixes)
            px4.configure_offboard_no_rc()
            if not px4.wait_home(timeout=60):
                raise SystemExit("EKF never set home — is the bridge running?")

            traj_gen, duration = make_trajectory(
                args.traj, fig8_heading=args.fig8_heading)
            if args.duration:
                duration = args.duration
            _stream_viz_path(viz, traj_gen, duration)
            s0 = traj_gen.sample(0.0)
            start = s0["pos"].flatten()
            if not px4.set_offboard(hold=True):
                raise SystemExit("OFFBOARD refused")
            if not px4.arm(True):
                raise SystemExit("arming refused")
            px4.goto(start[0], start[1], start[2], float(s0["heading"]))
            deadline = time.time() + 40.0
            while time.time() < deadline:
                p = px4.state.pos_enu
                if p is not None and norm(np.array(p) - start) < 0.35:
                    break
                time.sleep(0.1)
            else:
                raise SystemExit("could not reach trajectory start")
            print("[handover] rate+thrust control (realtime)")

            t_start = time.monotonic()
            next_tick = t_start
            last_u = cal.normalized(params.mass * params.gravity)
            k = 0
            while True:
                t = time.monotonic() - t_start
                if t >= duration:
                    break
                sensors = drone.get_sensors()
                out = controller_tick(t, sensors, px4.state, controller, traj_gen,
                                      compensator, params.mass, dt)
                if out is not None:
                    state, trajectory, result, phi = out
                    w = result["omega_flu"]
                    last_u = cal.normalized(result["thrust"])
                    px4.send_rate_thrust(w[0], w[1], w[2], last_u, emit=True)
                    monitor.update(state=state, trajectory=trajectory,
                                   result=result, phi=phi,
                                   theta_norm=(compensator.theta_norm
                                               if compensator else None),
                                   theta_vector=None, dt=dt)
                    if k % max(1, int(round(CONTROL_HZ / VIZ_HZ))) == 0:
                        _stream_viz(viz, t, state, trajectory, phi)
                    if k % int(CONTROL_HZ) == 0:
                        e = norm(state["pos"] - trajectory["pos"])
                        print(f"[t={t:6.1f}s] err={e:5.2f} m  u={last_u:.2f}")
                k += 1
                next_tick += dt
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
            px4.land()
        finally:
            px4.close()


# ============================================================================
def report(monitor, traj_name=None, grace_s=3.0):
    """Position RMS after the capture transient + eR summary — the numbers
    stages 3-6 are compared on."""
    t = np.array(monitor.t)
    if len(t) == 0:
        return
    pos = np.array(monitor.pos)
    tgt = np.array(monitor.tgt)
    eR = np.array(monitor.eR)
    m = t >= grace_s
    if not np.any(m):
        # Early safety aborts can occur before the normal grace window. Still
        # emit a useful partial report instead of reducing empty arrays.
        m = np.ones_like(t, dtype=bool)
    err = np.linalg.norm(pos[m] - tgt[m], axis=1)
    print(f"\n[report] ticks={len(t)}  (grace {grace_s:.0f}s excluded)")
    print(f"[report] position RMS = {np.sqrt(np.mean(err**2)):.3f} m   "
          f"max = {err.max():.3f} m")
    print(f"[report] |e_R| mean = {np.linalg.norm(eR[m], axis=1).mean():.4f}   "
          f"max = {np.linalg.norm(eR[m], axis=1).max():.4f}")
    if traj_name == "yawsweep":
        tr = YawSweepTrajectory()
        yaw_rms = []
        for i, yaw in enumerate(tr.YAWS):
            # Exclude one second after each commanded yaw change, then score
            # the identical hover/out/back translation segment.
            seg = ((t >= i * tr.seg_s + 1.0)
                   & (t < (i + 1) * tr.seg_s - 0.25))
            if not np.any(seg):
                continue
            seg_err = np.linalg.norm(pos[seg] - tgt[seg], axis=1)
            rms = float(np.sqrt(np.mean(seg_err ** 2)))
            yaw_rms.append(rms)
            print(f"[report] yaw {math.degrees(yaw):5.0f}°: "
                  f"position RMS={rms:.3f} m, max={seg_err.max():.3f} m")
        if yaw_rms:
            print(f"[report] yaw-dependent RMS spread="
                  f"{max(yaw_rms) - min(yaw_rms):.3f} m")


def main():
    args = load_px4_geometric_config()
    print(f"[config] {args.source}")
    params = DroneParameters(args.vehicle)
    controller = GeometricControllerLee(
        params, LeeGainParameters(args.vehicle)
    )
    print(f"[setup] Sentinel translation gains: kx={controller.kx:.6f}, "
          f"kv={controller.kv:.6f}; PX4 rate head: "
          f"kR_rate={controller.kR_rate:.3f}, "
          f"omega_max={controller.omega_max.flatten().tolist()} rad/s")
    print("[setup] note: kR/komega/torque_max belong to the direct-torque "
          "head and are not used when PX4 closes the body-rate loop")
    cal = ThrustCalibration.load(args.calib)
    print(f"[setup] {cal}")
    if abs(cal.thrust_n(cal.normalized(params.mass * 9.80665))
           - params.mass * 9.80665) > 0.5:
        print("[setup] WARN calibration cannot represent hover thrust for "
              f"mass {params.mass} kg — wrong file?")

    compensator = None
    if args.adaptive:
        settings = args.adaptive_settings
        compensator = ResNetCompensator(AdaptiveConfig(
            learning_rate=settings.learning_rate,
            phi_max=settings.phi_max,
            sigma_mod=settings.sigma_mod,
            theta_bar=settings.theta_bar,
            k1=settings.k1,
        ))
        if args.weights:
            compensator.load_weights(args.weights)

    monitor = PerformanceMonitor(mass=params.mass, gravity=params.gravity,
                                 theta_bar=(compensator.cfg.theta_bar
                                            if compensator else None))
    viz = UdpViz(addr=(args.viz_host, args.viz_port), enabled=args.viz,
                 source="px4-geometric")
    if args.viz:
        print(f"[viz] streaming PX4 geometric flight to "
              f"udp://{args.viz_host}:{args.viz_port}")
    owned_sim = None
    try:
        owned_sim = _launch_owned_sim(args)
        if args.mode == "lockstep":
            run_lockstep(args, controller, compensator, params, cal, monitor,
                         viz)
        else:
            run_realtime(args, controller, compensator, params, cal, monitor,
                         viz)
    finally:
        try:
            viz.close()
        finally:
            _stop_owned_sim(owned_sim)
        report(monitor, traj_name=args.traj)
        os.makedirs("runs", exist_ok=True)
        tag = args.run_name or f"px4_{args.traj}{'_resnet' if compensator else ''}"
        path = os.path.join("runs", f"log_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        monitor.save_csv(path)
        if compensator:
            compensator.save_weights(path.replace("log_", "weights_")
                                     .replace(".csv", ".npz"))


if __name__ == "__main__":
    main()
