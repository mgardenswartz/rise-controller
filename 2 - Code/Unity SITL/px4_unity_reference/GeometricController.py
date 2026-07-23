#!/usr/bin/env python3
from contextlib import ExitStack
import os
import time
import numpy as np
from trajectories import Figure8Trajectory, AcroOvalTrajectory, GroundHoverFlipLandTrajectory
from utils import PerformanceMonitor, build_path_preview, sensors_to_state, reset_to_trajectory_start
from quadsim import QuadSim, UdpViz
from resnet_compensator import (ResNetCompensator, AdaptiveConfig)
from aviary_geometric_controller.controller import (
    GeometricControllerLee as _GeometricControllerCore,
)
from project_config import (
    DEFAULT_GEOMETRIC_CONFIG,
    load_geometric_config,
    load_vehicle_profile,
)


# ============================================================================
# Drone Parameters
# ============================================================================

class DroneParameters:
    """Controller plant parameters loaded from a named vehicle profile."""

    def __init__(self, profile="sentinel"):
        self.profile = load_vehicle_profile(profile)
        self.name = self.profile.name
        self.unity_config = self.profile.unity_config
        self.mass = self.profile.mass
        self.gravity = self.profile.gravity
        self.Jxx, self.Jyy, self.Jzz = self.profile.inertia
        self.J = np.diag(self.profile.inertia)
        self.arm_length = self.profile.arm_length
        self.max_thrust = self.profile.max_collective_thrust


class LeeGainParameters:
    """Direct-torque and PX4 rate-interface gains from a vehicle profile."""

    def __init__(self, profile="sentinel"):
        self.profile = load_vehicle_profile(profile)
        self.kx = self.profile.kx
        self.kv = self.profile.kv
        self.kR = self.profile.kR
        self.komega = self.profile.komega
        self.torque_max = np.asarray(
            self.profile.torque_max, dtype=float
        ).reshape(3, 1)
        self.kR_rate = self.profile.kR_rate
        self.omega_max = np.asarray(
            self.profile.omega_max, dtype=float
        ).reshape(3, 1)


class GeometricControllerLee(_GeometricControllerCore):
    """Compatibility wrapper around the hardware-shareable controller core."""

    def __init__(self, params: DroneParameters, gains: LeeGainParameters = None):
        super().__init__(params, gains or LeeGainParameters(params.profile))


def main():
    config = load_geometric_config(DEFAULT_GEOMETRIC_CONFIG)

    # --- Configuration (JSON files only) ---
    vehicle_name = config.vehicle
    trajectory_name = config.trajectory
    CONTROL_HZ = config.control_hz
    DURATION = config.duration
    SPEED = config.speed
    WIND_MODE = config.wind_mode
    WIND_SPEED = config.wind_speed
    USE_ADAPTIVE = config.adaptive.enabled
    SEND_VIZ = config.visualization_enabled
    VIZ_EVERY_N = config.visualization_every_n
    DEBUG_TICKS = config.debug_ticks
    RUN_NAME = config.run_name
    if DURATION <= 0.0:
        raise ValueError("duration must be greater than zero")
    if SPEED < 0.0:
        raise ValueError("speed must be greater than or equal to zero")

    RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = config.output_dir
    WEIGHT_SNAPSHOT_EVERY_S = config.weight_snapshot_every_s

    # --- Setup ---
    profile = load_vehicle_profile(vehicle_name)
    params = DroneParameters(profile)
    gains = LeeGainParameters(profile)
    controller = GeometricControllerLee(params, gains)

    print(f"[config] {config.source}")
    print(
        f"[vehicle] {profile.name} -> Unity config '{profile.unity_config}', "
        f"gains={profile.gains_status}"
    )
    print(
        "[vehicle] Confirm Unity is using that drone config; selecting a Python "
        "profile does not switch the Unity model."
    )

    compensator = None
    if USE_ADAPTIVE:
        adaptive_cfg = AdaptiveConfig(
            learning_rate=config.adaptive.learning_rate,
            phi_max=config.adaptive.phi_max,
            sigma_mod=config.adaptive.sigma_mod,
            theta_bar=config.adaptive.theta_bar,
            k1=config.adaptive.k1,
        )
        compensator = ResNetCompensator(adaptive_cfg)

    monitor = PerformanceMonitor(
        mass=params.mass, gravity=params.gravity,
        theta_bar=(compensator.cfg.theta_bar if compensator else None),
    )

    trajectories = {
        "acro_oval": AcroOvalTrajectory,
        "backflip": GroundHoverFlipLandTrajectory,
        "figure8": Figure8Trajectory,
    }
    if trajectory_name == "figure8":
        traj_gen = Figure8Trajectory(heading_mode=config.figure8_heading)
    else:
        traj_gen = trajectories[trajectory_name]()

    viz = UdpViz(
        source=(RUN_NAME if RUN_NAME else "geometric-controller"),
        enabled=SEND_VIZ,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with QuadSim(
            host=config.host,
            command_port=config.command_port,
            telemetry_port=config.telemetry_port,
    ) as sim:
        if WIND_MODE != "leave":
            resp = sim.set_wind(enabled=(WIND_MODE == "on"), wind_speed=WIND_SPEED)
            print(f"[wind] {resp}")

        drone = sim.drone()

        # Pause first so the drone doesn't freefall during setup
        sim.pause()
        reset_to_trajectory_start(sim, drone, traj_gen)

        loop_stack = ExitStack()
        try:
            loop = loop_stack.enter_context(
                sim.lockstep(drone, control_hz=CONTROL_HZ, speed=SPEED)
            )
            controller.set_timestep(loop.dt)

            sensors = drone.get_sensors()

            controller.reset()
            if compensator:
                compensator.reset(reset_weights=False)
            start_pos = sensors.gps_position

            drone._transport.request("draw_trajectory",
                                     {"points": build_path_preview(traj_gen)})
            if SEND_VIZ:
                viz.path(build_path_preview(traj_gen))
            print(f"[Setup] Drone at: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
            print(f"[Setup] Entering lockstep control loop")

            perf_start = time.perf_counter()
            log_interval = int(CONTROL_HZ)
            last_cmd = (0.0, 0.0, 0.0, 0.0)

            while True:
                sim_time = loop.time

                if sim_time >= DURATION:
                    print(f"\n[Exit] Reached time limit ({sim_time:.1f}s)")
                    break

                # 1. Convert last reply's sensors to controller state
                state = sensors_to_state(sensors)

                if state is None:
                    print("[WARN] Degenerate quaternion — holding wrench, skipping tick")
                    sensors = loop.step_with_wrench(*last_cmd)
                    continue

                current_pos = sensors.gps_position

                # 2. Sample the analytic trajectory
                trajectory = traj_gen.sample(sim_time)

                # 3. Adaptive ResNet disturbance compensation

                phi = None
                trajectory_adapted = trajectory
                if compensator:
                    phi = compensator.step(
                        state,
                        trajectory,
                        loop.dt,
                    )
                    trajectory_adapted = dict(trajectory)
                    trajectory_adapted['accel'] = (
                            trajectory['accel']
                            - phi / params.mass
                    )

                # 4. Compute control
                result = controller.compute_control(
                    state,
                    trajectory_adapted,
                )

                # Get the full weight vector if adaptive control is on
                theta_vec = np.asarray(compensator.theta).flatten() if compensator else None

                # Periodic weight checkpoint (binary .npz — never CSV/console)
                if (compensator and WEIGHT_SNAPSHOT_EVERY_S > 0
                        and loop.tick > 0
                        and loop.tick % int(WEIGHT_SNAPSHOT_EVERY_S * CONTROL_HZ) == 0):
                    compensator.save_weights(
                        os.path.join(OUTPUT_DIR, f"weights_{RUN_TAG}.npz"))

                last_cmd = (float(result['torque'][0]), float(result['torque'][1]),
                            float(result['torque'][2]), float(result['thrust']))

                if SEND_VIZ and loop.tick % VIZ_EVERY_N == 0:
                    pos_f = state["pos"].flatten()
                    target_f = trajectory["pos"].flatten()
                    vel_f = state["vel"].flatten()
                    err = float(np.linalg.norm(pos_f - target_f))

                    viz.sample(
                        sim_time,
                        pos_f,
                        target=target_f,
                        err=err,
                        speed=float(np.linalg.norm(vel_f)),
                        trial=0,
                        phi_norm=(float(np.linalg.norm(phi.flatten()))
                                  if phi is not None else 0.0),
                    )

                sensors = loop.step_with_wrench(*last_cmd)
                # tel = drone.get_telemetry()  # controller outputs — separate from sensors

                monitor.update(state=state, trajectory=trajectory, result=result, phi=phi,
                               theta_norm=(compensator.theta_norm if compensator else None),
                               theta_vector=theta_vec, dt=loop.dt,
                               motor_thrusts=None)

                # 7. Logging
                if loop.tick == 500:
                    elapsed = time.perf_counter() - perf_start
                    print(
                        f"\n[PERF] sim_time={sim_time:.2f}s "
                        f"wall_time={elapsed:.2f}s "
                        f"speedup={sim_time / elapsed:.2f}x "
                        f"ticks/sec={loop.tick / elapsed:.2f}\n"
                    )
                if loop.tick <= DEBUG_TICKS:
                    pos = state['pos'].flatten()
                    vel = state['vel'].flatten()
                    omega = state['omega'].flatten()
                    body_z = state['R'] @ np.array([0, 0, 1])
                    phi_dbg = (compensator.last_phi.flatten()
                               if compensator else np.zeros(3))
                    print(
                        f"[DBG t={loop.tick}] "
                        f"pos=({pos[0]:+6.3f}, {pos[1]:+6.3f}, {pos[2]:+6.3f}) "
                        f"vel=({vel[0]:+5.2f}, {vel[1]:+5.2f}, {vel[2]:+5.2f}) "
                        f"omega=({omega[0]:+5.2f}, {omega[1]:+5.2f}, {omega[2]:+5.2f}) "
                        f"bodyZ=({body_z[0]:+5.2f}, {body_z[1]:+5.2f}, {body_z[2]:+5.2f}) "
                        f"thrust={result['thrust']:6.2f}N "
                        f"torque=({result['torque'][0]:+6.3f}, "
                        f"{result['torque'][1]:+6.3f}, "
                        f"{result['torque'][2]:+6.3f})"
                        f" phi=({phi_dbg[0]:+5.2f}, "
                        f"{phi_dbg[1]:+5.2f}, "
                        f"{phi_dbg[2]:+5.2f})"
                    )

                if loop.tick % log_interval == 0:
                    pos = current_pos
                    wp = trajectory['pos'].flatten()
                    dist = np.linalg.norm(np.array(pos) - np.array(wp))
                    th_dbg = compensator.theta_norm if compensator else 0.0
                    print(
                        f"[t={sim_time:6.2f}s] "
                        f"pos=({pos[0]:+6.2f}, {pos[1]:+6.2f}, {pos[2]:+6.2f}) "
                        f"wp=({wp[0]:+5.1f}, {wp[1]:+5.1f}, {wp[2]:+5.1f}) "
                        f"dist={dist:5.2f}m "
                        f"th_norm={th_dbg:5.2f}"
                    )

        except KeyboardInterrupt:
            print("\n[Exit] Ctrl+C — shutting down")
        finally:

            try:
                loop_stack.close()
            except Exception as e:
                print(f"[Cleanup] lockstep exit failed: {e}")
            try:
                sim.resume()
            except Exception as e:
                print(f"[Cleanup] sim.resume failed: {e}")

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            run_label = RUN_NAME or ("adaptive" if compensator else "baseline")
            tag = f"{run_label}_{RUN_TAG}"

            try:
                monitor.save_csv(os.path.join(OUTPUT_DIR, f"log_{tag}.csv"))
                if compensator and len(monitor.theta_history) > 0:
                    history_path = os.path.join(OUTPUT_DIR, f"weight_history_{RUN_TAG}.npy")
                    np.save(history_path, np.array(monitor.theta_history))
                    print(f"[Monitor] Weight history saved -> {history_path}")
            except Exception as e:
                print(f"[Cleanup] Save failed: {e}")

            if compensator:
                try:
                    compensator.save_weights(
                        os.path.join(OUTPUT_DIR, f"weights_{RUN_TAG}.npz"))
                except Exception as e:
                    print(f"[Cleanup] weight save failed: {e}")

            print("[Exit] Done")


if __name__ == "__main__":
    main()
