<p align="center">
  <h1 align="center">QuadSim Python SDK</h1>
  <p align="center">
    Research-oriented drone simulation control for Python.
    <br />
    High-level flight commands, raw low-level control, deterministic stepping, and live visualization.
  </p>
</p>

<p align="center">
  <a href="#getting-started">Getting Started</a> •
  <a href="#quick-example">Quick Example</a> •
  <a href="#control-modes">Control Modes</a> •
  <a href="#lockstep-control">Lockstep Control</a> •
  <a href="#coordinate-frame">Coordinate Frame</a> •
  <a href="#api-overview">API Overview</a> •
  <a href="#project-structure">Project Structure</a>
</p>

---

## What is QuadSim?

QuadSim is a research-oriented quadrotor simulation platform built in Unity for controls research, autonomy development, and sim-to-real workflows.

This package is the **Python SDK**. It connects to a running QuadSim Unity scene over ZeroMQ and lets Python code command drones, read sensors, switch control modes, step the simulator deterministically, and stream live visualization data.

The Unity runtime is installed separately:

- Unity plugin: [github.com/ninonick0607/Unity-QuadSim-Plugin](https://github.com/ninonick0607/Unity-QuadSim-Plugin)

The SDK exposes two layers through one simple object model:

```text
QuadSim       ← sim/world entry point, owns the connection
  └─ Drone    ← high-level flight, raw commands, sensors, telemetry, stepping
```

You can move from `takeoff()` to raw `send_command()` or lockstep `step_with_*()` calls without changing objects.

---

## Getting Started

### Requirements

- Python 3.8+
- A running QuadSim Unity scene or headless build
- RPC enabled in Unity through `ExternalRpcAdapter`
- Unity command port, default `5555`
- Unity telemetry port, default `5556`

### Install

From PyPI:

```bash
pip install quadsim-sdk
```

From source:

```bash
git clone https://github.com/ninonick0607/QuadSimLib.git
cd QuadSimLib
pip install -e .
```

This installs the Python dependencies, including `pyzmq` and `msgpack`.

For SDK development and the verification suite, install the local checkout
with its test extra:

```bash
python -m pip install -e ".[test]"
```

### Verify Connection

Start the Unity scene or headless build first, then run:

```python
from quadsim import QuadSim

with QuadSim() as sim:
    print(sim.get_status())
```

If it prints status information, the SDK is connected.

---

## Quick Example

```python
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()

    drone.takeoff(altitude=3.0)
    drone.fly_to(x=5, y=0, z=3)
    drone.hover(duration=2.0)
    drone.yaw_to(heading_deg=180)
    drone.fly_path([(5, 5, 3), (0, 5, 3), (0, 0, 3)])
    drone.land()
```

---

## Reinforcement Learning

QuadSim includes deterministic atomic multi-agent reset/stepping, stable agent
identities, privileged ENU/FLU state, per-drone RGB/grayscale/depth capture,
pooled runtime obstacles and goals, and collision events with object IDs.

The complete RL guide, including every RL-facing SDK command and runnable
training-loop examples, lives here:

- [QuadSim Reinforcement Learning Guide](docs/reinforcement_learning.md)
- [Gymnasium + Stable-Baselines3 RGB-D goal-hover task](Examples/rl_navigation/README.md)

---

## Control Modes

QuadSim supports high-level scripted flight, cascaded controller modes, motor passthrough, allocated wrench control, and direct wrench control.

The detailed control-mode documentation lives in:

- [docs/control_modes.md](docs/control_modes.md)

That page covers:

- When to use each mode.
- Exact `send_command(...)` layouts.
- How to call `set_mode(...)`.
- How to use one-shot commands versus lockstep `step_with_*` commands.
- Difference between position, velocity, angle, rate, passthrough, allocated wrench, direct wrench, and acceleration helper APIs.
- How to read sensors and telemetry after each command.

Minimal raw example:

```python
import time
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()

    drone.takeoff(3.0)

    drone.set_mode("velocity")
    for _ in range(100):
        drone.send_command(vx=1.0, vy=0.0, vz=0.0, yaw_rate=10.0)
        time.sleep(0.02)

    drone.hover(duration=2.0)
    drone.land()
```

---

## Lockstep Control

For tuning, learning, and deterministic experiments, bind the drone to a
lockstep loop. The loop owns pause restoration, physics-step quantization,
experiment time, optional wall pacing, and the final zero-wrench command.

```python
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()

    with sim.lockstep(drone, control_hz=50.0, speed=0) as loop:
        sensors = drone.get_sensors()

        while loop.time < 10.0:
            tx, ty, tz, thrust = my_controller(
                sensors=sensors,
                time=loop.time,
                dt=loop.dt,
            )

            sensors = loop.step_with_wrench(
                tx=tx, ty=ty, tz=tz, thrust=thrust
            )
```

The loop reflects every `drone.step_with_*` composite onto itself, injects its
owned `count`, and returns the composite's post-step sensors. The current
methods are:

- `loop.step_with_wrench(...)`
- `loop.step_with_wrench_bypass(...)`
- `loop.step_with_acceleration(...)`
- `loop.step_with_motors(...)`
- `loop.step(callable, **command)` for non-`Drone` callables accepting `count=`

Composite `step_with_*` calls apply a command, advance physics, and return
sensors in one RPC round trip. They remain public and can still be used
directly. In contrast, `send_*` methods are fire-and-forget commands for a
free-running simulator; the two families serve different clock regimes.

### Rate quantization and speed

Unity can advance only whole physics steps. At 250 Hz physics, 50 Hz control is
exactly five steps. A 60 Hz request realizes 62.5 Hz on four steps. Strict rate
checking is on by default and rejects a relative error over 1%; pass
`strict_rate=False` to accept quantization and log the realized geometry once.

Use `loop.dt` and `loop.actual_hz` for the realized rate. `loop.time` is
experiment time since loop entry (`loop.tick * loop.dt`), not Unity's absolute
simulation clock and not time spent in reset/settle steps outside the loop.
Call `sim.get_status()` explicitly when absolute simulation time is needed.

Wall-speed caps keep the existing convention:

```text
speed=0     unlimited (no sleep call)
speed=1     realtime
speed=2     twice realtime
speed=0.5   half realtime
```

Positive caps use absolute wall targets, so sleep error does not accumulate.
`loop.achieved_speed` reports measured simulated-seconds per wall-second.

### Advanced: how the loop works

The equivalent manual geometry is useful when building an adapter, but is no
longer required in ordinary experiments:

```python
import time

status = sim.get_status()
steps_per_tick = max(1, round((1.0 / CONTROL_HZ) / status.fixed_dt))
actual_dt = steps_per_tick * status.fixed_dt
sim.pause()

tick = 0
wall_origin = time.perf_counter()
while tick * actual_dt < DURATION:
    sensors = drone.step_with_wrench(
        tx=tx, ty=ty, tz=tz, thrust=thrust, count=steps_per_tick
    )
    tick += 1
    if SPEED > 0:
        target = wall_origin + (tick * actual_dt) / SPEED
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
```

Production code must also restore the prior pause state and neutralize the
drone on every exit path; `sim.lockstep(...)` supplies that exception-safe
cleanup.

### Verify the implementation yourself

Run the no-Unity unit suite:

```bash
python -m pip install -e ".[test]"
python -m pytest Testing/test_lockstep.py -v
```

Use the editable local install above when testing workspace changes. Installing
`git+https://github.com/.../QuadSimLib.git` tests the latest pushed commit, which
may not yet contain local milestone work.

With a Unity build running and the vehicle resting safely, run the integration
smoke test:

```bash
python Testing/verify_lockstep_unity.py
```

The live script checks realized geometry, tick/time accounting, one-call
composite stepping, zero-command cleanup, and restoration of both initially
running and initially paused simulator states. Use `--help` for connection,
rate, speed, and quantization options.

---

## Async Flight with Polling

High-level methods also have `_async` variants that return a future-like handle:

```python
import time
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()
    drone.takeoff(3.0)

    future = drone.fly_to_async(x=10, y=0, z=3, speed=2.0)

    while not future.done:
        pos = drone.get_position()
        print(f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        time.sleep(0.5)

    drone.land()
```

---

## Coordinate Frames

QuadSim distinguishes **world** frames from **body** frames. Mixing the
two is the classic drone-sim bug, so every field and command below is
labeled with both the frame *type* and the *convention*.

**World frames** (fixed to the map, independent of vehicle attitude):

| Convention | Axes | Used for |
|------------|------|----------|
| **ENU** | x=East, y=North, z=Up | Canonical. All positions, waypoints, `reset_pose`, viz. `z` is altitude. |
| **Unity world** | x=East, y=Up, z=North | Native Unity scene axes. Place objects with Unity `y` as altitude; Unity `(x, y, z)` maps to ENU `(x, z, y)`. |
| **NED** | x=North, y=East, z=Down | PX4/MAVLink convention. Available via `*_ned` accessors and `gps_vel_ned`. |
| WGS84 | lat, lon, ellipsoidal alt | `gps_lat/lon/alt`, computed in-sim. |

**Body frames** (attached to the vehicle):

| Convention | Axes | Used for |
|------------|------|----------|
| **FLU** | x=Forward, y=Left, z=Up | Canonical (ROS). Default for IMU/mag readings and velocity/accel/wrench commands. |
| **Unity body** | x=Forward, y=Up, z=Left | Native Unity `Transform`/`Rigidbody` local axes. Unity body `(x, y, z)` maps to FLU `(x, z, y)`. |
| **FRD** | x=Forward, y=Right, z=Down | PX4 HIL convention. Active when the sim is in PX4 mode; advertised by `SensorData.frame`. |

### Sensor fields

| Field | Frame | Notes |
|-------|-------|-------|
| `gps_position` | World **ENU** | Always ENU, in every sim mode. `z` = altitude. |
| `gps_lat/lon/alt` | WGS84 | Ellipsoidal altitude. |
| `gps_vel_ned` | World **NED** | `(vn, ve, vd)`. PX4 `HIL_GPS` wire quantity. |
| `imu_vel` | **Body** FLU/FRD | Body-frame velocity — **not world**. Rotate by `imu_orientation` for world, or use `velocity_enu`. |
| `imu_ang_vel` | Body FLU/FRD | rad/s. |
| `imu_accel` | Body FLU/FRD | Specific force, gravity included (hover reads ~(0,0,+9.81) FLU). |
| `imu_attitude` | Body convention | RPY degrees. Display-oriented; prefer the quaternion for math. |
| `imu_orientation` | body(FLU) -> world(ENU) | Quaternion `(x, y, z, w)`. Only well-defined in FLU mode. |
| `mag_field` | Body FLU/FRD | Gauss. |
| `frame` | — | `"flu"` or `"frd"`: body convention of the `imu_*`/`mag_*` fields. World fields are unaffected. |

### Explicit-frame accessors

When you want a specific convention rather than "whatever mode the sim
is in", use the explicit accessors — all pure axis relabels, no hidden
rotations:

```python
s = drone.get_sensors()
s.position_enu    # (E, N, Up)          — alias of gps_position
s.position_ned    # (N, E, Down)
s.velocity_enu    # (ve, vn, vu)        — world, from the GPS channel
s.velocity_ned    # (vn, ve, vd)        — alias of gps_vel_ned
s.vel_body_flu    # body velocity, FLU regardless of sim mode
s.vel_body_frd    # body velocity, FRD regardless of sim mode
s.ang_vel_flu / s.ang_vel_frd
s.accel_flu   / s.accel_frd
s.mag_flu     / s.mag_frd

drone.get_position()       # world ENU
drone.get_position_ned()   # world NED
drone.get_velocity()       # BODY frame (sim convention) — not world!
drone.get_velocity_enu()   # world ENU (GPS channel)
drone.get_velocity_ned()   # world NED (GPS channel)
```

Relabel math, for reference (both are proper rotations, so vectors and
pseudovectors transform identically):

```
ENU <-> NED : (n, e, d) = (enu.y, enu.x, -enu.z)
FLU <-> FRD : (x, -y, -z)                    # self-inverse
```

### Commands

| Command | Frame |
|---------|-------|
| `position` mode, `fly_to`, `fly_path`, `reset_pose`, viz | World **ENU** + yaw (deg) |
| `velocity` mode | **Body** FLU + yaw rate — the sim's velocity loop tracks body-frame velocity |
| `acceleration` mode | Body FLU |
| `wrench` / `wrench_bypassed` torques | Body FLU (thrust is a scalar along body z) |
| `passthrough` motors | Motor order: FL, FR, BL, BR |

The Unity adapter handles all Unity-frame conversion internally
(`Frames.cs` is the single conversion boundary); FRD/NED-to-MAVLink
conversion happens only inside `Px4Link`.

---

## Environment / Disturbance

Toggle the scene wind/disturbance field from Python:

```python
drone.set_wind(enabled=True)
drone.set_wind(enabled=True, wind_speed=8.0)
drone.set_wind(enabled=False)
```

With no `wind_speed`, each Unity `WindModule` keeps its scene-configured speed.

---

## Headless Visualization

The SDK includes `UdpViz`, a fire-and-forget UDP sender for live headless visualization.

```python
from quadsim import UdpViz

viz = UdpViz(source="my-run")
viz.path(planned_points)

# inside the loop
viz.sample(t, pos, target=target_pos, err=err, speed=speed, trial=trial_id)
```

Run the viewer separately:

```bash
python live_viewer.py --port 14660
```

Positions are FLU `[x, y, z]`, z-up. If no viewer is listening, packets are dropped and the run continues.

---

## API Overview

### `QuadSim` — Sim / World

| Method | Description |
|--------|-------------|
| `connect()` | Connect to Unity server |
| `disconnect()` | Clean disconnect |
| `drone()` | Get a `Drone` handle |
| `lockstep(drone, control_hz, speed, ...)` | Exception-safe deterministic control loop |
| `get_status()` | Sim time, fixed dt, pause state, authority |
| `pause()` / `resume()` | Pause/resume simulation |
| `step(count)` | Advance N physics steps |
| `set_time_scale(scale)` | Speed up / slow down free-run mode |
| `reset()` | Reset entire simulation |

### `Drone` — Flight / Control

| Method | Description |
|--------|-------------|
| `takeoff(altitude, speed)` | Climb and stabilize |
| `land(speed)` | Descend to ground |
| `hover(duration)` | Hold position |
| `fly_to(x, y, z, speed, yaw)` | Fly to a position |
| `fly_path(waypoints, speed)` | Follow a waypoint sequence |
| `yaw_to(heading_deg)` | Rotate to heading |
| `set_mode(mode)` | Set raw goal mode |
| `set_controller(controller)` | Switch controller kind |
| `send_command(...)` | Send raw Axis4 command |
| `send_motors(...)` | Send motor passthrough command |
| `send_wrench(...)` | Send allocated wrench command |
| `send_wrench_bypass(...)` | Send direct Rigidbody wrench |
| `step_with_*` | Apply command + step + return sensors |
| `get_sensors()` | Full sensor snapshot |
| `get_telemetry()` | Controller state and motor outputs |
| `reset_pose(...)` | Teleport pose |
| `reset_physics()` | Zero velocities |
| `reset_controller()` | Clear controller state |

### Streaming

| Method | Description |
|--------|-------------|
| `subscribe_sensors(callback, hz)` | Push-based sensor data |
| `subscribe_telemetry(callback, hz)` | Push-based telemetry |
| `subscribe(callback, topics, hz)` | Raw topic subscription |
| `unsubscribe()` | Stop streaming |

---

## API Reference

### SensorData Fields

```python
sensors = drone.get_sensors()

sensors.gps_position
sensors.imu_attitude
sensors.imu_vel
sensors.imu_accel
sensors.imu_ang_vel
sensors.imu_orientation
sensors.gps_lat
sensors.gps_lon
sensors.gps_alt
sensors.gps_vel_ned
sensors.baro_pressure_hpa
sensors.baro_temperature_c
sensors.mag_field
```

### Telemetry Fields

```python
telem = drone.get_telemetry()

telem.drone_id
telem.mode
telem.controller
telem.motors
telem.motor_thrusts
telem.desired_rates_deg
telem.desired_angles_deg
telem.desired_vel
telem.external_cmd
```

### SimStatus Fields

```python
status = sim.get_status()

status.is_paused
status.time_scale
status.sim_time
status.fixed_dt
status.authority
status.client_connected
```

### Tuning Parameters

Set these on the `Drone` object before or during high-level flight:

```python
drone.position_tolerance = 0.5
drone.altitude_tolerance = 0.3
drone.landing_altitude = 0.15
drone.default_speed = 2.0
drone.control_loop_hz = 50.0
drone.default_leg_timeout = 30.0
drone.use_velocity_mode_navigation = False
```

### Exceptions

```python
from quadsim import QuadSimError, ConnectionError, CommandError, TimeoutError, ProtocolError
```

| Exception | When |
|-----------|------|
| `ConnectionError` | Not connected or connection denied |
| `CommandError` | Authority rejected, invalid drone, invalid mode |
| `TimeoutError` | RPC timeout or flight command timeout |
| `ProtocolError` | Wire-level serialization issue |

All inherit from `QuadSimError`.

---

## How It Works

The SDK communicates with QuadSim's Unity runtime over ZeroMQ:

| Socket | Default port | Purpose |
|--------|--------------|---------|
| REQ/REP | `5555` | Commands, queries, mode changes, composite steps |
| PUB/SUB | `5556` | Streaming telemetry |

Messages are serialized with MessagePack. A background heartbeat keeps the connection alive. Socket handling is internal to `_transport.py`.

---

## Project Structure

```text
QuadSimLib/
├── quadsim/
│   ├── __init__.py
│   ├── sim.py
│   ├── drone.py
│   ├── viz.py
│   ├── _transport.py
│   ├── _control_loops.py
│   ├── types.py
│   ├── exceptions.py
│   └── future.py
├── docs/
│   └── control_modes.md
├── live_viewer.py
├── Examples/
│   ├── flight_demo.py
│   └── async_demo.py
├── Testing/
│   ├── quadsim_test_client.py
│   ├── quadsim_test_commands.py
│   └── test_high_level.py
└── pyproject.toml
```

---

## Related

- Unity runtime/plugin: [github.com/ninonick0607/Unity-QuadSim-Plugin](https://github.com/ninonick0607/Unity-QuadSim-Plugin)
- Main Unity development repo: [github.com/ninonick0607/Unity_QuadSim](https://github.com/ninonick0607/Unity_QuadSim)

---

## License

MIT
