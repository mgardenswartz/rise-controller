# QuadSim Python SDK Control Modes

This document is the in-depth Python-side guide for choosing control modes, switching modes, sending commands, and stepping QuadSim deterministically.

The Unity README should stay focused on simulator setup and architecture. This page owns the mode-by-mode Python API details.

---

## Mental Model

QuadSim has one drone object:

```python
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()
```

That one `Drone` object supports both:

```python
drone.takeoff(3.0)       # high-level scripted command
drone.send_command(...)  # raw low-level command
```

You do not need separate classes for high-level flight versus low-level research control.

There are three related ideas:

| Concept | Meaning |
|--------|---------|
| Control source | Who owns the drone: UI, Internal C#, or External Python |
| Goal mode | What kind of command the controller expects |
| Controller kind | Which onboard controller consumes the command |

Most Python workflows use the `External` source automatically after `sim.connect()`.

---

## Coordinate Frame

The SDK uses **FLU**:

| Axis | Direction |
|------|-----------|
| `X` | Forward |
| `Y` | Left |
| `Z` | Up |

Practical rules:

- Position commands use `z` as altitude.
- Body rates, angular velocity, and wrench torques use body-frame FLU.
- GPS position is world-frame FLU.
- IMU velocity is body-frame FLU.
- Motor passthrough is not a frame; it is motor order.

Motor order is:

```text
FL, FR, BL, BR
```

---

## Choosing a Mode

| Mode / helper | Command | Uses onboard cascade? | Uses actuation model? | Best for |
|--------------|---------|-----------------------|------------------------|----------|
| High-level flight | `takeoff`, `fly_to`, `land` | Yes | Yes | Demos, scripted missions, simple autonomy |
| `position` | `(x, y, z, yaw)` | Full cascade | Yes | Waypoints, navigation, outermost control |
| `velocity` | `(vx, vy, vz, yaw_rate)` | Velocity → angle → rate | Yes | Guidance laws and velocity policies |
| `angle` | `(roll, pitch, yaw_rate, throttle/alt)` | Angle → rate | Yes | Stabilized manual-style control |
| `rate` | `(roll_rate, pitch_rate, yaw_rate, throttle)` | Rate loop | Yes | Inner-loop controller work |
| acceleration helper | `(ax, ay, az, yaw_rate)` | Attitude/rate stack | Yes | Policy/guidance output as acceleration |
| `passthrough` | `(FL, FR, BL, BR)` | No | Yes | PX4 bridge, motor/mixer testing |
| `wrench_allocated` | `(Mx, My, Mz, thrust)` | No cascade | Yes, through allocator | Geometric controllers and feasibility studies |
| `wrench` | `(Mx, My, Mz, thrust)` | No cascade | No | Idealized sanity checks |

Recommended defaults:

- Use **high-level methods** for a demo.
- Use **position** or **velocity** for autonomy/navigation logic.
- Use **wrench_allocated** for a geometric controller that should respect motor feasibility.
- Use **wrench** only to isolate controller math from actuator limits.
- Use **passthrough** for PX4/SITL/HIL or motor allocation debugging.

---

## One-Shot Commands vs Lockstep Commands

### One-shot

A one-shot command sets the command and lets the simulator keep running.

```python
drone.set_mode("velocity")
drone.send_command(vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0)
```

Use this for wall-clock demos or interactive scripts.

### Lockstep

A lockstep loop applies each command, advances physics, and returns fresh
sensors while owning timing and cleanup:

```python
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()

    with sim.lockstep(drone, control_hz=50.0, speed=0) as loop:
        sensors = drone.get_sensors()

        while loop.time < 10.0:
            result = controller(sensors=sensors, time=loop.time, dt=loop.dt)
            sensors = loop.step_with_wrench(
                tx=result.tx,
                ty=result.ty,
                tz=result.tz,
                thrust=result.thrust,
            )
```

Use this for tuning, learning, and reproducible experiments.

The loop exposes every `step_with_*` method found on its bound `Drone`, plus
`loop.step(callable, **command)` as an escape hatch for non-drone adapters.
It injects `count` itself, so supplying `count` is an error. Each underlying
composite still performs command + physics + sensor readback in one RPC.

At 250 Hz physics and 50 Hz control, the loop realizes:

```python
loop.physics_steps == 5
loop.dt == 0.02
loop.actual_hz == 50.0
```

`loop.time` is experiment time since loop entry (`tick * dt`). It deliberately
does not include reset or settle steps performed before entry and is not
Unity's absolute simulation clock. Read `sim.get_status().sim_time` explicitly
when absolute simulation time is required.

### Rate quantization

Physics-step counts are whole numbers. A 60 Hz request on 250 Hz physics uses
four steps and realizes 62.5 Hz. The default `strict_rate=True` rejects a
relative rate error over 1%, protecting rate-sensitive gains and learned
models. Opt into quantization explicitly:

```python
with sim.lockstep(
    drone,
    control_hz=60.0,
    strict_rate=False,
) as loop:
    print(loop.actual_hz)  # 62.5
```

That logs the requested rate, realized rate, and physics-step count once on
entry.

### Wall-speed cap

`speed` limits wall execution without changing simulated control timing:

```text
speed=0     unlimited; no sleep call is made
speed=1     realtime
speed=2     twice realtime
speed=0.5   half realtime
```

Positive values use absolute wall targets so oversleep does not accumulate.
`loop.achieved_speed` reports the measured faster-than-realtime ratio.

### Composite steps versus `send_*`

Use `send_*` in a free-running simulator: it sets a command and returns without
advancing physics. Use `step_with_*` directly when manually managing a paused
simulator, or through `LockstepLoop` for ordinary deterministic experiments.
Neither family replaces the other.

### Advanced: manual lockstep mechanics

This is the timing logic supplied by `sim.lockstep(...)`. Keep it for custom
adapter work; normal controller programs should use the public loop above.

```python
import time
from quadsim import QuadSim

CONTROL_HZ = 50.0
SPEED = 1.0

with QuadSim() as sim:
    drone = sim.drone()

    status = sim.get_status()
    steps_per_tick = max(1, round((1.0 / CONTROL_HZ) / status.fixed_dt))
    actual_dt = steps_per_tick * status.fixed_dt

    sim.pause()

    sensors = drone.get_sensors()
    tick = 0
    wall_origin = time.perf_counter()
    for tick in range(1000):
        command = controller(sensors)

        sensors = drone.step_with_wrench(
            tx=command.tx,
            ty=command.ty,
            tz=command.tz,
            thrust=command.thrust,
            count=steps_per_tick,
        )

        elapsed_sim = (tick + 1) * actual_dt
        target_wall = wall_origin + elapsed_sim / SPEED
        remaining = target_wall - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    sim.resume()
```

Real code must also restore the prior pause state and neutralize the last
command on exceptions and Ctrl+C. The public loop handles those exit paths.

### Verification scripts

From the `QuadSimLib` root, run the fake-sim suite without Unity:

```bash
python -m pip install -e ".[test]"
python -m pytest Testing/test_lockstep.py -v
```

The editable install is important: a `git+https://...` pip install uses the
latest pushed commit rather than uncommitted changes in this checkout.

With a Unity build running and the vehicle resting safely, run:

```bash
python Testing/verify_lockstep_unity.py
```

The smoke script verifies control geometry, tick/time accounting, composite
stepping, cleanup, and restoration of initially running and paused states. Run
it with `--help` to see connection and rate options.

---

## High-Level Flight

High-level methods run Python-side control loops and are meant for readable mission scripts.

```python
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()

    drone.takeoff(altitude=3.0)
    drone.fly_to(x=5.0, y=0.0, z=3.0)
    drone.hover(duration=2.0)
    drone.yaw_to(heading_deg=180.0)
    drone.land()
```

High-level methods are blocking by default.

Async variants return a future:

```python
future = drone.fly_to_async(x=10.0, y=0.0, z=3.0, speed=2.0)

while not future.done:
    sensors = drone.get_sensors()
```

Use high-level flight when you care about the mission behavior, not raw controller inputs.

---

## `position` Mode

Position mode is the outermost cascaded mode.

You command:

```text
x, y, z, yaw
```

Meaning:

| Field | Meaning | Units |
|-------|---------|-------|
| `x` | Forward/world X position | m |
| `y` | Left/world Y position | m |
| `z` | Altitude/up position | m |
| `yaw` | Heading | deg |

Example:

```python
drone.set_mode("position")
drone.send_command(x=5.0, y=0.0, z=3.0, yaw=90.0)
```

Lockstep:

```python
sensors = drone.step_with_command(
    mode="position",
    x=5.0,
    y=0.0,
    z=3.0,
    w=90.0,
    count=5,
)
```

Use position mode for waypoint tracking, mission scripts, and testing the full cascade.

---

## `velocity` Mode

Velocity mode bypasses the position PID and commands the velocity stage.

You command:

```text
vx, vy, vz, yaw_rate
```

Meaning:

| Field | Meaning | Units |
|-------|---------|-------|
| `vx` | Forward velocity command | m/s |
| `vy` | Left velocity command | m/s |
| `vz` | Up velocity command | m/s |
| `yaw_rate` | Yaw rate | deg/s |

Example:

```python
drone.set_mode("velocity")
drone.send_command(vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0)
```

Lockstep:

```python
sensors = drone.step_with_command(
    mode="velocity",
    x=1.0,   # vx
    y=0.0,   # vy
    z=0.0,   # vz
    w=0.0,   # yaw_rate
    count=5,
)
```

Use velocity mode when Python owns the navigation logic but you still want Unity's onboard cascade to stabilize attitude and motors.

---

## `angle` Mode

Angle mode bypasses position and horizontal velocity control.

You command:

```text
roll, pitch, yaw_rate, throttle
```

Meaning:

| Field | Meaning | Units |
|-------|---------|-------|
| `roll` | Desired roll angle | deg |
| `pitch` | Desired pitch angle | deg |
| `yaw_rate` | Desired yaw rate | deg/s |
| `throttle` | Normalized throttle | 0..1 |

Example:

```python
drone.set_mode("angle")
drone.send_command(roll=5.0, pitch=0.0, yaw_rate=0.0, throttle=0.45)
```

Raw layout:

```python
drone.send_command(x=5.0, y=0.0, z=0.0, w=0.45)
```

Use angle mode for stabilized manual-style control or testing attitude response.

---

## `rate` Mode

Rate mode enters at the innermost body-rate controller.

You command:

```text
roll_rate, pitch_rate, yaw_rate, throttle
```

Meaning:

| Field | Meaning | Units |
|-------|---------|-------|
| `roll_rate` | Body roll rate | deg/s |
| `pitch_rate` | Body pitch rate | deg/s |
| `yaw_rate` | Body yaw rate | deg/s |
| `throttle` | Normalized throttle | 0..1 |

Example:

```python
drone.set_mode("rate")
drone.send_command(roll_rate=0.0, pitch_rate=0.0, yaw_rate=30.0, throttle=0.45)
```

Raw layout:

```python
drone.send_command(x=0.0, y=0.0, z=30.0, w=0.45)
```

Use rate mode for inner-loop work and low-level controller testing while still using Unity's motor path.

---

## Acceleration Helper

Acceleration is best thought of as an SDK convenience interface, not as a replacement for the Unity simulator overview.

You command:

```text
ax, ay, az, yaw_rate
```

Meaning:

| Field | Meaning | Units |
|-------|---------|-------|
| `ax` | Forward acceleration | m/s² |
| `ay` | Left acceleration | m/s² |
| `az` | Up acceleration | m/s² |
| `yaw_rate` | Yaw rate | deg/s |

Example:

```python
drone.send_acceleration(
    ax=1.0,
    ay=0.0,
    az=0.0,
    yaw_rate=0.0,
)
```

Lockstep:

```python
sensors = drone.step_with_acceleration(
    ax=1.0,
    ay=0.0,
    az=0.0,
    yaw_rate=0.0,
    count=5,
)
```

Conceptually:

```text
acceleration command → desired attitude/thrust → rate loop → motors
```

Use this when a planner, learned policy, or outer-loop controller naturally outputs acceleration instead of position, velocity, torque, or motor commands.

Examples:

```python
# Forward acceleration
drone.send_acceleration(ax=1.0, ay=0.0, az=0.0, yaw_rate=0.0)

# Upward acceleration
drone.send_acceleration(ax=0.0, ay=0.0, az=1.0, yaw_rate=0.0)

# Left acceleration while yawing
drone.send_acceleration(ax=0.0, ay=1.0, az=0.0, yaw_rate=20.0)
```

---

## `passthrough` Mode

Passthrough bypasses the cascaded controller and commands the four motors directly.

You command:

```text
FL, FR, BL, BR
```

Each value is normalized:

```text
0.0 = off
1.0 = full command
```

Example:

```python
drone.send_motors(fl=0.5, fr=0.5, bl=0.5, br=0.5)
```

Raw mode:

```python
drone.set_mode("passthrough")
drone.send_command(x=0.5, y=0.5, z=0.5, w=0.5)
```

Lockstep:

```python
sensors = drone.step_with_motors(
    fl=0.5,
    fr=0.5,
    bl=0.5,
    br=0.5,
    count=5,
)
```

Use passthrough for:

- PX4 SITL/HIL bridge actuator outputs.
- Testing motor order.
- Testing the actuation model without the controller.
- Mixer/allocation debugging.

---

## `wrench_allocated` Mode

Allocated wrench is the main low-level mode for geometric controllers.

You command:

```text
Mx, My, Mz, thrust
```

Meaning:

| Field | Meaning | Units |
|-------|---------|-------|
| `Mx` / `tx` | Body roll torque | N·m |
| `My` / `ty` | Body pitch torque | N·m |
| `Mz` / `tz` | Body yaw torque | N·m |
| `thrust` | Total thrust along body-up | N |

Example:

```python
drone.send_wrench(
    tx=0.0,
    ty=0.1,
    tz=0.0,
    thrust=7.2,
)
```

Lockstep:

```python
sensors = drone.step_with_wrench(
    tx=0.0,
    ty=0.1,
    tz=0.0,
    thrust=7.2,
    count=5,
)
```

Conceptually:

```text
requested body wrench
  → effectiveness-matrix allocator
  → per-motor normalized commands
  → actuation model
  → Rigidbody force/torque
```

Use allocated wrench when you want the controller to face the real actuator limits. This is the correct mode for testing whether a geometric controller can track a trajectory with the available motors.

Important behavior:

- The requested wrench may be distorted under saturation.
- Equal thrust should not create roll/pitch torque if geometry and center of mass are consistent.
- Telemetry can show the motor outputs and per-motor thrusts after allocation.

Telemetry check:

```python
sensors = drone.step_with_wrench(tx, ty, tz, thrust, count=5)
tel = drone.get_telemetry()

print(tel.motors)          # normalized FL, FR, BL, BR
print(tel.motor_thrusts)   # Newtons, if exposed by your SDK version
```

---

## `wrench` Mode

Direct wrench applies force/torque straight to the Unity Rigidbody.

You command the same logical values:

```text
Mx, My, Mz, thrust
```

Example:

```python
drone.send_wrench_bypass(
    tx=0.0,
    ty=0.1,
    tz=0.0,
    thrust=7.2,
)
```

Lockstep:

```python
sensors = drone.step_with_wrench_bypass(
    tx=0.0,
    ty=0.1,
    tz=0.0,
    thrust=7.2,
    count=5,
)
```

Conceptually:

```text
requested body wrench → Rigidbody
```

This bypasses:

- Motor limits.
- Control allocation.
- Mixer geometry.
- Per-motor saturation.
- Actuation model.

Use direct wrench for:

- Sanity-checking controller math.
- Comparing idealized versus physically allocated behavior.
- Isolating whether a tracking issue comes from the controller or the allocator/actuator path.

Do not use direct wrench as the final physical validation path.

---

## `none` Mode

`none` clears commanded control.

```python
drone.set_mode("none")
drone.send_command(x=0.0, y=0.0, z=0.0, w=0.0)
```

Use it for reset/cleanup or to intentionally let the vehicle fall/free-run.

---

## Switching Controllers

If the Unity build exposes multiple controller kinds, switch them from Python:

```python
drone.set_controller("cascade")
```

Potential controller strings:

```text
cascade
geometric
```

The current production path uses the cascaded Unity controller for onboard modes. Python-side geometric controllers usually compute a wrench and send it through `wrench_allocated`.

---

## Mode Switching Examples

### High-level takeoff, then raw velocity

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

    drone.hover(2.0)
    drone.land()
```

### Geometric controller in lockstep

```python
from quadsim import QuadSim

CONTROL_HZ = 50.0

with QuadSim() as sim:
    drone = sim.drone()

    with sim.lockstep(drone, control_hz=CONTROL_HZ, speed=0) as loop:
        controller.set_timestep(loop.dt)
        sensors = drone.get_sensors()

        while loop.tick < 3000:
            state = sensors_to_state(sensors)
            traj = trajectory.sample(loop.time)

            result = controller.compute_control(state, traj)
            tx, ty, tz = result["torque"]
            thrust = result["thrust"]

            sensors = loop.step_with_wrench(
                tx=tx, ty=ty, tz=tz, thrust=thrust
            )
```

### PX4-style motor passthrough loop

```python
from quadsim import QuadSim

with QuadSim() as sim:
    drone = sim.drone()
    sim.pause()

    sensors = drone.get_sensors()

    while True:
        motors = px4_or_mixer_step(sensors)  # returns FL, FR, BL, BR

        sensors = drone.step_with_motors(
            fl=motors[0],
            fr=motors[1],
            bl=motors[2],
            br=motors[3],
            count=1,
        )
```

---

## Reading Sensors

```python
sensors = drone.get_sensors()

pos = sensors.gps_position
q = sensors.imu_orientation
vel_body = sensors.imu_vel
omega_body = sensors.imu_ang_vel
accel_body = sensors.imu_accel
```

Common controller pattern:

```python
import numpy as np

def sensors_to_state(sensors):
    q = sensors.imu_orientation
    R = quaternion_to_rotation_matrix(q)

    pos = np.array(sensors.gps_position).reshape(3, 1)
    vel_body = np.array(sensors.imu_vel).reshape(3, 1)
    vel_world = R @ vel_body
    omega = np.array(sensors.imu_ang_vel).reshape(3, 1)

    return {
        "pos": pos,
        "vel": vel_world,
        "R": R,
        "omega": omega,
    }
```

---

## Reading Telemetry

Telemetry is controller/actuator state. Sensors are measured state.

```python
tel = drone.get_telemetry()

print(tel.mode)
print(tel.controller)
print(tel.motors)
print(tel.desired_rates_deg)
print(tel.desired_angles_deg)
print(tel.desired_vel)
print(tel.external_cmd)
```

Use telemetry to debug:

- Which mode Unity thinks it is in.
- What command was received.
- Motor outputs after allocation.
- Desired rates/angles/velocity generated by the cascade.

---

## Reset Pattern for Experiments

For deterministic trials, reset pose, clear physics, clear controller state, then step a few ticks before starting your cost integration.

```python
sim.pause()

drone.send_wrench(tx=0.0, ty=0.0, tz=0.0, thrust=0.0)
drone.reset_pose(x=0.0, y=0.0, z=5.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
drone.reset_physics()
drone.reset_controller()

sim.step(count=10)

sensors = drone.get_sensors()
```

For trajectory tracking, reset to the trajectory's actual `t=0` pose and heading.

---

## Common Gotchas

### Direct wrench and allocated wrench are not the same

Direct wrench applies exactly what you requested. Allocated wrench asks the motors to produce it. If the command is infeasible, the allocator saturates and the actual wrench differs.

That difference is the point of allocated mode.

### Hover thrust is in Newtons for wrench modes

For wrench control:

```python
hover_thrust = mass * 9.81
```

For a 0.73 kg drone:

```python
hover_thrust ≈ 7.16 N
```

Do not send normalized throttle to `thrust` in wrench modes.

### Motor commands are normalized

For passthrough:

```python
0.0 <= motor <= 1.0
```

These are not Newtons.

### Avoid per-tick print spam

Printing every 250 Hz or 50 Hz tick can dominate runtime and destroy faster-than-real-time performance. Log sparsely or write arrays to memory and save at the end.

### Keep frame conversion at the boundary

Python should send FLU values. Unity's adapter and sensor stack own Unity-frame conversion. Do not scatter manual sign flips through controllers.

### Use lockstep for Optuna and learning

Do not tune controllers using wall-clock sleeps. Use `sim.lockstep(...)` so
the controller rate and physics stepping are deterministic.

---

## Quick Reference

### Mode strings

```text
none
position
velocity
angle
rate
passthrough
wrench_allocated
wrench
```

### Raw `send_command` layout

| Mode | `x` | `y` | `z` | `w` |
|------|-----|-----|-----|-----|
| `position` | x m | y m | z m | yaw deg |
| `velocity` | vx m/s | vy m/s | vz m/s | yaw_rate deg/s |
| `angle` | roll deg | pitch deg | yaw_rate deg/s | throttle 0..1 |
| `rate` | roll_rate deg/s | pitch_rate deg/s | yaw_rate deg/s | throttle 0..1 |
| `passthrough` | FL | FR | BL | BR |
| `wrench_allocated` | Mx N·m | My N·m | Mz N·m | thrust N |
| `wrench` | Mx N·m | My N·m | Mz N·m | thrust N |

### Mode-specific helpers

| Helper | Equivalent mode |
|--------|-----------------|
| `send_motors(...)` | `passthrough` |
| `step_with_motors(...)` | `passthrough` + step |
| `send_wrench(...)` | `wrench_allocated` |
| `step_with_wrench(...)` | `wrench_allocated` + step |
| `send_wrench_bypass(...)` | `wrench` |
| `step_with_wrench_bypass(...)` | `wrench` + step |
| `send_acceleration(...)` | SDK acceleration helper |
| `step_with_acceleration(...)` | SDK acceleration helper + step |

---

## Where This Fits

- Unity README: install, scene setup, architecture, sensors, configuration.
- This doc: Python control modes and command syntax.
- Research scripts: controllers, trajectories, tuning objectives, plots.
