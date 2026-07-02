from quadsim import QuadSim

CONTROL_HZ = 50.0

with QuadSim() as sim:
    drone = sim.drone()
    sim.pause()

    status = sim.get_status()
    steps_per_tick = max(1, round((1.0 / CONTROL_HZ) / status.fixed_dt))

    sensors = drone.get_sensors()

    drone.reset_pose(x=0,y=0,z=0)
    for _ in range(5):
        sensors = drone.step_with_acceleration(0,0,10,0,steps_per_tick)

    sim.pause()