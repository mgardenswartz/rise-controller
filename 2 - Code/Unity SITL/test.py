from quadsim import QuadSim
import time

CONTROL_HZ = 50.0
SIM_DURATION = 10.0

with QuadSim() as sim:
    drone = sim.drone()
    sim.pause()

    status = sim.get_status()
    steps_per_tick = max(1, round((1.0 / CONTROL_HZ) / status.fixed_dt))

    sensors = drone.get_sensors()

    drone.reset_pose(x=0.0,y=2.0,z=0.0)
    start_time = time.perf_counter()
    status = sim.get_status()
    initial_sim_time = status.sim_time
    for _ in range(round(CONTROL_HZ * SIM_DURATION)):
        sensors = drone.send_command(
        ax=0.0,
        ay=0.0,
        az=0.0,
        yaw_rate=0.0,
        count=steps_per_tick,   
        )
        
        sensors = drone.get_sensors()
        print(f"Your velocity is {sensors.gps_vel_ned}")

        status = sim.get_status()
        print(f"Sim time is {status.sim_time}")

    end_time = time.perf_counter()

    print(f"Sim ended. Took {round(end_time-start_time,2)} seconds")
    sim.pause()