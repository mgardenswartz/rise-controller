from quadsim import QuadSim
import time

CONTROL_HZ = 50.0
SIM_DURATION = 10.0

FIXED_K1 = 1.31
FIXED_K2 = 0.131
FIXED_K3 = 0.83
FIXED_K_RISE = 0.0287

INIT_X_NED = 0.0 #0.70
INIT_Y_NED = -2.37
INIT_Z_NED = -3.0

INIT_X_UNITY = INIT_Y_NED
INIT_Y_UNITY = -INIT_Z_NED
INIT_Z_NED = INIT_X_NED

with QuadSim() as sim:
    drone = sim.drone()
    sim.pause()

    status = sim.get_status()
    steps_per_tick = max(1, round((1.0 / CONTROL_HZ) / status.fixed_dt))

    sensors = drone.get_sensors()

    drone.reset_pose(x=INIT_X_UNITY,y=INIT_Y_UNITY,z=INIT_Z_NED)
    start_time = time.perf_counter()
    status = sim.get_status()
    initial_sim_time = status.sim_time
    for _ in range(round(CONTROL_HZ * SIM_DURATION)):
        sensors = drone.step_with_acceleration(
        ax=0.0,
        ay=0.0,
        az=1.0,
        yaw_rate=0.0,
        count=5,   
        )
        
        print(f"Sim time is {status.sim_time}")

    end_time = time.perf_counter()

    print(f"Sim ended. Took {round(end_time-start_time,2)} seconds")
    sim.pause()