from quadsim import QuadSim
from quadsim.types import SensorData

with QuadSim() as sim:
    drone = sim.drone()
    sim.pause()
    drone.reset_pose(x=0.0,y=0.1,z=0.0)

    drone._transport.request("set_frame", {"frame": "flu"})
    yaw_cmd = 0.0

    for _ in range(1, 50):
        response = drone._transport.request("step_with_command", {
            "x": 3.0,       # FLU forward
            "y": 0.0,       # FLU left
            "z": 5.0,       # FLU up
            "w": 45, # deg/s       # yaw
            "mode": "position",
            "count": 5,     # 5 × 250 Hz steps = 50 Hz control
        })

        sensors = SensorData.from_dict(response)

        print(
            "Position NED:", sensors.gps_position, # _ned
            "| Velocity NED:", sensors.gps_vel_ned,
            "Orientation", sensors.imu_orientation #_flu
        )

        # x NED = +x in Unity
        # y NED = +z in Unity
        # z NED = +y in Unity (up)



    sim.pause()
