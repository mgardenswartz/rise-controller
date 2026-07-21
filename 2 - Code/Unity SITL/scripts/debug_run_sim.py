import sys
import time
from src.run_sim import RunSim
import numpy as np

def run_debug():
    # Force print to flush
    import builtins
    original_print = builtins.print
    def _print(*args, **kwargs):
        kwargs['flush'] = True
        original_print(*args, **kwargs)
    builtins.print = _print

    sim = RunSim(config_path="conf/config.yaml", controller_type="baseline")
    
    # Patch step_with_acceleration to print out the commands
    from quadsim import QuadSim
    original_step = None
    
    def patched_run():
        with QuadSim() as sim_q:
            drone = sim_q.drone()
            original_step = drone.step_with_acceleration
            
            # We want to print z height
            def my_step(ax, ay, az, yaw_rate, **kwargs):
                s = drone.get_sensors()
                print(f"Z_enu: {s.gps_position[2]:.3f}, Z_ned: {-s.gps_position[2]:.3f}, az_cmd: {az:.3f}")
                return original_step(ax, ay, az, yaw_rate, **kwargs)
            
            # Replace the bound method on drone object... Actually, loop returns sensors
            # Let's just patch it in the script?
            # Instead of patching, let's just let original_run run and see if it fails.
            pass
            
    # Actually, if I just run it from python with some print statements in run_sim.py it's easier.
    sim.run()

if __name__ == '__main__':
    run_debug()
