import os
import sys
import time
import math
import numpy as np
import subprocess
from pymavlink import mavutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_sim import SimRun
from quadsim.sim import QuadSim
from quadsim.tools.px4_bridge import HilBridge
from quadsim.px4 import Px4Link
from quadsim.tools.px4_lockstep_runner import LockstepRunner

def main():
    print("[*] Launching PX4 SITL...")
    px4_process = subprocess.Popen(
        ["make", "px4_sitl", "none_iris"],
        cwd="/Users/max/PX4-Autopilot",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True
    )
    
    for line in px4_process.stdout:
        if "Waiting for simulator" in line or "px4 starting." in line:
            print("[*] PX4 is up! Connecting Unity...")
            break

    try:
        QUADSIM_HOST = "localhost"
        QUADSIM_PORT = 5555
        PX4_HOST = "127.0.0.1"
        PX4_INSTANCE = 0
        PX4_CLIENT = False
        
        sim = QuadSim(host=QUADSIM_HOST, command_port=QUADSIM_PORT, telemetry_port=QUADSIM_PORT + 1)
        sim.connect()
        sim.set_wind(enabled=True, wind_speed=0.0)

        bridge = HilBridge(px4_host=PX4_HOST, instance=PX4_INSTANCE, px4_client=PX4_CLIENT)
        px4 = Px4Link(stream_hz=0, instance=PX4_INSTANCE)
        runner = LockstepRunner(sim, bridge, px4, control_hz=100)
        
        runner.start()
        runner.wait_px4()
        px4.configure_offboard_no_rc()
        if not runner.wait_ekf_ready(): raise SystemExit("EKF never set home")
        if not runner.wait_heading(): raise SystemExit("heading chain failed")
        
        print("[*] Setting PX4 MPC parameters for Unity mass...")
        px4.param_set("MPC_THR_HOVER", 0.8) # Unity drone is heavy
        px4.param_set("MPC_ACC_UP_MAX", 10.0)
        px4.param_set("MPC_ACC_DOWN_MAX", 10.0)
        px4.param_set("MPC_ACC_HOR_MAX", 10.0)
        
        print("[*] EKF Ready! Engaging OFFBOARD...")
        
        # MAVLink acceleration mask: ignore pos, vel, yaw_rate
        _ACCEL_MASK = 2111
        
        def send_accel(ax, ay, az):
            tgt_sys = px4._mav.target_system
            tgt_comp = px4._mav.target_component
            t_ms = int(time.time() * 1e3) & 0xFFFFFFFF
            px4._mav.mav.set_position_target_local_ned_send(
                t_ms, tgt_sys, tgt_comp, mavutil.mavlink.MAV_FRAME_LOCAL_NED, _ACCEL_MASK,
                0.0, 0.0, 0.0, # pos
                0.0, 0.0, 0.0, # vel
                ax, ay, az,    # accel (N, E, D)
                0.0, 0.0       # yaw, yaw_rate
            )
            n_hil = runner.hil_per_control
            for _ in range(n_hil):
                runner.hil_tick()

        # Prime with hover (az = 0.0 in PX4 NED)
        prime_deadline = runner.sim_time + 1.0
        while runner.sim_time < prime_deadline:
            send_accel(0.0, 0.0, 0.0)
            
        px4.request_offboard()
        px4.request_arm(arm=True)
        
        accept_deadline = runner.sim_time + 5.0
        while runner.sim_time < accept_deadline:
            send_accel(0.0, 0.0, 0.0)
            if px4.in_offboard() and px4.is_armed():
                break
                
        if not (px4.in_offboard() and px4.is_armed()):
            raise SystemExit("PX4 refused OFFBOARD/arm")
            
        print("\n=== STARTING PURE ACCELERATION TEST ===")
        print("Time(s) | Z_ENU (m) | VZ_ENU (m/s) | Cmd_AZ_NED (m/s^2)")
        print("-" * 55)
        
        # Test 1: Hover (AZ = 0.0)
        target_az = 0.0
        deadline = runner.sim_time + 3.0
        while runner.sim_time < deadline:
            send_accel(0.0, 0.0, target_az)
            if int(runner.sim_time * 100) % 25 == 0:
                pos_z = px4.state.pos_enu[2] if px4.state.pos_enu else 0.0
                vel_z = px4.state.vel_enu[2] if px4.state.vel_enu else 0.0
                print(f"{runner.sim_time:7.2f} | {pos_z:9.3f} | {vel_z:12.3f} | {target_az:16.3f}  <-- HOVER")
                
        # Test 2: Fly UP
        target_az = -3.0
        deadline = runner.sim_time + 6.0
        while runner.sim_time < deadline:
            send_accel(0.0, 0.0, target_az)
            if int(runner.sim_time * 100) % 25 == 0:
                pos_z = px4.state.pos_enu[2] if px4.state.pos_enu else 0.0
                vel_z = px4.state.vel_enu[2] if px4.state.vel_enu else 0.0
                print(f"{runner.sim_time:7.2f} | {pos_z:9.3f} | {vel_z:12.3f} | {target_az:16.3f}  <-- FLY UP (-3.0 NED)")

    finally:
        os.killpg(os.getpgid(px4_process.pid), 9)
        px4_process.wait()
        try:
            runner.close()
        except:
            pass

if __name__ == "__main__":
    main()
