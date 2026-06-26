import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, FancyArrow
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import time

class Trajectory2Environment:
    def __init__(self):
        # --- Physical Constraints ---
        self.x_lim = (-6.0, 6.0)
        self.y_lim = (-12.5, 12.5)
        self.z_lim = (0.0, -8.0)
        
        # --- Initial Tunable Parameters (DEFAULTS) ---
        self.def_V0 = 1.0       # Target constant speed (m/s)
        self.def_A = 2.5        # Petal amplitude (m)
        self.def_zc = -0.5      # Constant altitude
        self.def_speed = 1.0
        
        # Fans placed orthogonally outside the 3x3 footprint (A=1.5 means footprint is [-1.5, 1.5])
        self.def_fan1 = [3.0, 0.0, 180.0]  
        self.def_fan2 = [0.0, 3.0, -90.0] 

        # Active Parameters
        self.V0 = self.def_V0
        self.A = self.def_A
        self.zc = self.def_zc
        
        self.fan1_state = list(self.def_fan1)
        self.fan2_state = list(self.def_fan2)
        
        # Simulation State
        self.theta = 0.0
        self.sim_time = 0.0
        self.playback_speed = self.def_speed
        self.last_wall_time = time.time()
        
        self.fan_surfaces = []
        
        self.setup_plot()
        
    def calc_kinematics_at_theta(self, theta):
        """Stateless kinematic calculation for a given theta to enforce constant speed."""
        f_theta = 1.0 + 3.0 * (math.sin(2.0 * theta)**2)
        
        # Required angular velocity for constant speed V0
        theta_dot = self.V0 / (self.A * math.sqrt(f_theta))
        
        # Angular acceleration
        sin_4theta = math.sin(4.0 * theta)
        theta_ddot = - (3.0 * (self.V0**2) * sin_4theta) / ((self.A**2) * (f_theta**2))
        
        # Polar derivatives
        r = self.A * math.cos(2.0 * theta)
        r_dot = -2.0 * self.A * math.sin(2.0 * theta) * theta_dot
        r_ddot = -4.0 * self.A * math.cos(2.0 * theta) * (theta_dot**2) - 2.0 * self.A * math.sin(2.0 * theta) * theta_ddot
        
        # Cartesian conversion
        ct, st = math.cos(theta), math.sin(theta)
        
        xd, yd, zd = r * ct, r * st, self.zc
        vxd = r_dot * ct - r * st * theta_dot
        vyd = r_dot * st + r * ct * theta_dot
        
        axd = (r_ddot - r * (theta_dot**2)) * ct - (r * theta_ddot + 2.0 * r_dot * theta_dot) * st
        ayd = (r_ddot - r * (theta_dot**2)) * st + (r * theta_ddot + 2.0 * r_dot * theta_dot) * ct
        
        return np.array([xd, yd, zd]), np.array([vxd, vyd, 0.0]), np.array([axd, ayd, 0.0])

    def get_desired_state_traj2(self, dt: float):
        """Integrates the required angular velocity to step the state forward."""
        steps = max(1, int(dt / 0.01))
        sub_dt = dt / steps
        
        for _ in range(steps):
            f_theta = 1.0 + 3.0 * (math.sin(2.0 * self.theta)**2)
            theta_dot = self.V0 / (self.A * math.sqrt(f_theta))
            self.theta += theta_dot * sub_dt
            
        return self.calc_kinematics_at_theta(self.theta)

    def precompute_trace(self):
        # The Rose curve is periodic over 2*pi in theta
        theta_eval = np.linspace(0, 2 * math.pi, 500)
        xs, ys, zs, speeds, accels = [], [], [], [], []
        
        for th in theta_eval:
            p, v, a = self.calc_kinematics_at_theta(th)
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
            speeds.append(np.linalg.norm(v))
            accels.append(np.linalg.norm(a))
            
        return xs, ys, zs, max(speeds), max(accels)

    def setup_plot(self):
        self.fig = plt.figure(figsize=(16, 9))
        
        # --- 3D Isometric View ---
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax3d.set_title("Isometric View (NED)")
        
        dx = self.x_lim[1] - self.x_lim[0]
        dy = self.y_lim[1] - self.y_lim[0]
        dz = abs(self.z_lim[1] - self.z_lim[0])
        self.ax3d.set_box_aspect([dx, dy, dz])
        
        self.ax3d.set_xlim(self.x_lim[1], self.x_lim[0])
        self.ax3d.set_ylim(self.y_lim[0], self.y_lim[1])
        self.ax3d.set_zlim(self.z_lim[0], self.z_lim[1]) 
        self.ax3d.view_init(elev=20, azim=-45)
        
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Z (m)')
        
        # Environment rendering
        xx, yy = np.meshgrid([self.x_lim[0], self.x_lim[1]], [self.y_lim[0], self.y_lim[1]])
        zz_floor = np.zeros_like(xx)
        self.ax3d.plot_surface(xx, yy, zz_floor, color='black', alpha=0.8)
        
        z_min, z_max = self.z_lim[1], self.z_lim[0]
        w_y1 = [[self.x_lim[0], self.y_lim[0], z_min], [self.x_lim[1], self.y_lim[0], z_min], [self.x_lim[1], self.y_lim[0], z_max], [self.x_lim[0], self.y_lim[0], z_max]]
        w_y2 = [[self.x_lim[0], self.y_lim[1], z_min], [self.x_lim[1], self.y_lim[1], z_min], [self.x_lim[1], self.y_lim[1], z_max], [self.x_lim[0], self.y_lim[1], z_max]]
        w_x1 = [[self.x_lim[0], self.y_lim[0], z_min], [self.x_lim[0], self.y_lim[1], z_min], [self.x_lim[0], self.y_lim[1], z_max], [self.x_lim[0], self.y_lim[0], z_max]]
        w_x2 = [[self.x_lim[1], self.y_lim[0], z_min], [self.x_lim[1], self.y_lim[1], z_min], [self.x_lim[1], self.y_lim[1], z_max], [self.x_lim[1], self.y_lim[0], z_max]]
        self.ax3d.add_collection3d(Poly3DCollection([w_y1, w_y2, w_x1, w_x2], facecolors='gray', alpha=0.1))

        # --- 2D Top-Down View ---
        self.ax2d = self.fig.add_subplot(122)
        self.ax2d.set_title("Top-Down View (+X Left, +Y Up)")
        self.ax2d.set_aspect('equal', adjustable='box')
        self.ax2d.set_xlim(self.x_lim[1], self.x_lim[0]) 
        self.ax2d.set_ylim(self.y_lim[0], self.y_lim[1]) 
        self.ax2d.set_xlabel('X (m)')
        self.ax2d.set_ylabel('Y (m)')
        self.ax2d.grid(True)
        self.ax2d.add_patch(Rectangle((self.x_lim[0], self.y_lim[0]), dx, dy, fill=False, edgecolor='black', linewidth=3))

        # Animations
        self.static_path3d, = self.ax3d.plot([], [], [], color='lightblue', alpha=0.6, linewidth=2)
        self.static_path2d, = self.ax2d.plot([], [], color='lightblue', alpha=0.6, linewidth=2)
        self.point3d, = self.ax3d.plot([], [], [], 'bo', markersize=8)
        self.path3d, = self.ax3d.plot([], [], [], 'b--', alpha=0.5)
        self.point2d, = self.ax2d.plot([], [], 'bo', markersize=8)
        self.path2d, = self.ax2d.plot([], [], 'b--', alpha=0.5)
        
        self.fan1_patch = Rectangle((0, 0), 1.5, 1.5, color='green', alpha=0.3)
        self.fan1_arrow = FancyArrow(0, 0, 1, 0, width=0.1, color='green')
        self.fan2_patch = Rectangle((0, 0), 1.5, 1.5, color='orange', alpha=0.3)
        self.fan2_arrow = FancyArrow(0, 0, 1, 0, width=0.1, color='orange')
        self.ax2d.add_patch(self.fan1_patch); self.ax2d.add_patch(self.fan1_arrow)
        self.ax2d.add_patch(self.fan2_patch); self.ax2d.add_patch(self.fan2_arrow)
        
        # UI Texts
        self.stats_text = self.fig.text(0.5, 0.96, '', ha='center', fontsize=12, fontweight='bold', color='red')
        self.timer_text = self.fig.text(0.5, 0.93, 'Time: 0.00 s', ha='center', fontsize=14, fontweight='bold')
        self.live_stats_text = self.fig.text(0.5, 0.90, 'Current Speed: 0.00 m/s | Current Accel: 0.00 m/s^2', ha='center', fontsize=12, color='blue')
        
        # --- UI Layout ---
        plt.subplots_adjust(bottom=0.35)
        
        # Updated sliders for Trajectory 2
        self.s_V0 = Slider(plt.axes([0.05, 0.25, 0.25, 0.02]), 'Target Spd (V0)', 0.1, 3.0, valinit=self.V0)
        self.s_A = Slider(plt.axes([0.05, 0.20, 0.25, 0.02]), 'Radius (A)', 0.5, 5.0, valinit=self.A)
        self.s_zc = Slider(plt.axes([0.05, 0.15, 0.25, 0.02]), 'Alt (zc)', -4.0, 0.0, valinit=self.zc)
        
        for s in [self.s_V0, self.s_A, self.s_zc]: s.on_changed(self.trigger_recalc)
        
        self.tb_speed = TextBox(plt.axes([0.05, 0.05, 0.1, 0.03]), 'Speed x ', initial=str(self.def_speed))
        self.tb_speed.on_submit(self.update_speed)
        
        fan1_ax_x = plt.axes([0.45, 0.20, 0.05, 0.03])
        fan1_ax_y = plt.axes([0.55, 0.20, 0.05, 0.03])
        fan1_ax_th = plt.axes([0.65, 0.20, 0.05, 0.03])
        plt.text(-2.5, 0.5, "Fan 1 (Grn):", transform=fan1_ax_x.transAxes, fontweight='bold')
        
        fan2_ax_x = plt.axes([0.45, 0.10, 0.05, 0.03])
        fan2_ax_y = plt.axes([0.55, 0.10, 0.05, 0.03])
        fan2_ax_th = plt.axes([0.65, 0.10, 0.05, 0.03])
        plt.text(-2.5, 0.5, "Fan 2 (Orn):", transform=fan2_ax_x.transAxes, fontweight='bold')
        
        self.tb_f1_x = TextBox(fan1_ax_x, 'X: ', initial=str(self.fan1_state[0]))
        self.tb_f1_y = TextBox(fan1_ax_y, 'Y: ', initial=str(self.fan1_state[1]))
        self.tb_f1_th = TextBox(fan1_ax_th, 'Deg: ', initial=str(self.fan1_state[2]))
        
        self.tb_f2_x = TextBox(fan2_ax_x, 'X: ', initial=str(self.fan2_state[0]))
        self.tb_f2_y = TextBox(fan2_ax_y, 'Y: ', initial=str(self.fan2_state[1]))
        self.tb_f2_th = TextBox(fan2_ax_th, 'Deg: ', initial=str(self.fan2_state[2]))
        
        for tb in [self.tb_f1_x, self.tb_f1_y, self.tb_f1_th, self.tb_f2_x, self.tb_f2_y, self.tb_f2_th]:
            tb.on_submit(self.update_fans)

        self.btn_reset_all = Button(plt.axes([0.8, 0.20, 0.15, 0.04]), 'Reset All Defaults')
        self.btn_reset_traj = Button(plt.axes([0.8, 0.10, 0.15, 0.04]), 'Reset Trajectory')
        self.btn_reset_all.on_clicked(self.cb_reset_all)
        self.btn_reset_traj.on_clicked(self.cb_reset_traj)

        self.history_x, self.history_y, self.history_z = [], [], []
        
        self.trigger_recalc(None)
        self.update_fans(None)
        self.last_wall_time = time.time()
        self.anim = FuncAnimation(self.fig, self.update_frame, interval=20, blit=False, cache_frame_data=False)
        plt.show()

    def cb_reset_all(self, event):
        self.s_V0.set_val(self.def_V0)
        self.s_A.set_val(self.def_A)
        self.s_zc.set_val(self.def_zc)
        self.tb_speed.set_val(str(self.def_speed))
        
        self.tb_f1_x.set_val(str(self.def_fan1[0]))
        self.tb_f1_y.set_val(str(self.def_fan1[1]))
        self.tb_f1_th.set_val(str(self.def_fan1[2]))
        self.tb_f2_x.set_val(str(self.def_fan2[0]))
        self.tb_f2_y.set_val(str(self.def_fan2[1]))
        self.tb_f2_th.set_val(str(self.def_fan2[2]))
        
        self.update_fans(None)
        self.cb_reset_traj(None)

    def cb_reset_traj(self, event):
        self.sim_time = 0.0
        self.theta = 0.0
        self.history_x.clear()
        self.history_y.clear()
        self.history_z.clear()

    def trigger_recalc(self, val):
        self.V0 = self.s_V0.val
        self.A = self.s_A.val
        self.zc = self.s_zc.val
        
        self.cb_reset_traj(None)
        
        xs, ys, zs, max_v, max_a = self.precompute_trace()
        self.static_path3d.set_data(xs, ys)
        self.static_path3d.set_3d_properties(zs)
        self.static_path2d.set_data(xs, ys)
        self.stats_text.set_text(f"ANALYTICS | Max Speed: {max_v:.2f} m/s | Max Accel: {max_a:.2f} m/s^2")

    def update_speed(self, val):
        try: self.playback_speed = float(val)
        except ValueError: pass

    def build_3d_cylinder(self, state, color):
        x, y, th = state
        R = 0.75  
        L = 0.5   
        rad = math.radians(th)
        z_c = -R
        
        dir_ext = np.array([-math.cos(rad), -math.sin(rad), 0])
        n_plane = np.array([-math.sin(rad), math.cos(rad), 0])
        z_up = np.array([0, 0, 1])
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, L, 2)
        U, V = np.meshgrid(u, v)
        
        X = x + V * dir_ext[0] + R * np.cos(U) * n_plane[0] + R * np.sin(U) * z_up[0]
        Y = y + V * dir_ext[1] + R * np.cos(U) * n_plane[1] + R * np.sin(U) * z_up[1]
        Z = z_c + V * dir_ext[2] + R * np.cos(U) * n_plane[2] + R * np.sin(U) * z_up[2]
        
        return self.ax3d.plot_surface(X, Y, Z, color=color, alpha=0.8)

    def update_fans(self, val):
        try:
            self.fan1_state = [float(self.tb_f1_x.text), float(self.tb_f1_y.text), float(self.tb_f1_th.text)]
            self.fan2_state = [float(self.tb_f2_x.text), float(self.tb_f2_y.text), float(self.tb_f2_th.text)]
        except ValueError:
            pass 
        
        def render_fan_2d(patch, arrow, state):
            x, y, th = state
            rad = math.radians(th)
            dx, dy = 1.5 * math.cos(rad), 1.5 * math.sin(rad)
            arrow.set_data(x=x, y=y, dx=dx, dy=dy)
            
            patch.set_xy((0, -0.75))
            patch.set_width(1.5); patch.set_height(1.5)
            t = mtransforms.Affine2D().rotate(rad).translate(x, y) + self.ax2d.transData
            patch.set_transform(t)

        render_fan_2d(self.fan1_patch, self.fan1_arrow, self.fan1_state)
        render_fan_2d(self.fan2_patch, self.fan2_arrow, self.fan2_state)

        for surf in self.fan_surfaces:
            surf.remove()
        self.fan_surfaces.clear()
        
        surf1 = self.build_3d_cylinder(self.fan1_state, 'green')
        surf2 = self.build_3d_cylinder(self.fan2_state, 'orange')
        self.fan_surfaces.extend([surf1, surf2])

    def update_frame(self, frame):
        current_time = time.time()
        dt_wall = current_time - self.last_wall_time
        self.last_wall_time = current_time
        
        dt_sim = dt_wall * self.playback_speed
        self.sim_time += dt_sim
        
        self.timer_text.set_text(f"Elapsed Time: {self.sim_time:.2f} s")
        pos, vel, acc = self.get_desired_state_traj2(dt_sim)
        
        cur_speed = np.linalg.norm(vel)
        cur_accel = np.linalg.norm(acc)
        self.live_stats_text.set_text(f"Current Speed: {cur_speed:.2f} m/s | Current Accel: {cur_accel:.2f} m/s^2")
        
        self.history_x.append(pos[0])
        self.history_y.append(pos[1])
        self.history_z.append(pos[2])
        
        # Calculate dynamic trail length based on orbital time to keep it from taking over the screen
        est_period = (2.0 * math.pi * self.A) / self.V0 if self.V0 > 0 else 10
        if len(self.history_x) > max(10, int(est_period * 50)): 
            self.history_x.pop(0); self.history_y.pop(0); self.history_z.pop(0)
        
        self.point3d.set_data([pos[0]], [pos[1]])
        self.point3d.set_3d_properties([pos[2]])
        self.path3d.set_data(self.history_x, self.history_y)
        self.path3d.set_3d_properties(self.history_z)
        
        self.point2d.set_data([pos[0]], [pos[1]])
        self.path2d.set_data(self.history_x, self.history_y)

if __name__ == "__main__":
    env = Trajectory2Environment()