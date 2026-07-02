import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, FancyArrow
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import time

class TrajectoryEnvironment:
    def __init__(self):
        # --- Physical Constraints ---
        self.x_lim = (-6.0, 6.0)
        self.y_lim = (-12.5, 12.5)
        self.z_lim = (0.0, -8.0)
        
        # --- Initial Tunable Parameters (DEFAULTS) ---
        self.def_T_period = 30.0
        self.def_alpha_warp = 0.5
        self.def_zc = -0.75
        self.def_Az = 0.25
        self.def_speed = 1.0
        self.def_fan1 = [-2, 3, 0.0]
        self.def_fan2 = [-2, -3, 0.0]

        # Active Parameters
        self.T_period = self.def_T_period
        self.alpha_warp = self.def_alpha_warp
        self.warp_c = 1.0 
        self.zc = self.def_zc
        self.Az = self.def_Az
        self.Ax = 1.5
        self.Ay = 3.0
        
        # Fan States
        self.fan1_state = list(self.def_fan1)
        self.fan2_state = list(self.def_fan2)
        
        # Simulation State
        self.tau = 0.0
        self.sim_time = 0.0
        self.playback_speed = self.def_speed
        self.last_wall_time = time.time()
        
        self.fan_surfaces = [] # Store 3D cylinder collections to remove/update
        
        self.setup_plot()
        
    def get_desired_state_traj1(self, dt: float):
        w = (2.0 * math.pi) / self.T_period
        
        steps = max(1, int(dt / 0.01))
        sub_dt = dt / steps
        tau_ddot = 0.0
        
        for _ in range(steps):
            tau_dot = self.warp_c * (1.0 - self.alpha_warp * (math.sin(w * self.tau)**2))
            tau_ddot = -self.warp_c * self.alpha_warp * w * math.sin(2.0 * w * self.tau) * tau_dot
            self.tau += tau_dot * sub_dt
            
        wx, wy, wz = 2.0 * w, 1.0 * w, 4.0 * w
        
        xd = self.Ax * math.sin(wx * self.tau)
        yd = self.Ay * math.sin(wy * self.tau)
        zd = self.Az * math.sin(wz * self.tau) + self.zc
        
        vxd = (self.Ax * wx * math.cos(wx * self.tau)) * tau_dot
        vyd = (self.Ay * wy * math.cos(wy * self.tau)) * tau_dot
        vzd = (self.Az * wz * math.cos(wz * self.tau)) * tau_dot
        
        axd = -(self.Ax * wx**2 * math.sin(wx * self.tau)) * (tau_dot**2) + (self.Ax * wx * math.cos(wx * self.tau)) * tau_ddot
        ayd = -(self.Ay * wy**2 * math.sin(wy * self.tau)) * (tau_dot**2) + (self.Ay * wy * math.cos(wy * self.tau)) * tau_ddot
        azd = -(self.Az * wz**2 * math.sin(wz * self.tau)) * (tau_dot**2) + (self.Az * wz * math.cos(wz * self.tau)) * tau_ddot
        
        return (np.array([xd, yd, zd]), np.array([vxd, vyd, vzd]), np.array([axd, ayd, azd]))

    def precompute_trace(self):
        saved_tau = self.tau
        self.tau = 0.0
        
        t_eval = np.linspace(0, self.T_period, 500)
        xs, ys, zs, speeds, accels = [], [], [], [], []
        
        for i in range(len(t_eval)):
            dt = t_eval[i] if i == 0 else t_eval[i] - t_eval[i-1]
            p, v, a = self.get_desired_state_traj1(dt)
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
            speeds.append(np.linalg.norm(v))
            accels.append(np.linalg.norm(a))
            
        self.tau = saved_tau 
        return xs, ys, zs, min(speeds), max(speeds), max(accels)

    def setup_plot(self):
        self.fig = plt.figure(figsize=(16, 9))
        
        # --- 3D Isometric View ---
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax3d.set_title("Isometric View (NED)")
        
        dx = self.x_lim[1] - self.x_lim[0]
        dy = self.y_lim[1] - self.y_lim[0]
        dz = abs(self.z_lim[1] - self.z_lim[0])
        self.ax3d.set_box_aspect([dx, dy, dz])
        
        # X-Axis Flipped: max to min
        self.ax3d.set_xlim(self.x_lim[1], self.x_lim[0])
        self.ax3d.set_ylim(self.y_lim[0], self.y_lim[1])
        self.ax3d.set_zlim(self.z_lim[0], self.z_lim[1]) 
        self.ax3d.view_init(elev=20, azim=-45)
        
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Z (m)')
        
        # 3D Environment rendering
        xx, yy = np.meshgrid([self.x_lim[0], self.x_lim[1]], [self.y_lim[0], self.y_lim[1]])
        zz_floor = np.zeros_like(xx)
        self.ax3d.plot_surface(xx, yy, zz_floor, color='gray', alpha=0.8)
        
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
        self.ax2d.add_patch(Rectangle((self.x_lim[0], self.y_lim[0]), dx, dy, fill=False, edgecolor='gray', linewidth=3))

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
        
        self.s_T = Slider(plt.axes([0.05, 0.25, 0.25, 0.02]), 'T_period', 10.0, 40.0, valinit=self.T_period)
        self.s_alpha = Slider(plt.axes([0.05, 0.20, 0.25, 0.02]), 'alpha_warp', 0.0, 0.99, valinit=self.alpha_warp)
        self.s_zc = Slider(plt.axes([0.05, 0.15, 0.25, 0.02]), 'Alt (zc)', -4.0, 0.0, valinit=self.zc)
        self.s_Az = Slider(plt.axes([0.05, 0.10, 0.25, 0.02]), 'Z-Amp', 0.0, 2.0, valinit=self.Az)
        
        for s in [self.s_T, self.s_alpha, self.s_zc, self.s_Az]: s.on_changed(self.trigger_recalc)
        
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
        self.s_T.set_val(self.def_T_period)
        self.s_alpha.set_val(self.def_alpha_warp)
        self.s_zc.set_val(self.def_zc)
        self.s_Az.set_val(self.def_Az)
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
        self.tau = 0.0
        self.history_x.clear()
        self.history_y.clear()
        self.history_z.clear()

    def trigger_recalc(self, val):
        self.T_period = self.s_T.val
        self.alpha_warp = self.s_alpha.val
        self.zc = self.s_zc.val
        self.Az = self.s_Az.val
        self.warp_c = 1.0 / math.sqrt(1.0 - self.alpha_warp) if self.alpha_warp < 1.0 else 1.0
        
        self.cb_reset_traj(None)
        
        # Unpack the new min_v value
        xs, ys, zs, min_v, max_v, max_a = self.precompute_trace()
        self.static_path3d.set_data(xs, ys)
        self.static_path3d.set_3d_properties(zs)
        self.static_path2d.set_data(xs, ys)
        
        # Update display string
        self.stats_text.set_text(
            f"ANALYTICS | Speed Range: [{min_v:.2f}, {max_v:.2f}] m/s | Max Accel: {max_a:.2f} m/s^2"
        )

    def update_speed(self, val):
        try: self.playback_speed = float(val)
        except ValueError: pass

    def build_3d_cylinder(self, state, color):
        """Generates parametric mesh for a solid 3D cylinder tangent to Z=0."""
        x, y, th = state
        R = 0.75  # 1.5m diameter
        L = 0.5   # 0.5m thick extrusion
        rad = math.radians(th)
        
        # Center of the base (tangent to z=0 means z center is -R)
        z_c = -R
        
        # Directions
        dir_ext = np.array([-math.cos(rad), -math.sin(rad), 0]) # Extrude opposite to arrow
        n_plane = np.array([-math.sin(rad), math.cos(rad), 0])
        z_up = np.array([0, 0, 1])
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, L, 2)
        U, V = np.meshgrid(u, v)
        
        # Parametric Surface
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

        # Update 3D Cylinders
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
        pos, vel, acc = self.get_desired_state_traj1(dt_sim)
        
        # Live Stats Update
        cur_speed = np.linalg.norm(vel)
        cur_accel = np.linalg.norm(acc)
        self.live_stats_text.set_text(f"Current Speed: {cur_speed:.2f} m/s | Current Accel: {cur_accel:.2f} m/s^2")
        
        self.history_x.append(pos[0])
        self.history_y.append(pos[1])
        self.history_z.append(pos[2])
        
        if len(self.history_x) > max(10, int(self.T_period * 50)): 
            self.history_x.pop(0); self.history_y.pop(0); self.history_z.pop(0)
        
        self.point3d.set_data([pos[0]], [pos[1]])
        self.point3d.set_3d_properties([pos[2]])
        self.path3d.set_data(self.history_x, self.history_y)
        self.path3d.set_3d_properties(self.history_z)
        
        self.point2d.set_data([pos[0]], [pos[1]])
        self.path2d.set_data(self.history_x, self.history_y)

if __name__ == "__main__":
    env = TrajectoryEnvironment()