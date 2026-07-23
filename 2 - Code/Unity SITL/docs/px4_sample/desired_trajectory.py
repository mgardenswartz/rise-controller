"""NED trajectory generation for the QuadSim/PX4 RISE experiments.

The public interface intentionally stays small: :class:`TrajectoryGenerator`
returns position, velocity, and kinematic acceleration in world NED.  The
implementation is NumPy/SciPy-only so the PID bring-up and future Optuna
workers do not need JAX just to sample a reference trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
from scipy.integrate import solve_ivp


class TrajectoryGenerator:
    """Generate the original warped figure-eight or rose trajectory in NED."""

    FIGURE_EIGHT = 1
    ROSE = 2

    def __init__(self, config: dict[str, Any]) -> None:
        self.desired_traj = int(config["desired_trajectory"])
        self.run_length_s = float(config["run_length_s"]) + 1.0
        if self.run_length_s <= 1.0:
            raise ValueError("run_length_s must be positive")

        if self.desired_traj == self.FIGURE_EIGHT:
            self.traj1_center_z_m_ned_aviary = float(
                config["traj1_center_z_m_ned_aviary"]
            )
            self.traj1_period_s = float(config["traj1_period_s"])
            self.traj1_x_amp_m_ned_aviary = float(
                config["traj1_x_amp_m_ned_aviary"]
            )
            self.traj1_y_amp_m_ned_aviary = float(
                config["traj1_y_amp_m_ned_aviary"]
            )
            self.traj1_z_amp_m_ned_aviary = float(
                config["traj1_z_amp_m_ned_aviary"]
            )
            self.traj1_alpha_warp = float(config["traj1_alpha_warp"])
            if self.traj1_period_s <= 0.0:
                raise ValueError("traj1_period_s must be positive")
            if not 0.0 <= self.traj1_alpha_warp < 1.0:
                raise ValueError("traj1_alpha_warp must be in [0, 1)")
            self.traj1_warp_c = 1.0 / math.sqrt(
                1.0 - self.traj1_alpha_warp
            )
        elif self.desired_traj == self.ROSE:
            self.traj2_center_z_m_ned_aviary = float(
                config["traj2_center_z_m_ned_aviary"]
            )
            self.traj2_petal_radius_m = float(config["traj2_petal_radius_m"])
            self.traj2_target_speed_mps = float(
                config["traj2_target_speed_mps"]
            )
            if self.traj2_petal_radius_m <= 0.0:
                raise ValueError("traj2_petal_radius_m must be positive")
            if self.traj2_target_speed_mps <= 0.0:
                raise ValueError("traj2_target_speed_mps must be positive")
        else:
            raise ValueError("desired_trajectory must be 1 (figure-eight) or 2 (rose)")

        self._precompute_phase()

    def _precompute_phase(self) -> None:
        """Integrate the scalar time-warp state once at construction."""
        if self.desired_traj == self.FIGURE_EIGHT:
            w = 2.0 * math.pi / self.traj1_period_s

            def dtau_dt(_t: float, tau: np.ndarray) -> np.ndarray:
                rate = self.traj1_warp_c * (
                    1.0
                    - self.traj1_alpha_warp * math.sin(w * tau[0]) ** 2
                )
                return np.array([rate])

            initial_phase = self.traj1_period_s / 4.0
            solution = solve_ivp(
                dtau_dt,
                (0.0, self.run_length_s),
                (initial_phase,),
                max_step=0.01,
            )
            self._time_grid = solution.t
            self._phase_grid = solution.y[0]
        else:

            def dtheta_dt(_t: float, theta: np.ndarray) -> np.ndarray:
                metric = 1.0 + 3.0 * math.sin(2.0 * theta[0]) ** 2
                rate = self.traj2_target_speed_mps / (
                    self.traj2_petal_radius_m * math.sqrt(metric)
                )
                return np.array([rate])

            solution = solve_ivp(
                dtheta_dt,
                (0.0, self.run_length_s),
                (0.0,),
                max_step=0.01,
            )
            self._time_grid = solution.t
            self._phase_grid = solution.y[0]

        if not solution.success:
            raise RuntimeError(
                f"trajectory phase integration failed: {solution.message}"
            )

    def _get_figure_eight(
        self, time_s: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = float(np.interp(time_s, self._time_grid, self._phase_grid))
        w = 2.0 * math.pi / self.traj1_period_s
        frequencies = np.array([2.0 * w, w, 4.0 * w])
        amplitudes = np.array(
            [
                self.traj1_x_amp_m_ned_aviary,
                self.traj1_y_amp_m_ned_aviary,
                self.traj1_z_amp_m_ned_aviary,
            ]
        )

        tau_dot = self.traj1_warp_c * (
            1.0 - self.traj1_alpha_warp * math.sin(w * tau) ** 2
        )
        tau_ddot = (
            -2.0
            * self.traj1_warp_c
            * self.traj1_alpha_warp
            * w
            * math.sin(w * tau)
            * math.cos(w * tau)
            * tau_dot
        )

        phase = frequencies * tau
        position = amplitudes * np.sin(phase)
        position[2] += self.traj1_center_z_m_ned_aviary
        dp_dtau = amplitudes * frequencies * np.cos(phase)
        d2p_dtau2 = -amplitudes * frequencies**2 * np.sin(phase)
        velocity = dp_dtau * tau_dot
        acceleration = d2p_dtau2 * tau_dot**2 + dp_dtau * tau_ddot
        return position, velocity, acceleration

    def _get_rose(
        self, time_s: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = float(np.interp(time_s, self._time_grid, self._phase_grid))
        radius = self.traj2_petal_radius_m
        metric = 1.0 + 3.0 * math.sin(2.0 * theta) ** 2
        theta_dot = self.traj2_target_speed_mps / (radius * math.sqrt(metric))
        theta_ddot = -(
            3.0
            * self.traj2_target_speed_mps**2
            * math.sin(4.0 * theta)
            / (radius**2 * metric**2)
        )

        # Harmonic form of r=A*cos(2*theta), x=r*cos(theta), y=r*sin(theta).
        position = np.array(
            [
                0.5 * radius * (math.cos(3.0 * theta) + math.cos(theta)),
                0.5 * radius * (math.sin(3.0 * theta) - math.sin(theta)),
                self.traj2_center_z_m_ned_aviary,
            ]
        )
        dp_dtheta = np.array(
            [
                0.5 * radius * (-3.0 * math.sin(3.0 * theta) - math.sin(theta)),
                0.5 * radius * (3.0 * math.cos(3.0 * theta) - math.cos(theta)),
                0.0,
            ]
        )
        d2p_dtheta2 = np.array(
            [
                0.5 * radius * (-9.0 * math.cos(3.0 * theta) - math.cos(theta)),
                0.5 * radius * (-9.0 * math.sin(3.0 * theta) + math.sin(theta)),
                0.0,
            ]
        )
        velocity = dp_dtheta * theta_dot
        acceleration = d2p_dtheta2 * theta_dot**2 + dp_dtheta * theta_ddot
        return position, velocity, acceleration

    def get_desired_state(
        self, time_s: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(position, velocity, acceleration)`` in world NED."""
        if not math.isfinite(time_s):
            raise ValueError("trajectory time must be finite")
        clamped_time = min(max(0.0, float(time_s)), self.run_length_s)
        if self.desired_traj == self.FIGURE_EIGHT:
            return self._get_figure_eight(clamped_time)
        return self._get_rose(clamped_time)
