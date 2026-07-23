"""Single outer-loop position PID producing acceleration in world NED."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PidOutput:
    acceleration_ned: np.ndarray
    position_error_ned: np.ndarray


class AccelerationPid:
    """Base RISE PID branch without the RISE term."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_horizontal_accel: float,
        max_vertical_accel: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_horizontal_accel = max_horizontal_accel
        self.max_vertical_accel = max_vertical_accel
        self.reset()

    def reset(self) -> None:
        self.integral_control = np.zeros(3)
        self.last_integrand = np.zeros(3)
        self.freeze_integral_xy = False
        self.freeze_integral_z = False

    def update(
        self,
        position_ned: np.ndarray,
        velocity_ned: np.ndarray,
        desired_position_ned: np.ndarray,
        desired_velocity_ned: np.ndarray,
        dt: float,
    ) -> PidOutput:
        error = desired_position_ned - position_ned
        error_derivative = desired_velocity_ned - velocity_ned

        integrand = self.ki * error
        integral_step = 0.5 * dt * (integrand + self.last_integrand)
        if not self.freeze_integral_xy:
            self.integral_control[:2] += integral_step[:2]
        if not self.freeze_integral_z:
            self.integral_control[2] += integral_step[2]
        self.last_integrand = integrand

        acceleration = (
            self.kp * error
            + self.kd * error_derivative
            + self.integral_control
        )

        self.freeze_integral_xy = False
        self.freeze_integral_z = False

        horizontal_norm = float(np.linalg.norm(acceleration[:2]))
        if horizontal_norm > self.max_horizontal_accel:
            acceleration[:2] *= self.max_horizontal_accel / horizontal_norm
            if np.dot(error[:2], acceleration[:2]) > 0.0:
                self.freeze_integral_xy = True

        if abs(acceleration[2]) > self.max_vertical_accel:
            acceleration[2] = self.max_vertical_accel * np.sign(acceleration[2])
            if np.sign(error[2]) == np.sign(acceleration[2]):
                self.freeze_integral_z = True

        return PidOutput(
            acceleration_ned=acceleration,
            position_error_ned=error,
        )
