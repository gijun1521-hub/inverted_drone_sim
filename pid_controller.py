from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import DroneConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DroneConfig


@dataclass(frozen=True)
class ControlBreakdown:
    throttle: float
    ax_cmd: float
    theta_term: float
    position_term: float
    velocity_term: float
    omega_term: float


class PIDController:
    """Altitude PID plus cart-pole-style horizontal base controller."""

    def __init__(
        self,
        cfg: DroneConfig,
        Kp_z: float = 2.0,
        Ki_z: float = 0.0,
        Kd_z: float = 1.2,
        Kx: float = 2.0,
        Kvx: float = 3.0,
        Ktheta: float = 28.0,
        Komega: float = 5.0,
        integral_limit_z: float = 2.0,
    ):
        self.cfg = cfg
        self.Kp_z = Kp_z
        self.Ki_z = Ki_z
        self.Kd_z = Kd_z
        self.Kx = Kx
        self.Kvx = Kvx
        self.Ktheta = Ktheta
        self.Komega = Komega
        self.integral_limit_z = integral_limit_z
        self.last_ax_cmd = 0.0
        self.reset()

    def reset(self) -> None:
        self.alt_integral = 0.0
        self.last_ax_cmd = 0.0

    def compute_breakdown(self, state: np.ndarray) -> ControlBreakdown:
        x, z, theta, vx, vz, omega = state

        z_error = self.cfg.target_z - z
        self.alt_integral += z_error * self.cfg.dt
        self.alt_integral = float(
            np.clip(self.alt_integral, -self.integral_limit_z, self.integral_limit_z)
        )

        throttle = (
            self.cfg.hover_throttle
            + self.Kp_z * z_error
            + self.Ki_z * self.alt_integral
            - self.Kd_z * vz
        )

        theta_error = theta - self.cfg.target_theta
        x_error = x - self.cfg.target_x

        # Attitude is the priority. If theta < 0, the CG is left of the thrust
        # point, so this produces ax_cmd < 0 and the base moves left to get
        # under the CG. The x/vx terms use the cart-pole stabilizing sign:
        # near upright this may accelerate toward the current offset briefly,
        # which creates the lean needed to return to the target instead of
        # drifting away.
        theta_term = self.Ktheta * theta_error
        omega_term = self.Komega * omega
        position_term = self.Kx * x_error
        velocity_term = self.Kvx * vx
        ax_cmd = theta_term + omega_term + position_term + velocity_term
        ax_cmd = float(np.clip(ax_cmd, -self.cfg.ax_cmd_max, self.cfg.ax_cmd_max))
        self.last_ax_cmd = ax_cmd

        return ControlBreakdown(
            throttle=float(throttle),
            ax_cmd=ax_cmd,
            theta_term=float(theta_term),
            position_term=float(position_term),
            velocity_term=float(velocity_term),
            omega_term=float(omega_term),
        )

    def compute_action(self, state: np.ndarray) -> np.ndarray:
        terms = self.compute_breakdown(state)
        return np.array([terms.throttle, terms.ax_cmd], dtype=float)
