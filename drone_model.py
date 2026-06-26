from __future__ import annotations

import numpy as np

try:
    from .config import DroneConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DroneConfig


class InvertedDrone2D:
    """2D moving-base inverted pendulum drone model.

    State:
        [x, z, theta, vx, vz, omega]

    Action:
        [throttle, ax_cmd]

    Here x/z are the thrust point position. The center of gravity is offset by
    l along the body axis:
        cg_x = x + l * sin(theta)
        cg_z = z + l * cos(theta)
    """

    def __init__(self, config: DroneConfig):
        self.cfg = config
        self.state = np.zeros(6, dtype=float)
        self.last_thrust = 0.0
        self.last_ax_cmd = 0.0
        self.last_theta_ddot = 0.0

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = np.array(
                [0.0, self.cfg.target_z, np.deg2rad(10.0), 0.0, 0.0, 0.0],
                dtype=float,
            )

        self.state = np.asarray(state, dtype=float).copy()
        if self.state.shape != (6,):
            raise ValueError("state must have shape (6,)")

        self.last_thrust = 0.0
        self.last_ax_cmd = 0.0
        self.last_theta_ddot = 0.0
        return self.state.copy()

    def clamp_action(self, action: np.ndarray) -> np.ndarray:
        throttle, ax_cmd = np.asarray(action, dtype=float)
        throttle = np.clip(throttle, 0.0, 1.0)
        ax_cmd = np.clip(ax_cmd, -self.cfg.ax_cmd_max, self.cfg.ax_cmd_max)
        return np.array([throttle, ax_cmd], dtype=float)

    def cg_position(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = self.state
        x, z, theta, *_ = state
        return np.array(
            [
                x + self.cfg.l * np.sin(theta),
                z + self.cfg.l * np.cos(theta),
            ],
            dtype=float,
        )

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        _x, _z, theta, vx, vz, omega = state
        throttle, ax_cmd = self.clamp_action(action)

        thrust = throttle * self.cfg.T_max
        x_ddot = ax_cmd
        z_ddot = thrust / self.cfg.m - self.cfg.g
        theta_ddot = (
            (self.cfg.g * np.sin(theta) - x_ddot * np.cos(theta)) / self.cfg.l
            - self.cfg.damping * omega
        )

        return np.array([vx, vz, omega, x_ddot, z_ddot, theta_ddot], dtype=float)

    def accelerations(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.dynamics(state, action)[3:]

    def step(self, action: np.ndarray) -> np.ndarray:
        throttle, ax_cmd = self.clamp_action(action)
        x, z, theta, vx, vz, omega = self.state

        thrust = throttle * self.cfg.T_max
        x_ddot = ax_cmd
        z_ddot = thrust / self.cfg.m - self.cfg.g
        theta_ddot = (
            (self.cfg.g * np.sin(theta) - x_ddot * np.cos(theta)) / self.cfg.l
            - self.cfg.damping * omega
        )

        dt = self.cfg.dt
        vx += x_ddot * dt
        vz += z_ddot * dt
        omega += theta_ddot * dt

        x += vx * dt
        z += vz * dt
        theta += omega * dt

        self.state = np.array([x, z, theta, vx, vz, omega], dtype=float)
        self.last_thrust = thrust
        self.last_ax_cmd = ax_cmd
        self.last_theta_ddot = theta_ddot
        return self.state.copy()
