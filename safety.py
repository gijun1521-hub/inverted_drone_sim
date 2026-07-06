from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import RigidBodyConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import RigidBodyConfig


@dataclass(frozen=True)
class SafetyStatus:
    crashed: bool
    reason: str
    min_body_z: float


def body_corners(state: np.ndarray, cfg: RigidBodyConfig) -> np.ndarray:
    x, z, theta, *_ = state
    cg = np.array([x, z], dtype=float)
    body_up = np.array([np.sin(theta), np.cos(theta)], dtype=float)
    body_right = np.array([np.cos(theta), -np.sin(theta)], dtype=float)
    top = cg + cfg.l * body_up
    bottom = cg - cfg.l * body_up
    half_w = 0.5 * cfg.W
    return np.array(
        [
            bottom - half_w * body_right,
            bottom + half_w * body_right,
            top + half_w * body_right,
            top - half_w * body_right,
        ],
        dtype=float,
    )


def check_safety(state: np.ndarray, cfg: RigidBodyConfig) -> SafetyStatus:
    state = np.asarray(state, dtype=float)
    if not np.all(np.isfinite(state)):
        return SafetyStatus(True, "non-finite state", float("nan"))

    corners = body_corners(state, cfg)
    min_body_z = float(np.min(corners[:, 1]))
    if min_body_z <= 0.0:
        return SafetyStatus(True, "ground contact", min_body_z)

    x, z, theta, vx, vz, omega, thrust, vane_angle = state
    if abs(x) > cfg.x_limit_abs:
        return SafetyStatus(True, "x limit exceeded", min_body_z)
    if z < cfg.z_limit_min or z > cfg.z_limit_max:
        return SafetyStatus(True, "z limit exceeded", min_body_z)
    if abs(theta) > cfg.theta_limit_abs:
        return SafetyStatus(True, "theta limit exceeded", min_body_z)
    if max(abs(vx), abs(vz)) > cfg.velocity_limit_abs:
        return SafetyStatus(True, "velocity limit exceeded", min_body_z)
    if abs(omega) > cfg.omega_limit_abs:
        return SafetyStatus(True, "omega limit exceeded", min_body_z)
    if thrust < -1e-9 or thrust > cfg.T_max + 1e-9:
        return SafetyStatus(True, "thrust state limit exceeded", min_body_z)
    if abs(vane_angle) > cfg.vane_angle_max + 1e-9:
        return SafetyStatus(True, "vane state limit exceeded", min_body_z)
    return SafetyStatus(False, "", min_body_z)
