from __future__ import annotations

import numpy as np


def moving_mass_reaction_body_delta(I_body: float, I_moving: float, delta_q: float) -> float:
    return float(-I_moving / max(I_body + I_moving, 1e-12) * delta_q)


def moving_mass_reaction_rate(I_body: float, I_moving: float, qdot: float) -> float:
    return float(-I_moving / max(I_body + I_moving, 1e-12) * qdot)


def moving_mass_reaction_accel(I_body: float, I_moving: float, qddot: float) -> float:
    return float(-I_moving / max(I_body + I_moving, 1e-12) * qddot)


def compute_total_cg_body(
    m_body: float,
    m_moving: float,
    body_cg_body: tuple[float, float] = (0.0, 0.0),
    moving_mass_body: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    body = np.asarray(body_cg_body, dtype=float)
    moving = np.asarray(moving_mass_body, dtype=float)
    return (m_body * body + m_moving * moving) / max(m_body + m_moving, 1e-12)


def compute_cg_offset_from_thrust_line(total_cg_body: np.ndarray, thrust_line_x_body: float = 0.0) -> float:
    return float(total_cg_body[0] - thrust_line_x_body)


def compute_thrust_offset_moment(thrust: float, lateral_offset: float) -> float:
    return float(thrust * lateral_offset)


def rotating_mass_position_body(hinge: tuple[float, float], offset: tuple[float, float], q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(hinge, dtype=float) + rot @ np.asarray(offset, dtype=float)
