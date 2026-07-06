from __future__ import annotations

import numpy as np


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def shortest_angle_error(target: float, current: float) -> float:
    return float(wrap_pi(target - current))
