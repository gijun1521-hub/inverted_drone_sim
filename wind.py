from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import RigidBodyConfig
except ImportError:  # pragma: no cover
    from config import RigidBodyConfig


@dataclass(frozen=True)
class WindDisturbance:
    force_world: np.ndarray
    moment: float


class SimpleWindModel:
    """Constant wind velocity plus an optional finite-duration gust impulse."""

    def __init__(self, cfg: RigidBodyConfig):
        self.cfg = cfg

    def disturbance_at(self, time_s: float) -> WindDisturbance:
        if 0.0 <= time_s < self.cfg.gust_duration_s:
            return WindDisturbance(
                np.asarray(self.cfg.gust_force_world, dtype=float),
                float(self.cfg.gust_moment),
            )
        return WindDisturbance(np.zeros(2, dtype=float), 0.0)
