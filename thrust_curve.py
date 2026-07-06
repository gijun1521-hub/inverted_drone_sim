from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from .config import RigidBodyConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import RigidBodyConfig


class ThrottleToThrustModel:
    def __init__(self, cfg: RigidBodyConfig):
        self.cfg = cfg
        self.lookup: tuple[np.ndarray, np.ndarray] | None = None
        if cfg.thrust_curve_model == "lookup_csv" and cfg.thrust_curve_lookup_csv:
            self.lookup = self._load_lookup(cfg.thrust_curve_lookup_csv)

    def _load_lookup(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        throttles: list[float] = []
        thrusts: list[float] = []
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                throttles.append(float(row["throttle"]))
                thrusts.append(float(row["thrust"]))
        if len(throttles) < 2:
            raise ValueError("lookup CSV needs at least two rows")
        xs = np.asarray(throttles, dtype=float)
        ys = np.asarray(thrusts, dtype=float)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            raise ValueError("lookup CSV contains non-finite values")
        if np.any(np.diff(xs) <= 0.0):
            raise ValueError("lookup CSV throttle values must be strictly increasing")
        if xs[0] < 0.0 or xs[-1] > 1.0:
            raise ValueError("lookup CSV throttle values must stay within [0, 1]")
        return xs, ys

    def thrust(self, throttle: float) -> float:
        u = float(np.clip(throttle, 0.0, 1.0))
        model = self.cfg.thrust_curve_model
        if model == "linear":
            value = self.cfg.T_max * u
        elif model == "quadratic":
            value = self.cfg.T_max * u * u
        elif model == "polynomial":
            if not self.cfg.thrust_curve_coefficients:
                raise ValueError("polynomial thrust curve requires coefficients, e.g. (T_max, 0.0)")
            coeffs = self.cfg.thrust_curve_coefficients
            value = float(np.polyval(coeffs, u))
        elif model == "lookup_csv":
            if self.lookup is None:
                raise ValueError("lookup_csv selected but no lookup table is loaded")
            xs, ys = self.lookup
            value = float(np.interp(u, xs, ys))
        else:
            raise ValueError(f"unknown thrust curve model: {model}")
        return float(np.clip(value, 0.0, self.cfg.T_max))

    def throttle_for_hover(self) -> float:
        if self.cfg.thrust_curve_model == "linear":
            return self.cfg.hover_thrust / self.cfg.T_max
        samples = np.linspace(0.0, 1.0, 1001)
        thrusts = np.array([self.thrust(v) for v in samples])
        idx = int(np.argmin(np.abs(thrusts - self.cfg.hover_thrust)))
        return float(samples[idx])
