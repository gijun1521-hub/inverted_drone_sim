from __future__ import annotations

import numpy as np


def finite_difference_sensitivity(fn, base_value: float, step_fraction: float = 0.01) -> float:
    step = max(abs(base_value) * step_fraction, 1e-9)
    return float((fn(base_value + step) - fn(base_value - step)) / (2.0 * step))


def monotonic(values) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(arr) >= -1e-12))
