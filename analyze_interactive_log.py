from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _as_float(rows: list[dict[str, str]], key: str, default: float = 0.0) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, default) or default))
        except ValueError:
            vals.append(default)
    return np.asarray(vals, dtype=float)


def _pct(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return 100.0 * float(np.mean(values > 0.5))


def analyze(path: str | Path) -> dict[str, float | str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV contains no rows")
    x_err = _as_float(rows, "x_error")
    z_err = _as_float(rows, "z_error")
    theta = _as_float(rows, "theta")
    omega = _as_float(rows, "omega")
    crash = next((r.get("crash_reason", "") for r in reversed(rows) if r.get("crash_reason", "")), "")
    return {
        "max x error": float(np.max(np.abs(x_err))) if x_err.size else 0.0,
        "max z error": float(np.max(np.abs(z_err))) if z_err.size else 0.0,
        "final x error": float(x_err[-1]) if x_err.size else 0.0,
        "final z error": float(z_err[-1]) if z_err.size else 0.0,
        "max theta deg": float(np.rad2deg(np.max(np.abs(theta)))) if theta.size else 0.0,
        "max omega deg/s": float(np.rad2deg(np.max(np.abs(omega)))) if omega.size else 0.0,
        "percent saturated": _pct(_as_float(rows, "mixer_saturated")),
        "percent authority limited": _pct(_as_float(rows, "mixer_authority_limited")),
        "motor saturation percent": _pct(_as_float(rows, "motor_saturated")),
        "servo rate saturation percent": _pct(_as_float(rows, "servo_rate_saturated")),
        "crash reason": crash or "-",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an interactive simulator CSV log.")
    parser.add_argument("csv", help="Path to results/interactive_logs/<log>.csv")
    args = parser.parse_args()
    metrics = analyze(args.csv)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()