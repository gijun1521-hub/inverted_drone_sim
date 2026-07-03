from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from .config import DroneConfig
    from .pid_controller import ControlBreakdown
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DroneConfig
    from pid_controller import ControlBreakdown


CSV_FIELDS = [
    "time",
    "x",
    "z",
    "theta_deg",
    "vx",
    "vz",
    "omega",
    "cg_x",
    "cg_z",
    "target_x",
    "target_z",
    "ax_cmd",
    "throttle",
    "thrust",
    "theta_control_term",
    "omega_control_term",
    "position_control_term",
    "velocity_control_term",
    "theta_ddot",
    "ax_saturated",
]


def cg_position(state: np.ndarray, cfg: DroneConfig) -> tuple[float, float]:
    x, z, theta, *_ = state
    return (
        float(x + cfg.l * np.sin(theta)),
        float(z + cfg.l * np.cos(theta)),
    )


def assert_geometry_sign(state: np.ndarray, cfg: DroneConfig, atol: float = 1e-12) -> None:
    x = float(state[0])
    theta = float(state[2])
    cg_x, _cg_z = cg_position(state, cfg)

    if theta < -atol and not cg_x < x:
        raise AssertionError("theta < 0 must place CG left of the thrust point")
    if theta > atol and not cg_x > x:
        raise AssertionError("theta > 0 must place CG right of the thrust point")


def make_log_row(
    time: float,
    state: np.ndarray,
    action: np.ndarray,
    cfg: DroneConfig,
    control: ControlBreakdown,
    theta_ddot: float,
) -> dict[str, float | int]:
    cg_x, cg_z = cg_position(state, cfg)
    throttle = float(action[0])
    ax_cmd = float(action[1])
    unclipped_ax = (
        control.theta_term
        + control.omega_term
        + control.position_term
        + control.velocity_term
    )

    return {
        "time": float(time),
        "x": float(state[0]),
        "z": float(state[1]),
        "theta_deg": float(np.rad2deg(state[2])),
        "vx": float(state[3]),
        "vz": float(state[4]),
        "omega": float(state[5]),
        "cg_x": cg_x,
        "cg_z": cg_z,
        "target_x": float(cfg.target_x),
        "target_z": float(cfg.target_z),
        "ax_cmd": ax_cmd,
        "throttle": throttle,
        "thrust": throttle * cfg.T_max,
        "theta_control_term": float(control.theta_term),
        "omega_control_term": float(control.omega_term),
        "position_control_term": float(control.position_term),
        "velocity_control_term": float(control.velocity_term),
        "theta_ddot": float(theta_ddot),
        "ax_saturated": int(abs(unclipped_ax) > cfg.ax_cmd_max),
    }


def save_csv(rows: list[dict[str, float | int]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
