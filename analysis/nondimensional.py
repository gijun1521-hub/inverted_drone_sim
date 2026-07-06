from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from ..config import MovingMassConfig, RigidBodyConfig
except ImportError:  # pragma: no cover
    from config import MovingMassConfig, RigidBodyConfig


def rigid_body_groups(cfg: RigidBodyConfig, thrust: float | None = None, requested_moment: float = 0.1) -> dict[str, float]:
    thrust = cfg.hover_thrust if thrust is None else thrust
    max_vane_moment = abs(cfg.k_moment) * thrust * cfg.vane_angle_max
    return {
        "thrust_to_weight": thrust / (cfg.m * cfg.g),
        "vane_angle_fraction": 1.0,
        "vane_authority": max_vane_moment / cfg.Iyy,
        "normalized_pitch_accel": (max_vane_moment / cfg.Iyy) / cfg.g,
        "control_moment_ratio": max_vane_moment / max(abs(requested_moment), 1e-9),
        "authority_margin": max_vane_moment / max(abs(requested_moment), 1e-9),
    }


def moving_mass_groups(cfg: MovingMassConfig, qddot: float | None = None, thrust: float | None = None) -> dict[str, float]:
    qddot = cfg.q_accel_limit if qddot is None else qddot
    thrust = cfg.thrust if thrust is None else thrust
    I_total = cfg.I_body_without_battery + cfg.I_moving_about_hinge
    reaction = abs(cfg.I_moving_about_hinge * qddot)
    cg_offset = cfg.m_moving / cfg.m_total * abs(cfg.mass_center_offset_body[1] * np.sin(cfg.q_limit))
    thrust_offset = thrust * cg_offset
    return {
        "moving_mass_ratio": cfg.m_moving / cfg.m_total,
        "inertia_ratio": cfg.I_moving_about_hinge / max(cfg.I_body_without_battery, 1e-9),
        "q_fraction": 1.0,
        "q_rate_fraction": 1.0,
        "q_accel_fraction": 1.0,
        "reaction_authority": reaction / max(I_total, 1e-9),
        "cg_offset_fraction": cg_offset / cfg.H,
        "thrust_offset_moment": thrust_offset,
    }


def save_nondimensional_summary(path: str | Path, rb_cfg: RigidBodyConfig | None = None, mm_cfg: MovingMassConfig | None = None) -> Path:
    rb_cfg = rb_cfg or RigidBodyConfig()
    mm_cfg = mm_cfg or MovingMassConfig()
    row = {}
    row.update({f"rigid_{k}": v for k, v in rigid_body_groups(rb_cfg).items()})
    row.update({f"moving_{k}": v for k, v in moving_mass_groups(mm_cfg).items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path
