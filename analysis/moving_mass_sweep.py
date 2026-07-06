from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from ..config import MovingMassConfig, RigidBodyConfig
except ImportError:  # pragma: no cover
    from config import MovingMassConfig, RigidBodyConfig


def sweep_rows() -> list[dict[str, float | int]]:
    rb = RigidBodyConfig()
    rows: list[dict[str, float | int]] = []
    for mass_ratio in np.linspace(0.1, 0.7, 7):
        for inertia_ratio in np.geomspace(0.1, 10.0, 7):
            for q_limit_deg in (5.0, 15.0, 25.0, 40.0):
                for tw in (0.8, 1.2, 1.8, 2.5):
                    total_mass = 1.5
                    mm = MovingMassConfig(
                        m_body_without_battery=total_mass * (1.0 - mass_ratio),
                        m_moving=total_mass * mass_ratio,
                        I_moving_about_hinge=0.025 * inertia_ratio,
                        q_limit=np.deg2rad(q_limit_deg),
                        thrust=tw * total_mass * 9.81,
                    )
                    I_total = mm.I_body_without_battery + mm.I_moving_about_hinge
                    reaction_moment = abs(mm.I_moving_about_hinge * mm.q_accel_limit)
                    expected_delta = abs(mm.I_moving_about_hinge / I_total * mm.q_limit)
                    max_cg_offset = mass_ratio * abs(mm.mass_center_offset_body[1] * np.sin(mm.q_limit))
                    cg_moment = mm.thrust * max_cg_offset
                    vane_moment = abs(rb.k_moment) * mm.thrust * rb.vane_angle_max
                    rows.append(
                        {
                            "moving_mass_ratio": mass_ratio,
                            "inertia_ratio": inertia_ratio,
                            "q_limit_deg": q_limit_deg,
                            "q_rate_limit": mm.q_rate_limit,
                            "q_accel_limit": mm.q_accel_limit,
                            "thrust_to_weight": tw,
                            "vane_angle_max_deg": rb.vane_angle_max_deg,
                            "vane_area_over_disk_area": rb.vane_count_effective * rb.vane_area / (np.pi * (0.5 * rb.duct_diameter) ** 2),
                            "max_reaction_angular_accel": reaction_moment / I_total,
                            "expected_body_angle_change": expected_delta,
                            "max_cg_offset": max_cg_offset,
                            "max_cg_offset_moment": cg_moment,
                            "max_vane_moment": vane_moment,
                            "moving_mass_to_vane_ratio": (reaction_moment + cg_moment) / max(vane_moment, 1e-9),
                            "mass_can_reach_10deg": int(expected_delta >= np.deg2rad(10.0)),
                            "vane_can_damp_rate_proxy": int(vane_moment > reaction_moment * 0.25),
                            "hybrid_authority_margin": (reaction_moment + cg_moment + vane_moment) / max(vane_moment, 1e-9),
                        }
                    )
    return rows


def save_sweep_csv(path: str | Path = "results/analysis/moving_mass_sweep.csv") -> Path:
    rows = sweep_rows()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
