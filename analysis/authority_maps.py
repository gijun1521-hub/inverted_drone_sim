from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from ..config import MovingMassConfig, RigidBodyConfig
except ImportError:  # pragma: no cover
    from config import MovingMassConfig, RigidBodyConfig


def vane_authority_grid(cfg: RigidBodyConfig, thrust_to_weight_values, vane_fraction_values) -> np.ndarray:
    grid = np.zeros((len(thrust_to_weight_values), len(vane_fraction_values)))
    for i, tw in enumerate(thrust_to_weight_values):
        thrust = tw * cfg.m * cfg.g
        for j, frac in enumerate(vane_fraction_values):
            grid[i, j] = abs(cfg.k_moment) * thrust * abs(frac) * cfg.vane_angle_max
    return grid


def moving_mass_reaction_grid(cfg: MovingMassConfig, inertia_ratios, q_accel_values) -> np.ndarray:
    grid = np.zeros((len(inertia_ratios), len(q_accel_values)))
    for i, ratio in enumerate(inertia_ratios):
        I_moving = cfg.I_body_without_battery * ratio
        for j, qddot in enumerate(q_accel_values):
            grid[i, j] = abs(I_moving * qddot)
    return grid


def save_heatmap(data: np.ndarray, x, y, path: str | Path, title: str, xlabel: str, ylabel: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(data, origin="lower", aspect="auto", extent=[min(x), max(x), min(y), max(y)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, label="moment [N m]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_authority_maps(results_dir: str | Path = "results/analysis") -> list[Path]:
    rb = RigidBodyConfig()
    mm = MovingMassConfig()
    results_dir = Path(results_dir)
    tw = np.linspace(0.8, 2.5, 40)
    vf = np.linspace(0.0, 1.0, 40)
    vane = vane_authority_grid(rb, tw, vf)
    inertia = np.linspace(0.1, 10.0, 40)
    accel = np.linspace(1.0, mm.q_accel_limit, 40)
    moving = moving_mass_reaction_grid(mm, inertia, accel)
    hybrid = vane.mean(axis=1)[:, None] + moving.mean(axis=0)[None, :]
    return [
        save_heatmap(vane, vf, tw, results_dir / "vane_authority_map.png", "Vane authority", "vane fraction", "T/W"),
        save_heatmap(moving, accel, inertia, results_dir / "moving_mass_authority_map.png", "Moving-mass reaction authority", "q accel [rad/s^2]", "I moving / I body"),
        save_heatmap(hybrid, accel, tw, results_dir / "hybrid_authority_map.png", "Hybrid authority proxy", "q accel [rad/s^2]", "T/W"),
    ]
