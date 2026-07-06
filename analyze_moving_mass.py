from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .analysis.moving_mass_sweep import save_sweep_csv
except ImportError:  # pragma: no cover
    from analysis.moving_mass_sweep import save_sweep_csv


def _load_rows(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: np.asarray([float(r[key]) for r in rows]) for key in rows[0]}


def _scatter(x, y, c, path: Path, title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.scatter(x, y, c=c, s=24)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    out = Path("results/analysis")
    csv_path = save_sweep_csv(out / "moving_mass_sweep.csv")
    data = _load_rows(csv_path)
    plots = [
        _scatter(data["moving_mass_ratio"], data["inertia_ratio"], data["max_reaction_angular_accel"], out / "moving_mass_heatmap_reaction.png", "Reaction authority", "moving mass ratio", "inertia ratio"),
        _scatter(data["moving_mass_ratio"], data["q_limit_deg"], data["max_cg_offset_moment"], out / "moving_mass_heatmap_cg_offset.png", "CG-offset moment", "moving mass ratio", "q limit [deg]"),
        _scatter(data["moving_mass_to_vane_ratio"], data["hybrid_authority_margin"], data["thrust_to_weight"], out / "vane_vs_moving_mass_authority.png", "Vane vs moving mass", "moving/vane ratio", "hybrid margin"),
    ]
    report = out / "moving_mass_analysis.md"
    report.write_text(
        "# Moving Mass Sweep\n\n"
        "This sweep varies symbolic parameters to reveal trends. It is not fit to bench data.\n",
        encoding="utf-8",
    )
    print(f"saved: {csv_path}")
    for path in plots:
        print(f"saved: {path}")
    print(f"saved: {report}")


if __name__ == "__main__":
    main()
