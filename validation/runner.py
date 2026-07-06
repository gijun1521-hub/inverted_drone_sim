from __future__ import annotations

from pathlib import Path

try:
    from ..config import RigidBodyConfig
    from .report import save_markdown, save_summary
    from .scenarios import SCENARIOS
except ImportError:  # pragma: no cover
    from config import RigidBodyConfig
    from validation.report import save_markdown, save_summary
    from validation.scenarios import SCENARIOS


def run_validation(cfg: RigidBodyConfig, results_dir: str | Path = "results"):
    results = [scenario(cfg) for scenario in SCENARIOS]
    results_dir = Path(results_dir)
    summary = save_summary(results, results_dir / "validation_summary.csv")
    report = save_markdown(results, results_dir / "validation_report.md")
    return results, summary, report
