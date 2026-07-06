from __future__ import annotations

import csv
from pathlib import Path

from .metrics import ScenarioResult


def save_summary(results: list[ScenarioResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({k for r in results for k in r.metrics})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "passed", "details", *metric_keys])
        writer.writeheader()
        for r in results:
            row = {"name": r.name, "passed": int(r.passed), "details": r.details}
            row.update(r.metrics)
            writer.writerow(row)
    return path


def save_markdown(results: list[ScenarioResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Validation Report",
        "",
        f"Passed: {passed}/{len(results)}",
        "",
        "| Scenario | Result | Details |",
        "| --- | --- | --- |",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"| {r.name} | {status} | {r.details} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
