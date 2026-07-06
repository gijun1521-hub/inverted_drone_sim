from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    details: str
    metrics: dict[str, float | int | str]
