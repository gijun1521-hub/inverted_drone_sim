from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import TypeVar

try:
    from .config import RigidBodyConfig
except ImportError:  # pragma: no cover - direct script execution
    from config import RigidBodyConfig

T = TypeVar("T")


def load_json_overrides(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("parameter file root must be a JSON object")
    return data


def apply_dataclass_overrides(instance: T, overrides: dict) -> T:
    allowed = {f.name for f in fields(instance)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
    return replace(instance, **overrides)


def load_rigid_body_config(path: str | Path | None = None, base: RigidBodyConfig | None = None) -> RigidBodyConfig:
    cfg = base or RigidBodyConfig()
    if path is None:
        return cfg
    data = load_json_overrides(path)
    return apply_dataclass_overrides(cfg, data)
