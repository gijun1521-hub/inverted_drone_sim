from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import TypeVar

try:
    from .config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
except ImportError:  # pragma: no cover - direct script execution
    from config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig

T = TypeVar("T")


STRUCTURED_SECTIONS = {"rigid_body", "interactive", "controller"}


def load_json_overrides(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("parameter file root must be a JSON object")
    return data


def apply_dataclass_overrides(instance: T, overrides: dict, section: str = "parameters") -> T:
    allowed = {f.name for f in fields(instance)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} parameter(s): {', '.join(unknown)}")
    return replace(instance, **overrides)


def _split_overrides(data: dict) -> tuple[dict, dict, dict]:
    structured_keys = set(data) & STRUCTURED_SECTIONS
    if structured_keys:
        unknown = sorted(set(data) - STRUCTURED_SECTIONS)
        if unknown:
            raise ValueError(
                "structured parameter files may only contain rigid_body, interactive, and controller sections; "
                f"unknown section(s): {', '.join(unknown)}"
            )
        sections = {}
        for key in STRUCTURED_SECTIONS:
            section = data.get(key, {})
            if not isinstance(section, dict):
                raise ValueError(f"{key} section must be a JSON object")
            sections[key] = section
        return sections["rigid_body"], sections["interactive"], sections["controller"]
    return data, {}, {}


def load_config_bundle(
    path: str | Path | None = None,
    rb_base: RigidBodyConfig | None = None,
    ui_base: InteractiveSimConfig | None = None,
    controller_base: ControllerConfig | None = None,
) -> tuple[RigidBodyConfig, InteractiveSimConfig, ControllerConfig]:
    rb_cfg = rb_base or RigidBodyConfig()
    ui_cfg = ui_base or InteractiveSimConfig(physics_dt=rb_cfg.dt, controller_dt=0.01)
    controller_cfg = controller_base or ControllerConfig()
    if path is None:
        return rb_cfg, ui_cfg, controller_cfg

    data = load_json_overrides(path)
    rb_overrides, ui_overrides, controller_overrides = _split_overrides(data)
    rb_cfg = apply_dataclass_overrides(rb_cfg, rb_overrides, "rigid_body")
    if ui_base is None:
        ui_cfg = InteractiveSimConfig(physics_dt=rb_cfg.dt, controller_dt=ui_cfg.controller_dt)
    ui_cfg = apply_dataclass_overrides(ui_cfg, ui_overrides, "interactive")
    controller_cfg = apply_dataclass_overrides(controller_cfg, controller_overrides, "controller")
    return rb_cfg, ui_cfg, controller_cfg


def load_interactive_config(path: str | Path | None = None) -> tuple[RigidBodyConfig, InteractiveSimConfig, ControllerConfig]:
    return load_config_bundle(path)


def load_rigid_body_config(path: str | Path | None = None, base: RigidBodyConfig | None = None) -> RigidBodyConfig:
    if path is None:
        return base or RigidBodyConfig()
    rb_cfg, _ui_cfg, _controller_cfg = load_config_bundle(path, rb_base=base)
    return rb_cfg