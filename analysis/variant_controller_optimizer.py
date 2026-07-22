"""Deterministic independent controller optimization for the two actuator variants.

The workflow is deliberately simulation-only.  It does not change the rigid-body
model, and it treats the merged PR #24 controller/results as preserved reference
material rather than as output files to update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .final_loiter_regression import PATTERNS, _audit as audit_loiter_pattern
from .headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter
from .moving_mass_gain_resweep import (
    MOVING_MASS_CHATTER_THRESHOLDS,
    MOVING_MASS_LIMITER_HARD_GATES,
    MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE,
    _moving_mass_metrics,
    atomic_csv,
    atomic_json,
    atomic_text,
)
from .pitch_damping_retune import (
    CHATTER_THRESHOLDS,
    HARD_GATE_THRESHOLDS,
    ScenarioDefinition,
    _canonical_json,
    _sha256_bytes,
    asymmetry_fraction,
    compute_metrics,
    required_scenarios,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "params" / "variant_controller_search_space.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "variant_controller_optimization"
PR24_RESULTS = REPO_ROOT / "results" / "analysis" / "final_loiter_regression"
PR24_PROFILE = REPO_ROOT / "params" / "moving_mass_gain_resweep_provisional.json"
VARIANTS = ("vane_only", "moving_mass_assist")
SEARCH_PARAMETER_ORDER = (
    "atc_rat_pit_p",
    "atc_rat_pit_d",
    "atc_ang_pit_p",
    "psc_ne_pos_p",
    "psc_ne_vel_p",
    "moving_mass_assist_gain_m_per_Nm",
)
OLD_CONTROLLER = {
    "atc_rat_pit_p": 0.070,
    "atc_rat_pit_d": 0.008,
    "atc_ang_pit_p": 10.0,
    "psc_ne_pos_p": 0.55,
    "psc_ne_vel_p": 0.70,
}
PR24_CONTROLLER = {
    "atc_rat_pit_p": 0.09375,
    "atc_rat_pit_d": 0.021,
    "atc_ang_pit_p": 25.0,
    "psc_ne_pos_p": 0.55,
    "psc_ne_vel_p": 0.70,
}
PR24_ASSIST_GAIN = 0.0415
SYMMETRY_PAIRS = (
    ("forward_1m", "backward_1m"),
    ("loiter_positive_disturbance", "loiter_negative_disturbance"),
)
SYMMETRY_METRICS = (
    "settling_time_s",
    "rise_time_s",
    "overshoot_fraction",
    "final_abs_position_error_m",
    "tail_rms_position_error_m",
    "tail_rms_horizontal_velocity_m_s",
    "peak_abs_pitch_deg",
    "vane_command_rms_deg",
    "moving_mass_rms_displacement_m",
)


@dataclass(frozen=True)
class VariantCandidate:
    variant: str
    stage: str
    atc_rat_pit_p: float
    atc_rat_pit_d: float
    atc_ang_pit_p: float
    psc_ne_pos_p: float
    psc_ne_vel_p: float
    moving_mass_assist_gain_m_per_Nm: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant: {self.variant}")
        if self.variant == "vane_only" and self.moving_mass_assist_gain_m_per_Nm != 0.0:
            raise ValueError("Vane-only moving-mass gain must be exactly zero")

    @property
    def parameters(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SEARCH_PARAMETER_ORDER}

    @property
    def key(self) -> str:
        values = ":".join(f"{name}={self.parameters[name]:.10g}" for name in SEARCH_PARAMETER_ORDER)
        return f"{self.variant}:{values}"

    def controller_overrides(self, spec: Mapping[str, Any]) -> dict[str, float | bool | int]:
        return {
            **spec["fixed"],
            **{name: value for name, value in self.parameters.items() if name != "moving_mass_assist_gain_m_per_Nm"},
            "enable_noise": False,
            "random_seed": int(spec["seed"]),
        }


def load_search_spec(path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in SEARCH_PARAMETER_ORDER if name not in spec.get("ranges", {})]
    if missing:
        raise ValueError(f"missing search ranges: {', '.join(missing)}")
    for name, bounds in spec["ranges"].items():
        if len(bounds) != 2 or not float(bounds[0]) < float(bounds[1]):
            raise ValueError(f"invalid range for {name}: {bounds!r}")
    if float(spec["fixed"].get("atc_rat_pit_i", math.nan)) != 0.0:
        raise ValueError("atc_rat_pit_i must remain exactly zero")
    preferred = spec["targets"]["preferred_overshoot_fraction"]
    if len(preferred) != 2 or not 0.0 <= float(preferred[0]) <= float(preferred[1]):
        raise ValueError("invalid preferred overshoot band")
    return spec


def overshoot_band_distance(value: float, preferred: Sequence[float]) -> float:
    lower, upper = map(float, preferred)
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def target_crossing_count(error: Sequence[float], deadband: float = 0.002) -> int:
    """Count meaningful target crossings while ignoring samples inside a deadband."""
    array = np.asarray(error, dtype=float)
    signs = np.sign(array[np.abs(array) > float(deadband)])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def detect_settling_time(
    times: Sequence[float],
    position_error: Sequence[float],
    velocity: Sequence[float],
    *,
    event_time_s: float,
    position_band_m: float,
    velocity_band_m_s: float,
    required_duration_s: float,
) -> tuple[float | None, float, bool]:
    """Return the first final entry into the settling band and its duration.

    The condition must remain true through the end of the record, so an early
    0.75 s quiet interval followed by a second lobe is not accepted.
    """
    t = np.asarray(times, dtype=float)
    error = np.asarray(position_error, dtype=float)
    speed = np.asarray(velocity, dtype=float)
    if not (t.size and t.size == error.size == speed.size):
        return None, 0.0, False
    condition = (
        (t >= float(event_time_s) - 1e-12)
        & (np.abs(error) <= float(position_band_m))
        & (np.abs(speed) <= float(velocity_band_m_s))
    )
    suffix_all = np.logical_and.accumulate(condition[::-1])[::-1]
    for index in np.flatnonzero(suffix_all):
        duration = float(t[-1] - t[index])
        if duration + 1e-12 >= float(required_duration_s):
            return float(t[index] - event_time_s), duration, True
    longest = current = 0.0
    previous = float(t[0])
    for sample_time, active in zip(t, condition):
        dt = max(0.0, float(sample_time) - previous)
        current = current + dt if active else 0.0
        longest = max(longest, current)
        previous = float(sample_time)
    return None, longest, False


def _array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write portable LF-only CSV so manifest bytes survive every checkout."""
    rows = [dict(row) for row in rows]
    if not rows:
        atomic_text(path, "")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def _save_timeseries_atomic(result: LoiterRunResult, path: Path) -> None:
    atomic_csv(path, result.rows)


def response_metrics(
    definition: ScenarioDefinition,
    result: LoiterRunResult,
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    rows = result.rows
    times = _array(rows, "time")
    x = _array(rows, "x")
    vx = _array(rows, "vx")
    x_error = _array(rows, "x_error")
    event_time = float(definition.event_time_s)
    post_indices = np.flatnonzero(times >= event_time - 1e-12)
    start_index = int(post_indices[0]) if post_indices.size else 0
    target = float(definition.config.target_step_x if definition.config.target_step_x is not None else definition.config.target_x)
    initial = float(x[max(0, start_index - 1)])
    amplitude = abs(target - initial)
    direction = 1.0 if target >= initial else -1.0
    post_t = times[start_index:]
    post_x = x[start_index:]
    post_error = x_error[start_index:]

    def first_time(mask: np.ndarray) -> float | None:
        indices = np.flatnonzero(mask)
        return float(post_t[indices[0]] - event_time) if indices.size else None

    is_step = amplitude >= 0.5
    if is_step:
        progress = direction * (post_x - initial) / amplitude
        t10 = first_time(progress >= 0.10)
        t90 = first_time(progress >= 0.90)
        rise_time = None if t10 is None or t90 is None else max(0.0, t90 - t10)
        first_crossing = first_time(progress >= 1.0)
        overshoot_m = max(0.0, float(np.max(direction * (post_x - target))))
        overshoot_fraction = overshoot_m / amplitude
        crossings = target_crossing_count(post_x - target, float(targets["target_crossing_deadband_m"]))
    else:
        rise_time = None
        first_crossing = None
        overshoot_m = 0.0
        overshoot_fraction = 0.0
        crossings = 0

    settling_time, continuous_duration, settled = detect_settling_time(
        times,
        x_error,
        vx,
        event_time_s=event_time,
        position_band_m=float(targets["position_settling_band_m"]),
        velocity_band_m_s=float(targets["velocity_settling_band_m_s"]),
        required_duration_s=float(targets["required_continuous_settling_duration_s"]),
    )
    tail = times >= max(event_time, float(times[-1]) - 2.5) - 1e-12
    return {
        "is_position_step": is_step,
        "rise_time_s": rise_time,
        "first_target_crossing_s": first_crossing,
        "overshoot_m": overshoot_m,
        "overshoot_fraction": overshoot_fraction,
        "overshoot_band_distance": overshoot_band_distance(
            overshoot_fraction, targets["preferred_overshoot_fraction"]
        ),
        "soft_overshoot_exceeded": overshoot_fraction > float(targets["soft_maximum_overshoot_fraction"]),
        "target_crossing_count": crossings,
        "settling_time_s": settling_time,
        "continuous_settling_duration_s": continuous_duration,
        "settled": settled,
        "final_abs_position_error_m": abs(float(x_error[-1])),
        "tail_rms_position_error_m": _rms(x_error[tail]),
        "tail_rms_velocity_m_s": _rms(vx[tail]),
        "tail_path_length_m": float(np.sum(np.abs(np.diff(x[tail])))),
    }


def _candidate_parameter_reasons(candidate: VariantCandidate, metrics: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for name, expected in candidate.parameters.items():
        if name == "moving_mass_assist_gain_m_per_Nm":
            observed = float(metrics.get(name, math.nan))
        else:
            observed = float(metrics.get(f"effective_{name}", math.nan))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            reasons.append(f"effective_parameter_mismatch:{name}")
    if not math.isclose(float(metrics.get("effective_atc_rat_pit_i", math.nan)), 0.0, rel_tol=0.0, abs_tol=0.0):
        reasons.append("effective_parameter_mismatch:atc_rat_pit_i")
    return reasons


def hard_gate_reasons(
    candidate: VariantCandidate,
    definition: ScenarioDefinition,
    metrics: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[str]:
    """Apply every pre-existing controller/mass gate plus new response gates."""
    reasons: list[str] = []
    if not bool(metrics.get("finite", False)):
        reasons.append("non_finite")
    if bool(metrics.get("crash", False)):
        reasons.append("crash")
    if bool(metrics.get("ground_contact", False)):
        reasons.append("ground_contact")
    if float(metrics.get("peak_abs_pitch_deg", math.inf)) > HARD_GATE_THRESHOLDS["pitch_divergence_deg"]:
        reasons.append("pitch_divergence")
    for key in (
        "premature_pause",
        "early_velocity_reversal",
        "second_acceleration_lobe_after_full_pause",
        "capture_discontinuity",
        "shaped_velocity_sign_reversal_after_release",
    ):
        if bool(metrics.get(key, False)):
            reasons.append(key)
    if definition.requires_capture_gates and int(metrics.get("target_capture_count", 0)) != 1:
        reasons.append("capture_count_not_one")
    for metric, threshold, reason in (
        ("vane_saturation_percent", HARD_GATE_THRESHOLDS["vane_saturation_percent"], "excessive_vane_saturation"),
        ("servo_rate_saturation_percent", HARD_GATE_THRESHOLDS["servo_rate_saturation_percent"], "excessive_servo_rate_saturation"),
        ("mixer_saturation_percent", HARD_GATE_THRESHOLDS["mixer_saturation_percent"], "excessive_mixer_saturation"),
    ):
        if float(metrics.get(metric, math.inf)) > float(threshold):
            reasons.append(reason)
    if (
        int(metrics.get("meaningful_vane_sign_change_count", 0)) > CHATTER_THRESHOLDS["max_meaningful_sign_changes"]
        or float(metrics.get("vane_total_variation_per_second_deg_s", math.inf)) > CHATTER_THRESHOLDS["max_total_variation_per_second_deg_s"]
        or float(metrics.get("tail_high_frequency_vane_energy_deg2", math.inf)) > CHATTER_THRESHOLDS["max_tail_high_frequency_energy_deg2"]
    ):
        reasons.append("vane_chatter")
    for name, limits in MOVING_MASS_LIMITER_HARD_GATES.items():
        if float(metrics.get(f"moving_mass_{name}_duty_percent", math.inf)) > limits["max_duty_percent"]:
            reasons.append(f"excessive_moving_mass_{name}_duty")
        if float(metrics.get(f"moving_mass_{name}_longest_continuous_duration_s", math.inf)) > limits["max_continuous_duration_s"]:
            reasons.append(f"excessive_moving_mass_{name}_continuous_duration")
    physical = (
        ("moving_mass_max_abs_offset_m", "effective_moving_mass_max_offset_m", "offset"),
        ("moving_mass_max_abs_velocity_m_s", "effective_moving_mass_max_rate_m_s", "rate"),
        ("moving_mass_max_abs_acceleration_m_s2", "effective_moving_mass_max_accel_m_s2", "acceleration"),
    )
    for observed_key, limit_key, label in physical:
        if float(metrics.get(observed_key, math.inf)) > float(metrics.get(limit_key, 0.0)) + MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE:
            reasons.append(f"moving_mass_actual_{label}_physical_limit_violation")
    if (
        int(metrics.get("meaningful_moving_mass_direction_change_count", 0)) > MOVING_MASS_CHATTER_THRESHOLDS["max_meaningful_direction_changes"]
        or float(metrics.get("moving_mass_total_travel_per_second_m_s", math.inf)) > MOVING_MASS_CHATTER_THRESHOLDS["max_total_travel_per_second_m_s"]
        or float(metrics.get("tail_high_frequency_moving_mass_energy_m2", math.inf)) > MOVING_MASS_CHATTER_THRESHOLDS["max_tail_high_frequency_energy_m2"]
    ):
        reasons.append("moving_mass_chatter")
    reasons.extend(_candidate_parameter_reasons(candidate, metrics))
    if not bool(metrics.get("moving_mass_enabled", False)):
        reasons.append("physical_moving_mass_disabled")
    if not bool(metrics.get("total_com_geometry_active", False)):
        reasons.append("total_com_geometry_disabled")
    if bool(metrics.get("legacy_gravity_offset_active", True)):
        reasons.append("legacy_gravity_offset_active")
    for key, expected in (("total_mass_kg", 2.0), ("physical_moving_mass_kg", 0.5)):
        if not math.isclose(float(metrics.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-12):
            reasons.append(f"effective_parameter_mismatch:{key}")
    if candidate.variant == "vane_only":
        exact_zero_fields = (
            "moving_mass_assist_gain_m_per_Nm",
            "moving_mass_max_abs_target_m",
            "moving_mass_max_abs_offset_m",
            "moving_mass_max_abs_velocity_m_s",
            "moving_mass_max_abs_acceleration_m_s2",
        )
        for key in exact_zero_fields:
            if float(metrics.get(key, math.inf)) != 0.0:
                reasons.append(f"vane_only_not_exactly_locked:{key}")
    if bool(metrics.get("is_position_step", False)):
        targets = spec["targets"]
        if float(metrics.get("overshoot_fraction", math.inf)) > float(targets["hard_maximum_overshoot_fraction"]):
            reasons.append("hard_overshoot_limit")
        if int(metrics.get("target_crossing_count", 10**9)) > int(targets["maximum_target_crossings"]):
            reasons.append("repeated_target_crossing_limit")
        if not bool(metrics.get("settled", False)):
            reasons.append("failure_to_enter_or_remain_settled")
        if float(metrics.get("continuous_settling_duration_s", 0.0)) + 1e-12 < float(targets["required_continuous_settling_duration_s"]):
            reasons.append("continuous_settling_duration")
    return list(dict.fromkeys(reasons))


def screening_scenarios(spec: Mapping[str, Any]) -> tuple[ScenarioDefinition, ...]:
    duration = float(spec["search"]["fast_screen_duration_s"])
    wanted = {
        "forward_1m",
        "backward_1m",
        "loiter_positive_disturbance",
        "loiter_negative_disturbance",
    }
    return tuple(
        replace(item, config=replace(item.config, duration_s=duration))
        for item in required_scenarios(False)
        if item.key in wanted
    )


def evaluate_scenario(
    candidate: VariantCandidate,
    definition: ScenarioDefinition,
    spec: Mapping[str, Any],
    *,
    keep_result: bool = False,
) -> tuple[dict[str, Any], LoiterRunResult | None]:
    scenario = replace(
        definition.config,
        moving_mass_enabled=True,
        moving_mass_target_m=0.0,
        moving_mass_assist_gain_m_per_Nm=candidate.moving_mass_assist_gain_m_per_Nm,
    )
    source = REPO_ROOT / str(spec["source_profile"])
    result = run_headless_loiter(
        source,
        scenario,
        controller_overrides=candidate.controller_overrides(spec),
    )
    metrics = compute_metrics(definition, result, quick=False)
    metrics.update(response_metrics(definition, result, spec["targets"]))
    metrics.update(_moving_mass_metrics(result))
    metrics.update(
        {
            "effective_moving_mass_max_offset_m": result.metrics["effective_moving_mass_max_offset_m"],
            "effective_moving_mass_max_rate_m_s": result.metrics["effective_moving_mass_max_rate_m_s"],
            "effective_moving_mass_max_accel_m_s2": result.metrics["effective_moving_mass_max_accel_m_s2"],
        }
    )
    reasons = hard_gate_reasons(candidate, definition, metrics, spec)
    metrics["rejected"] = bool(reasons)
    metrics["rejection_reasons"] = "; ".join(reasons)
    return metrics, result if keep_result else None


def _finite_or_penalty(value: Any, penalty: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return penalty
    return number if math.isfinite(number) else penalty


def evaluate_candidate(candidate: VariantCandidate, spec: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    scenario_rows = []
    for definition in screening_scenarios(spec):
        metrics, _ = evaluate_scenario(candidate, definition, spec)
        scenario_rows.append(metrics)
    by_name = {str(row["scenario_name"]): row for row in scenario_rows}
    reasons = [
        f"{row['scenario_name']}:{reason}"
        for row in scenario_rows
        for reason in str(row.get("rejection_reasons", "")).split("; ")
        if reason
    ]
    symmetry: dict[str, float] = {}
    for positive, negative in SYMMETRY_PAIRS:
        for metric in SYMMETRY_METRICS:
            symmetry[f"{positive}__{negative}__{metric}"] = asymmetry_fraction(
                _finite_or_penalty(by_name[positive].get(metric), 1000.0),
                _finite_or_penalty(by_name[negative].get(metric), 1000.0),
            )
    worst_asymmetry = max(symmetry.values(), default=0.0)
    if worst_asymmetry > HARD_GATE_THRESHOLDS["symmetry_max_fraction"]:
        reasons.append("aggregate:severe_mirrored_scenario_asymmetry")
    steps = [row for row in scenario_rows if bool(row.get("is_position_step"))]
    settling = max((_finite_or_penalty(row.get("settling_time_s"), 1000.0) for row in steps), default=1000.0)
    rise = max((_finite_or_penalty(row.get("rise_time_s"), 1000.0) for row in steps), default=1000.0)
    band_distance = float(np.mean([float(row["overshoot_band_distance"]) for row in steps]))
    final_error = max((float(row["final_abs_position_error_m"]) for row in steps), default=1000.0)
    effort = float(
        np.mean(
            [
                float(row["vane_command_rms_deg"])
                + 0.01 * float(row["vane_command_total_variation_deg"])
                + 10.0 * float(row["moving_mass_total_travel_m"])
                + 0.01
                * sum(float(row[f"moving_mass_{name}_duty_percent"]) for name in MOVING_MASS_LIMITER_HARD_GATES)
                for row in scenario_rows
            ]
        )
    )
    robustness = float(
        worst_asymmetry
        + np.mean(
            [
                float(row["tail_rms_position_error_m"])
                + float(row["tail_rms_horizontal_velocity_m_s"])
                + 0.01 * float(row["peak_abs_pitch_deg"])
                for row in scenario_rows
            ]
        )
    )
    return {
        "candidate_key": candidate.key,
        "workflow_fingerprint": fingerprint,
        "variant": candidate.variant,
        "stage": candidate.stage,
        **candidate.parameters,
        "all_hard_gates_pass": not reasons,
        "rejected": bool(reasons),
        "rejection_reasons": "; ".join(dict.fromkeys(reasons)),
        "all_step_scenarios_settled": all(bool(row.get("settled")) for row in steps),
        "worst_settling_time_s": settling,
        "worst_rise_time_s": rise,
        "mean_overshoot_band_distance": band_distance,
        "mean_overshoot_fraction": float(np.mean([float(row["overshoot_fraction"]) for row in steps])),
        "worst_final_abs_position_error_m": final_error,
        "actuator_effort_index": effort,
        "robustness_index": robustness,
        "worst_asymmetry_fraction": worst_asymmetry,
        "scenario_metrics_json": _canonical_json(by_name),
        "symmetry_json": _canonical_json(symmetry),
    }


def candidate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """The requested lexicographic selection order, with a stable final key."""
    return (
        not bool(row.get("all_hard_gates_pass", False)),
        not bool(row.get("all_step_scenarios_settled", False)),
        _finite_or_penalty(row.get("worst_settling_time_s"), 1000.0),
        _finite_or_penalty(row.get("worst_rise_time_s"), 1000.0),
        _finite_or_penalty(row.get("mean_overshoot_band_distance"), 1000.0),
        _finite_or_penalty(row.get("worst_final_abs_position_error_m"), 1000.0),
        _finite_or_penalty(row.get("actuator_effort_index"), 1000.0),
        _finite_or_penalty(row.get("robustness_index"), 1000.0),
        str(row.get("candidate_key", "")),
    )


def rank_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted((dict(row) for row in rows), key=candidate_rank_key)
    for index, row in enumerate(ranked, 1):
        row["lexicographic_rank"] = index
    return ranked


PARETO_OBJECTIVES = (
    "worst_settling_time_s",
    "worst_rise_time_s",
    "mean_overshoot_band_distance",
    "worst_final_abs_position_error_m",
    "actuator_effort_index",
    "robustness_index",
)


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [dict(row) for row in rows if bool(row.get("all_hard_gates_pass", False))]
    front: list[dict[str, Any]] = []
    for candidate in valid:
        values = tuple(_finite_or_penalty(candidate.get(name), 1000.0) for name in PARETO_OBJECTIVES)
        dominated = False
        for other in valid:
            if other["candidate_key"] == candidate["candidate_key"]:
                continue
            other_values = tuple(_finite_or_penalty(other.get(name), 1000.0) for name in PARETO_OBJECTIVES)
            if all(a <= b for a, b in zip(other_values, values)) and any(a < b for a, b in zip(other_values, values)):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return rank_candidates(front)


class CandidateCache:
    """Parent-process atomic cache; worker processes never share mutable state."""

    def __init__(self, path: Path, fingerprint: str, *, resume: bool) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.rows: dict[str, dict[str, Any]] = {}
        if resume and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("workflow_fingerprint") == fingerprint:
                self.rows = {str(row["candidate_key"]): row for row in payload.get("candidates", [])}

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.rows.get(key)
        return dict(row) if row is not None else None

    def record(self, row: Mapping[str, Any]) -> None:
        self.rows[str(row["candidate_key"])] = dict(row)
        atomic_json(
            self.path,
            {
                "schema_version": 1,
                "workflow_fingerprint": self.fingerprint,
                "candidates": [self.rows[key] for key in sorted(self.rows)],
            },
        )


def evaluate_candidates(
    candidates: Sequence[VariantCandidate],
    spec: Mapping[str, Any],
    fingerprint: str,
    cache: CandidateCache,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    unique = {candidate.key: candidate for candidate in candidates}
    ordered = [unique[key] for key in sorted(unique)]
    pending = [candidate for candidate in ordered if cache.get(candidate.key) is None]
    if workers <= 1:
        for candidate in pending:
            cache.record(evaluate_candidate(candidate, spec, fingerprint))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate_candidate, candidate, dict(spec), fingerprint): candidate
                for candidate in pending
            }
            for future in as_completed(futures):
                cache.record(future.result())
    return [cache.get(candidate.key) for candidate in ordered if cache.get(candidate.key) is not None]


def _halton_value(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    while index > 0:
        fraction /= base
        index, remainder = divmod(index, base)
        result += remainder * fraction
    return result


def halton_points(count: int, dimensions: int, *, seed: int = 0) -> list[tuple[float, ...]]:
    primes = (2, 3, 5, 7, 11, 13)
    if dimensions > len(primes):
        raise ValueError("too many Halton dimensions")
    start = 1 + int(seed) % 997
    return [tuple(_halton_value(start + row, primes[col]) for col in range(dimensions)) for row in range(count)]


def _active_parameters(variant: str) -> tuple[str, ...]:
    return SEARCH_PARAMETER_ORDER if variant == "moving_mass_assist" else SEARCH_PARAMETER_ORDER[:-1]


def _candidate_from_parameters(variant: str, stage: str, values: Mapping[str, float]) -> VariantCandidate:
    return VariantCandidate(
        variant=variant,
        stage=stage,
        atc_rat_pit_p=float(values["atc_rat_pit_p"]),
        atc_rat_pit_d=float(values["atc_rat_pit_d"]),
        atc_ang_pit_p=float(values["atc_ang_pit_p"]),
        psc_ne_pos_p=float(values["psc_ne_pos_p"]),
        psc_ne_vel_p=float(values["psc_ne_vel_p"]),
        moving_mass_assist_gain_m_per_Nm=(
            float(values.get("moving_mass_assist_gain_m_per_Nm", 0.0)) if variant == "moving_mass_assist" else 0.0
        ),
    )


def coarse_candidates(variant: str, spec: Mapping[str, Any], *, quick: bool = False) -> list[VariantCandidate]:
    ranges = spec["ranges"]
    active = _active_parameters(variant)
    center = {name: float(np.mean(ranges[name])) for name in active}
    if variant == "vane_only":
        center["moving_mass_assist_gain_m_per_Nm"] = 0.0
    anchors = [
        {**OLD_CONTROLLER, "moving_mass_assist_gain_m_per_Nm": 0.0 if variant == "vane_only" else 0.1325},
        {**PR24_CONTROLLER, "moving_mass_assist_gain_m_per_Nm": 0.0 if variant == "vane_only" else PR24_ASSIST_GAIN},
        center,
    ]
    candidates = [_candidate_from_parameters(variant, "coarse_anchor", values) for values in anchors]
    for name in active:
        for boundary in ranges[name]:
            values = dict(center)
            values[name] = float(boundary)
            candidates.append(_candidate_from_parameters(variant, "coarse_boundary", values))
    count = 8 if quick else int(spec["search"]["coarse_samples"])
    for point in halton_points(count, len(active), seed=int(spec["seed"])):
        values = {
            name: float(ranges[name][0]) + coordinate * (float(ranges[name][1]) - float(ranges[name][0]))
            for name, coordinate in zip(active, point)
        }
        candidates.append(_candidate_from_parameters(variant, "coarse_halton", values))
    return list({candidate.key: candidate for candidate in candidates}.values())


def local_refinement_candidates(
    variant: str,
    best_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    quick: bool = False,
) -> list[VariantCandidate]:
    ranges = spec["ranges"]
    active = _active_parameters(variant)
    top_count = 1 if quick else int(spec["search"]["local_top_candidates"])
    fraction = float(spec["search"]["local_step_fraction"])
    candidates: list[VariantCandidate] = []
    for row in list(best_rows)[:top_count]:
        base = {name: float(row[name]) for name in active}
        for name in active:
            step = fraction * (float(ranges[name][1]) - float(ranges[name][0]))
            for sign in (-1.0, 1.0):
                values = dict(base)
                values[name] = min(float(ranges[name][1]), max(float(ranges[name][0]), base[name] + sign * step))
                candidates.append(_candidate_from_parameters(variant, "local_coordinate", values))
    return list({candidate.key: candidate for candidate in candidates}.values())


def joint_refinement_candidates(
    variant: str,
    best: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    quick: bool = False,
) -> list[VariantCandidate]:
    ranges = spec["ranges"]
    active = _active_parameters(variant)
    fraction = float(spec["search"]["joint_refinement_fraction"])
    local = {
        name: (
            max(float(ranges[name][0]), float(best[name]) - fraction * (float(ranges[name][1]) - float(ranges[name][0]))),
            min(float(ranges[name][1]), float(best[name]) + fraction * (float(ranges[name][1]) - float(ranges[name][0]))),
        )
        for name in active
    }
    count = 4 if quick else int(spec["search"]["joint_refinement_samples"])
    candidates = [_candidate_from_parameters(variant, "joint_center", best)]
    for point in halton_points(count, len(active), seed=int(spec["seed"]) + 313):
        values = {name: local[name][0] + coordinate * (local[name][1] - local[name][0]) for name, coordinate in zip(active, point)}
        candidates.append(_candidate_from_parameters(variant, "joint_halton", values))
    return list({candidate.key: candidate for candidate in candidates}.values())


def _boundary_axes(
    row: Mapping[str, Any], variant: str, ranges: Mapping[str, Sequence[float]]
) -> list[tuple[str, str]]:
    axes: list[tuple[str, str]] = []
    for name in _active_parameters(variant):
        value = float(row[name])
        if math.isclose(value, float(ranges[name][0]), rel_tol=0.0, abs_tol=1e-12):
            axes.append((name, "lower"))
        if math.isclose(value, float(ranges[name][1]), rel_tol=0.0, abs_tol=1e-12):
            axes.append((name, "upper"))
    return axes


def boundary_extension_candidates(
    variant: str,
    best: Mapping[str, Any],
    ranges: dict[str, list[float]],
    axes: Sequence[tuple[str, str]],
    spec: Mapping[str, Any],
    round_index: int,
    *,
    quick: bool = False,
) -> tuple[list[VariantCandidate], list[dict[str, Any]]]:
    active = _active_parameters(variant)
    fraction = float(spec["search"]["boundary_extension_fraction"])
    count = 4 if quick else int(spec["search"]["boundary_extension_samples"])
    candidates: list[VariantCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for axis_index, (name, side) in enumerate(axes):
        old_low, old_high = map(float, ranges[name])
        span = old_high - old_low
        if side == "lower" and name == "moving_mass_assist_gain_m_per_Nm" and old_low <= 0.0:
            diagnostics.append(
                {
                    "variant": variant,
                    "round": round_index,
                    "axis": name,
                    "side": side,
                    "action": "physical_lower_bound_retained",
                    "old_min": old_low,
                    "old_max": old_high,
                    "new_min": old_low,
                    "new_max": old_high,
                }
            )
            continue
        if side == "lower":
            ranges[name][0] = old_low - fraction * span
            slab = (ranges[name][0], old_low)
        else:
            ranges[name][1] = old_high + fraction * span
            slab = (old_high, ranges[name][1])
        diagnostics.append(
            {
                "variant": variant,
                "round": round_index,
                "axis": name,
                "side": side,
                "action": "extended",
                "old_min": old_low,
                "old_max": old_high,
                "new_min": ranges[name][0],
                "new_max": ranges[name][1],
            }
        )
        boundary_values = {parameter: float(best[parameter]) for parameter in active}
        boundary_values[name] = float(ranges[name][0] if side == "lower" else ranges[name][1])
        candidates.append(_candidate_from_parameters(variant, f"boundary_{round_index}", boundary_values))
        points = halton_points(count, len(active) - 1, seed=int(spec["seed"]) + 701 + round_index * 31 + axis_index)
        other_axes = [parameter for parameter in active if parameter != name]
        for point in points:
            values = {parameter: float(best[parameter]) for parameter in active}
            values[name] = slab[0] + 0.5 * (slab[1] - slab[0])
            for parameter, coordinate in zip(other_axes, point):
                low, high = map(float, ranges[parameter])
                half = 0.05 * (high - low)
                values[parameter] = min(high, max(low, float(best[parameter]) + (2.0 * coordinate - 1.0) * half))
            candidates.append(_candidate_from_parameters(variant, f"boundary_{round_index}", values))
    return list({candidate.key: candidate for candidate in candidates}.values()), diagnostics


def _source_fingerprint(spec: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    source_paths = (
        Path(__file__),
        REPO_ROOT / "analysis" / "headless_loiter.py",
        REPO_ROOT / "analysis" / "pitch_damping_retune.py",
        REPO_ROOT / "analysis" / "moving_mass_gain_resweep.py",
        REPO_ROOT / str(spec["source_profile"]),
    )
    payload = {
        "schema_version": 1,
        "spec": spec,
        "sources": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
    }
    return payload, _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _selected_candidate(row: Mapping[str, Any]) -> VariantCandidate:
    return _candidate_from_parameters(str(row["variant"]), "selected", row)


def _profile_payload(candidate: VariantCandidate, spec: Mapping[str, Any], selected_row: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / str(spec["source_profile"])
    profile = json.loads(source.read_text(encoding="utf-8"))
    controller = profile.setdefault("controller", {})
    controller.update({name: value for name, value in candidate.parameters.items() if name != "moving_mass_assist_gain_m_per_Nm"})
    controller["atc_rat_pit_i"] = 0.0
    analysis = profile.setdefault("analysis", {})
    analysis.update(
        {
            "profile_status": "provisional",
            "profile_notes": "Deterministic 2D simulation-only independently optimized controller; no hardware or real-flight claim.",
            "variant_controller_optimization": {
                "variant": candidate.variant,
                "selection_method": "hard gates then lexicographic ranking",
                "selected_candidate": dict(selected_row),
                "moving_mass_assist_gain_m_per_Nm": candidate.moving_mass_assist_gain_m_per_Nm,
            },
        }
    )
    return profile


def _validation_definition(item: ScenarioDefinition, candidate: VariantCandidate) -> ScenarioDefinition:
    return replace(
        item,
        config=replace(
            item.config,
            moving_mass_enabled=True,
            moving_mass_target_m=0.0,
            moving_mass_assist_gain_m_per_Nm=candidate.moving_mass_assist_gain_m_per_Nm,
        ),
    )


def _pattern_scenarios() -> Iterable[tuple[str, int, LoiterScenarioConfig]]:
    for pattern in PATTERNS:
        for direction in (1, -1) if pattern.mirror else (1,):
            timeline = tuple((start, end, direction * command) for start, end, command in pattern.timeline)
            yield pattern.key, direction, LoiterScenarioConfig(
                name=f"{pattern.key}_{'positive' if direction > 0 else 'negative'}",
                duration_s=10.0,
                initial_z=1.0,
                target_z=1.0,
                capture_current_target=True,
                stick_timeline=timeline,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                notes="selected-controller final LOITER regression",
            )


def _run_selected_validation(
    selected: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rerun_digests: list[str] = []
    first_rows: list[dict[str, Any]] = []
    representative: dict[str, LoiterRunResult] = {}
    all_passed = True
    for rerun in range(2):
        rows: list[dict[str, Any]] = []
        for variant in VARIANTS:
            candidate = _selected_candidate(selected[variant])
            for definition in required_scenarios(False):
                effective = _validation_definition(definition, candidate)
                metrics, result = evaluate_scenario(candidate, effective, spec, keep_result=True)
                assert result is not None
                failures = str(metrics.get("rejection_reasons", ""))
                row = {
                    "suite": "existing_seven",
                    "scenario": definition.key,
                    "variant": variant,
                    "passed": not bool(failures),
                    "failures": failures,
                    **candidate.parameters,
                    **metrics,
                }
                rows.append(row)
                if rerun == 0:
                    _save_timeseries_atomic(result, output_dir / "scenario_time_series" / f"seven__{definition.key}__{variant}.csv")
                    if definition.key == "forward_1m":
                        representative[f"forward_1m__{variant}"] = result
            for pattern, direction, scenario in _pattern_scenarios():
                effective = replace(
                    scenario,
                    moving_mass_assist_gain_m_per_Nm=candidate.moving_mass_assist_gain_m_per_Nm,
                )
                result = run_headless_loiter(
                    REPO_ROOT / str(spec["source_profile"]),
                    effective,
                    controller_overrides=candidate.controller_overrides(spec),
                )
                metrics, events, failures = audit_loiter_pattern(result)
                metrics.update(_moving_mass_metrics(result))
                failures = list(failures)
                failures.extend(_candidate_parameter_reasons(candidate, {**metrics, **result.metrics}))
                if candidate.variant == "vane_only":
                    acceleration = np.diff(_array(result.rows, "moving_mass_velocity_m_s"), prepend=0.0)
                    if any(
                        float(np.max(np.abs(_array(result.rows, field)))) != 0.0
                        for field in ("moving_mass_target_m", "moving_mass_offset_m", "moving_mass_velocity_m_s")
                    ) or float(np.max(np.abs(acceleration))) != 0.0:
                        failures.append("vane_only_not_exactly_locked")
                row = {
                    "suite": "final_32_loiter",
                    "scenario": pattern,
                    "direction": "positive" if direction > 0 else "negative",
                    "variant": variant,
                    "passed": not failures,
                    "failures": "; ".join(dict.fromkeys(failures)),
                    "event_count": len(events),
                    **candidate.parameters,
                    **metrics,
                }
                rows.append(row)
                if rerun == 0:
                    _save_timeseries_atomic(
                        result,
                        output_dir / "scenario_time_series" / f"loiter__{pattern}__{direction:+d}__{variant}.csv",
                    )
                    if pattern == "commanded_reversal" and direction > 0:
                        representative[f"commanded_reversal__{variant}"] = result
        digest_rows = [{key: value for key, value in row.items() if key not in {"runtime_s"}} for row in rows]
        rerun_digests.append(_sha256_bytes(_canonical_json(digest_rows).encode("utf-8")))
        if rerun == 0:
            first_rows = rows
            all_passed = all(bool(row["passed"]) for row in rows)
    deterministic = len(set(rerun_digests)) == 1
    atomic_csv(output_dir / "validation" / "selected_scenario_results.csv", first_rows)
    atomic_json(
        output_dir / "validation" / "deterministic_reruns.json",
        {"passed": deterministic and all_passed, "digests": rerun_digests, "scenario_count_per_rerun": len(first_rows)},
    )
    if not deterministic:
        raise RuntimeError("selected-controller validation reruns were not deterministic")
    return {
        "passed": all_passed,
        "deterministic": deterministic,
        "digests": rerun_digests,
        "rows": first_rows,
        "representative": representative,
    }


def _plot_selected(output_dir: Path, validation: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def plot_results(path: Path, title: str, results: Sequence[tuple[str, LoiterRunResult]]) -> None:
        figure, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        for label, result in results:
            rows = result.rows
            t = _array(rows, "time")
            axes[0].plot(t, _array(rows, "x"), label=label)
            axes[1].plot(t, _array(rows, "vx"), label=label)
            axes[2].plot(t, np.rad2deg(_array(rows, "theta")), label=label)
            axes[3].plot(t, np.rad2deg(_array(rows, "vane_angle_cmd")), label=label)
            axes[4].plot(t, _array(rows, "moving_mass_offset_m"), label=label)
        for axis, ylabel in zip(axes, ("x (m)", "vx (m/s)", "pitch (deg)", "vane (deg)", "mass offset (m)")):
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes[-1].set_xlabel("time (s)")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)

    representative = validation["representative"]
    plot_results(
        plots / "representative_position_step.png",
        "Selected independent controllers: +1 m step",
        [(variant, representative[f"forward_1m__{variant}"]) for variant in VARIANTS],
    )
    passed_rows = [row for row in validation["rows"] if row["suite"] == "existing_seven"]
    worst = max(passed_rows, key=lambda row: float(row.get("tail_rms_position_error_m", 0.0)))
    worst_key = f"forward_1m__{worst['variant']}" if worst["scenario"] == "forward_1m" else None
    if worst_key is None or worst_key not in representative:
        worst_key = f"commanded_reversal__{worst['variant']}"
    plot_results(
        plots / "worst_case.png",
        f"Worst retained validation trace ({worst['variant']}: {worst['scenario']})",
        [(str(worst["variant"]), representative[worst_key])],
    )


def _comparison_rows(
    selected: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, Any],
    fingerprint: str,
) -> list[dict[str, Any]]:
    reference_candidates = (
        _candidate_from_parameters("vane_only", "pr24_shared_reference", {**PR24_CONTROLLER, "moving_mass_assist_gain_m_per_Nm": 0.0}),
        _candidate_from_parameters("moving_mass_assist", "pr24_shared_reference", {**PR24_CONTROLLER, "moving_mass_assist_gain_m_per_Nm": PR24_ASSIST_GAIN}),
    )
    rows = [evaluate_candidate(candidate, spec, fingerprint) for candidate in reference_candidates]
    rows.extend(dict(selected[variant]) for variant in VARIANTS)
    for row in rows:
        row["comparison_role"] = (
            "PR #24 shared-controller actuator isolation"
            if str(row["stage"]).startswith("pr24")
            else "independently optimized variant"
        )
    return rows


def _summary_markdown(
    selected: Mapping[str, Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    candidate_count: int,
) -> str:
    lines = [
        "# Independent variant-controller optimization",
        "",
        "This is deterministic 2D simulation research only. It makes no real-flight, Pixhawk, Raspberry Pi, HIL, or hardware-safety claim, and it does not modify rigid-body physics.",
        "",
        "## Result categories",
        "",
        "1. **PR #24 shared-controller actuator isolation:** the merged controller and gain 0 / 0.0415 actuator variants are preserved as read-only references.",
        "2. **Independently optimized Vane-only:** its PID/position gains were selected by this executable workflow; the physical 0.5 kg mass stayed present and its gain, target, offset, rate, and acceleration were exactly zero.",
        "3. **Independently optimized Moving-mass-assist:** its PID/position gains and assist gain were searched independently and are not forced to match Vane-only.",
        "",
        f"Candidates evaluated: **{candidate_count}**. Selected validation cases per rerun: **{len(validation['rows'])}**. Two-rerun deterministic validation: **{validation['deterministic']}**. All finalist gates passed: **{validation['passed']}**.",
        "",
        "## Selected parameters",
        "",
        "| variant | Rate P | Rate D | Angle P | Position P | Velocity P | mass gain (m/Nm) | settle (s) | rise (s) | overshoot |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        row = selected[variant]
        lines.append(
            f"| {variant} | {float(row['atc_rat_pit_p']):.8f} | {float(row['atc_rat_pit_d']):.8f} | {float(row['atc_ang_pit_p']):.6f} | {float(row['psc_ne_pos_p']):.8f} | {float(row['psc_ne_vel_p']):.8f} | {float(row['moving_mass_assist_gain_m_per_Nm']):.8f} | {float(row['worst_settling_time_s']):.4f} | {float(row['worst_rise_time_s']):.4f} | {float(row['mean_overshoot_fraction']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Selection was lexicographic: all hard gates, settled status, settling time, rise time, preferred-band overshoot distance, steady-state error, actuator effort/limiter use, then mirrored/scenario robustness. Zero overshoot received a distance penalty whenever it fell below the preferred band; it was not treated as automatically optimal.",
            "",
            "The Pareto CSV retains valid tradeoffs rather than collapsing them into one weighted score. The shared-objective comparison CSV evaluates the PR #24 shared references and both independently optimized controllers with the same screening objectives.",
            "",
        ]
    )
    return "\n".join(lines)


def _public_export_hygiene(output_dir: Path) -> dict[str, Any]:
    forbidden = ("C:\\Users\\", "sk-proj-", "github_pat_", "ghp_")
    findings: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden:
            if token in text:
                findings.append({"file": path.relative_to(output_dir).as_posix(), "token": token})
    return {"passed": not findings, "findings": findings, "files_scanned": sum(1 for path in output_dir.rglob("*") if path.is_file())}


def verify_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest_path = Path(output_dir) / "sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["artifacts"].items():
        path = Path(output_dir) / relative
        observed = sha256_file(path) if path.exists() else "missing"
        if observed != expected["sha256"]:
            failures.append({"file": relative, "expected": expected["sha256"], "observed": observed})
    return {"passed": not failures, "failures": failures, "artifact_count": len(manifest["artifacts"])}


def _write_manifest(output_dir: Path) -> None:
    excluded = {"sha256_manifest.json", "validation/artifact_hash_verification.json"}
    artifacts = {
        path.relative_to(output_dir).as_posix(): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.relative_to(output_dir).as_posix() not in excluded
    }
    atomic_json(output_dir / "sha256_manifest.json", {"schema_version": 1, "algorithm": "SHA-256", "artifacts": artifacts})


def run_variant_search(
    variant: str,
    spec: dict[str, Any],
    fingerprint: str,
    cache: CandidateCache,
    *,
    workers: int,
    quick: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    all_rows: dict[str, dict[str, Any]] = {}
    coarse = coarse_candidates(variant, spec, quick=quick)
    for row in evaluate_candidates(coarse, spec, fingerprint, cache, workers=workers):
        all_rows[str(row["candidate_key"])] = row
    ranked = rank_candidates(all_rows.values())
    valid = [row for row in ranked if bool(row["all_hard_gates_pass"])]
    if not valid:
        raise RuntimeError(f"no valid {variant} candidate after coarse search")
    local = local_refinement_candidates(variant, valid, spec, quick=quick)
    for row in evaluate_candidates(local, spec, fingerprint, cache, workers=workers):
        all_rows[str(row["candidate_key"])] = row
    ranked = rank_candidates(all_rows.values())
    best = next(row for row in ranked if bool(row["all_hard_gates_pass"]))
    joint = joint_refinement_candidates(variant, best, spec, quick=quick)
    for row in evaluate_candidates(joint, spec, fingerprint, cache, workers=workers):
        all_rows[str(row["candidate_key"])] = row
    ranges = {name: list(map(float, bounds)) for name, bounds in spec["ranges"].items()}
    diagnostics: list[dict[str, Any]] = []
    maximum_rounds = 1 if quick else int(spec["search"]["maximum_boundary_extension_rounds"])
    for round_index in range(1, maximum_rounds + 1):
        ranked = rank_candidates(all_rows.values())
        best = next(row for row in ranked if bool(row["all_hard_gates_pass"]))
        axes = _boundary_axes(best, variant, ranges)
        if not axes:
            diagnostics.append({"variant": variant, "round": round_index, "action": "selected_interior"})
            break
        extension, round_diagnostics = boundary_extension_candidates(
            variant, best, ranges, axes, spec, round_index, quick=quick
        )
        diagnostics.extend(round_diagnostics)
        if not extension:
            break
        extended_spec = json.loads(json.dumps(spec))
        extended_spec["ranges"].update(ranges)
        for row in evaluate_candidates(extension, extended_spec, fingerprint, cache, workers=workers):
            all_rows[str(row["candidate_key"])] = row
    ranked = rank_candidates(all_rows.values())
    selected = next(row for row in ranked if bool(row["all_hard_gates_pass"]))
    return selected, ranked, diagnostics, ranges


def run_workflow(
    *,
    variant: str = "both",
    workers: int = 1,
    resume: bool = True,
    spec_path: Path = DEFAULT_SPEC,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    quick: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if variant not in {*VARIANTS, "both"}:
        raise ValueError(f"invalid variant: {variant}")
    workers = max(1, int(workers))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_search_spec(spec_path)
    fingerprint_payload, fingerprint = _source_fingerprint(spec)
    preserved_before = {**_tree_hashes(PR24_RESULTS), PR24_PROFILE.relative_to(REPO_ROOT).as_posix(): sha256_file(PR24_PROFILE)}
    atomic_json(output_dir / "search_space.json", spec)
    cache = CandidateCache(output_dir / "candidate_cache.json", fingerprint, resume=resume)
    requested = VARIANTS if variant == "both" else (variant,)
    selected: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    actual_ranges: dict[str, Any] = {}
    for name in requested:
        chosen, ranked, diagnostics, ranges = run_variant_search(
            name,
            spec,
            fingerprint,
            cache,
            workers=workers,
            quick=quick,
        )
        selected[name] = chosen
        all_rows.extend(ranked)
        boundary_rows.extend(diagnostics)
        actual_ranges[name] = ranges
        atomic_json(output_dir / f"{name}_selected.json", chosen)
        atomic_json(output_dir / "profiles" / f"{name}.json", _profile_payload(_selected_candidate(chosen), spec, chosen))
    all_rows = rank_candidates(all_rows)
    atomic_csv(output_dir / "candidate_results.csv", all_rows)
    atomic_csv(output_dir / "valid_candidates.csv", [row for row in all_rows if bool(row["all_hard_gates_pass"])])
    atomic_csv(output_dir / "rejected_candidates.csv", [row for row in all_rows if not bool(row["all_hard_gates_pass"])])
    atomic_csv(output_dir / "pareto_front.csv", pareto_front(all_rows))
    atomic_csv(output_dir / "boundary_diagnostics.csv", boundary_rows)
    atomic_json(output_dir / "actual_search_ranges.json", actual_ranges)
    validation: dict[str, Any] = {"passed": False, "deterministic": False, "rows": [], "representative": {}}
    comparison: list[dict[str, Any]] = []
    if set(requested) == set(VARIANTS):
        if selected["vane_only"]["candidate_key"] == selected["moving_mass_assist"]["candidate_key"]:
            raise RuntimeError("variant selections unexpectedly share an identical candidate key")
        validation = _run_selected_validation(selected, spec, output_dir)
        comparison = _comparison_rows(selected, spec, fingerprint)
        atomic_csv(output_dir / "shared-objective comparison.csv", comparison)
        _plot_selected(output_dir, validation)
    preserved_after = {**_tree_hashes(PR24_RESULTS), PR24_PROFILE.relative_to(REPO_ROOT).as_posix(): sha256_file(PR24_PROFILE)}
    preservation = {
        "passed": preserved_before == preserved_after,
        "before": preserved_before,
        "after": preserved_after,
        "changed": sorted(key for key in set(preserved_before) | set(preserved_after) if preserved_before.get(key) != preserved_after.get(key)),
    }
    atomic_json(output_dir / "validation" / "preservation_audit.json", preservation)
    summary = _summary_markdown(selected, comparison, validation, len(all_rows))
    atomic_text(output_dir / "summary.md", summary)
    atomic_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "workflow_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "variant": variant,
            "workers": workers,
            "resume": resume,
            "quick": quick,
            "candidate_count": len(all_rows),
            "selected": selected,
            "validation_passed": validation["passed"],
            "runtime_s": time.perf_counter() - started,
            "limitations": [
                "Deterministic 2D analytical simulation only.",
                "No real-flight, Pixhawk, Raspberry Pi, HIL, or hardware-safety validation.",
            ],
        },
    )
    hygiene = _public_export_hygiene(output_dir)
    atomic_json(output_dir / "validation" / "public_export_hygiene.json", hygiene)
    _write_manifest(output_dir)
    manifest_verification = verify_manifest(output_dir)
    atomic_json(output_dir / "validation" / "artifact_hash_verification.json", manifest_verification)
    if not preservation["passed"]:
        raise RuntimeError("PR #24 preservation audit failed")
    if not hygiene["passed"]:
        raise RuntimeError("public-export hygiene failed")
    if set(requested) == set(VARIANTS) and not validation["passed"]:
        raise RuntimeError("selected-controller validation failed one or more hard gates")
    return {
        "selected": selected,
        "candidate_count": len(all_rows),
        "validation_passed": validation["passed"],
        "deterministic": validation["deterministic"],
        "output_dir": str(output_dir),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("both", *VARIANTS), default="both")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true", help="reuse the fingerprinted atomic candidate cache")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument("--search-space", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="small deterministic development search")
    parser.add_argument("--verify-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_manifest:
        result = verify_manifest(args.output_dir)
    else:
        result = run_workflow(
            variant=args.variant,
            workers=args.workers,
            resume=args.resume,
            spec_path=args.search_space,
            output_dir=args.output_dir,
            quick=args.quick,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("validation_passed", result.get("passed", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
