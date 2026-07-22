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


class NoEligibleCandidateError(RuntimeError):
    def __init__(
        self,
        variant: str,
        ranked: Sequence[Mapping[str, Any]],
        diagnostics: Sequence[Mapping[str, Any]],
        ranges: Mapping[str, Sequence[float]],
        stats: Mapping[str, Any],
    ) -> None:
        super().__init__(
            f"{variant} is blocked: no candidate met <=8% overshoot and <=1 crossing after targeted refinement"
        )
        self.variant = variant
        self.ranked = [dict(row) for row in ranked]
        self.diagnostics = [dict(row) for row in diagnostics]
        self.ranges = {name: list(map(float, bounds)) for name, bounds in ranges.items()}
        self.stats = dict(stats)


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
    targets = spec["targets"]
    selection_overshoot = float(targets["selection_maximum_overshoot_fraction"])
    hard_overshoot = float(targets["hard_maximum_overshoot_fraction"])
    if not float(preferred[1]) <= selection_overshoot < hard_overshoot:
        raise ValueError("overshoot thresholds must satisfy preferred <= selection < hard")
    selection_crossings = int(targets["selection_maximum_target_crossings"])
    hard_crossings = int(targets["hard_maximum_target_crossings"])
    if not 0 <= selection_crossings <= hard_crossings:
        raise ValueError("target-crossing thresholds must satisfy selection <= hard")
    search = spec["search"]
    if int(search["targeted_parent_count"]) < 12:
        raise ValueError("targeted_parent_count must be at least 12")
    if int(search["targeted_adaptive_samples"]) < 160:
        raise ValueError("targeted_adaptive_samples must be at least 160")
    if int(search["targeted_joint_refinement_samples"]) < 96:
        raise ValueError("targeted_joint_refinement_samples must be at least 96")
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
        "selection_overshoot_exceeded": overshoot_fraction
        > float(targets["selection_maximum_overshoot_fraction"]),
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
        if int(metrics.get("target_crossing_count", 10**9)) > int(targets["hard_maximum_target_crossings"]):
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
    band_distances = [float(row["overshoot_band_distance"]) for row in steps]
    band_distance = float(np.mean(band_distances))
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
    row = {
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
        "worst_overshoot_band_distance": max(band_distances, default=1000.0),
        "mean_overshoot_fraction": float(np.mean([float(row["overshoot_fraction"]) for row in steps])),
        "worst_final_abs_position_error_m": final_error,
        "actuator_effort_index": effort,
        "robustness_index": robustness,
        "worst_asymmetry_fraction": worst_asymmetry,
        "scenario_metrics_json": _canonical_json(by_name),
        "symmetry_json": _canonical_json(symmetry),
    }
    return apply_corrected_selection_fields(row, spec)


def _scenario_metrics(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    payload = row.get("scenario_metrics_json", {})
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, Mapping)}


def apply_corrected_selection_fields(
    row: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive corrected policy fields from cached, simulation-produced metrics."""
    result = dict(row)
    scenarios = _scenario_metrics(result)
    steps = {name: metrics for name, metrics in scenarios.items() if bool(metrics.get("is_position_step"))}
    positive = steps.get("forward_1m")
    negative = steps.get("backward_1m")
    ordered_steps = list(steps.values())
    if positive is None and ordered_steps:
        positive = ordered_steps[0]
    if negative is None and len(ordered_steps) > 1:
        negative = ordered_steps[1]

    def directional(metric: str, item: Mapping[str, Any] | None, fallback: Any) -> Any:
        return item.get(metric, fallback) if item is not None else fallback

    positive_overshoot = _finite_or_penalty(
        directional("overshoot_fraction", positive, result.get("positive_step_overshoot_fraction")),
        1000.0,
    )
    negative_overshoot = _finite_or_penalty(
        directional("overshoot_fraction", negative, result.get("negative_step_overshoot_fraction")),
        1000.0,
    )
    positive_crossings = int(
        directional("target_crossing_count", positive, result.get("positive_step_target_crossing_count", 10**9))
    )
    negative_crossings = int(
        directional("target_crossing_count", negative, result.get("negative_step_target_crossing_count", 10**9))
    )
    preferred = spec["targets"]["preferred_overshoot_fraction"]
    result.update(
        {
            "positive_step_overshoot_fraction": positive_overshoot,
            "negative_step_overshoot_fraction": negative_overshoot,
            "worst_step_overshoot_fraction": max(positive_overshoot, negative_overshoot),
            "positive_step_target_crossing_count": positive_crossings,
            "negative_step_target_crossing_count": negative_crossings,
            "worst_step_target_crossing_count": max(positive_crossings, negative_crossings),
            "positive_step_settling_time_s": _finite_or_penalty(
                directional("settling_time_s", positive, result.get("positive_step_settling_time_s")), 1000.0
            ),
            "negative_step_settling_time_s": _finite_or_penalty(
                directional("settling_time_s", negative, result.get("negative_step_settling_time_s")), 1000.0
            ),
            "positive_step_rise_time_s": _finite_or_penalty(
                directional("rise_time_s", positive, result.get("positive_step_rise_time_s")), 1000.0
            ),
            "negative_step_rise_time_s": _finite_or_penalty(
                directional("rise_time_s", negative, result.get("negative_step_rise_time_s")), 1000.0
            ),
            "positive_step_final_abs_position_error_m": _finite_or_penalty(
                directional(
                    "final_abs_position_error_m",
                    positive,
                    result.get("positive_step_final_abs_position_error_m"),
                ),
                1000.0,
            ),
            "negative_step_final_abs_position_error_m": _finite_or_penalty(
                directional(
                    "final_abs_position_error_m",
                    negative,
                    result.get("negative_step_final_abs_position_error_m"),
                ),
                1000.0,
            ),
            "worst_overshoot_band_distance": max(
                overshoot_band_distance(positive_overshoot, preferred),
                overshoot_band_distance(negative_overshoot, preferred),
            ),
        }
    )
    selection_overshoot = float(spec["targets"]["selection_maximum_overshoot_fraction"])
    selection_crossings = int(spec["targets"]["selection_maximum_target_crossings"])
    overshoot_eligible = (
        positive_overshoot <= selection_overshoot + 1e-12
        and negative_overshoot <= selection_overshoot + 1e-12
    )
    crossing_eligible = positive_crossings <= selection_crossings and negative_crossings <= selection_crossings
    result["selection_overshoot_eligible"] = overshoot_eligible
    result["selection_crossing_eligible"] = crossing_eligible
    if not bool(result.get("all_hard_gates_pass", False)):
        selection_class = "hard_gate_rejected"
    elif not bool(result.get("all_step_scenarios_settled", False)):
        selection_class = "unsettled"
    elif not overshoot_eligible:
        selection_class = "overshoot_ineligible"
    elif not crossing_eligible:
        selection_class = "crossing_ineligible"
    else:
        selection_class = "eligible"
    result["corrected_selection_class"] = selection_class
    result["selection_eligible"] = selection_class == "eligible"

    if scenarios:
        result["worst_vane_saturation_percent"] = max(
            (_finite_or_penalty(item.get("vane_saturation_percent"), 1000.0) for item in scenarios.values()),
            default=1000.0,
        )
        result["worst_servo_rate_saturation_percent"] = max(
            (_finite_or_penalty(item.get("servo_rate_saturation_percent"), 1000.0) for item in scenarios.values()),
            default=1000.0,
        )
        result["worst_mixer_saturation_percent"] = max(
            (_finite_or_penalty(item.get("mixer_saturation_percent"), 1000.0) for item in scenarios.values()),
            default=1000.0,
        )
        limiter_metrics = [
            _finite_or_penalty(item.get(f"moving_mass_{name}_duty_percent"), 1000.0)
            for item in scenarios.values()
            for name in MOVING_MASS_LIMITER_HARD_GATES
        ]
        result["worst_moving_mass_limiter_duty_percent"] = max(limiter_metrics, default=0.0)
    return result


def candidate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Corrected selection order; speed cannot compensate for policy ineligibility."""
    return (
        not bool(row.get("all_hard_gates_pass", False)),
        not bool(row.get("all_step_scenarios_settled", False)),
        not bool(row.get("selection_overshoot_eligible", False)),
        not bool(row.get("selection_crossing_eligible", False)),
        _finite_or_penalty(row.get("worst_settling_time_s"), 1000.0),
        _finite_or_penalty(row.get("worst_overshoot_band_distance"), 1000.0),
        _finite_or_penalty(row.get("worst_rise_time_s"), 1000.0),
        _finite_or_penalty(row.get("worst_final_abs_position_error_m"), 1000.0),
        _finite_or_penalty(row.get("actuator_effort_index"), 1000.0),
        _finite_or_penalty(row.get("robustness_index"), 1000.0),
        str(row.get("candidate_key", "")),
    )


def rank_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted((dict(row) for row in rows), key=candidate_rank_key)
    for index, row in enumerate(ranked, 1):
        row["corrected_lexicographic_rank"] = index
    return ranked


def corrected_selection_result(
    rows: Iterable[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    ranked = rank_candidates(apply_corrected_selection_fields(row, spec) for row in rows)
    selected = next((row for row in ranked if bool(row.get("selection_eligible", False))), None)
    if selected is None:
        return {
            "status": "blocked",
            "reason": "no candidate satisfies all hard gates, settling, <=8% overshoot, and <=1 target crossing",
            "selected": None,
            "ranked": ranked,
        }
    return {"status": "selected", "reason": "corrected lexicographic selection", "selected": selected, "ranked": ranked}


PARETO_OBJECTIVES = (
    "worst_settling_time_s",
    "worst_rise_time_s",
    "worst_overshoot_band_distance",
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

    def __init__(
        self,
        path: Path,
        fingerprint: str,
        *,
        resume: bool,
        compatible_reuse: bool = False,
    ) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.rows: dict[str, dict[str, Any]] = {}
        self.compatible_reuse = False
        self.reused_candidate_count = 0
        self.newly_recorded_count = 0
        if resume and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            source_fingerprint = str(payload.get("workflow_fingerprint", ""))
            if source_fingerprint == fingerprint or compatible_reuse:
                self.compatible_reuse = source_fingerprint != fingerprint
                for original in payload.get("candidates", []):
                    row = dict(original)
                    if self.compatible_reuse:
                        row["source_workflow_fingerprint"] = source_fingerprint
                        row["workflow_fingerprint"] = fingerprint
                        row["simulation_metrics_reused"] = True
                    self.rows[str(row["candidate_key"])] = row
                if self.compatible_reuse:
                    self.reused_candidate_count = len(self.rows)
                    self.save()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.rows.get(key)
        return dict(row) if row is not None else None

    def record(self, row: Mapping[str, Any]) -> None:
        key = str(row["candidate_key"])
        if key not in self.rows:
            self.newly_recorded_count += 1
        stored = dict(row)
        stored["workflow_fingerprint"] = self.fingerprint
        stored.setdefault("simulation_metrics_reused", False)
        self.rows[key] = stored
        self.save()

    def save(self) -> None:
        atomic_json(
            self.path,
            {
                "schema_version": 2,
                "workflow_fingerprint": self.fingerprint,
                "candidates": [self.rows[key] for key in sorted(self.rows)],
            },
        )

    def replace_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self.rows = {str(row["candidate_key"]): dict(row) for row in rows}
        self.save()

    def all_rows(self, variant: str | None = None) -> list[dict[str, Any]]:
        rows = [dict(row) for row in self.rows.values()]
        if variant is not None:
            rows = [row for row in rows if str(row.get("variant")) == variant]
        return rows


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


def _targeted_parent_rows(
    rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    parent_count = int(spec["search"]["targeted_parent_count"])
    maximum = float(spec["targets"]["selection_maximum_overshoot_fraction"])
    feasible = [
        dict(row)
        for row in rows
        if bool(row.get("all_hard_gates_pass", False))
        and bool(row.get("all_step_scenarios_settled", False))
    ]
    pool = pareto_front(feasible) if feasible else []
    seen = {str(row["candidate_key"]) for row in pool}
    pool.extend(row for row in feasible if str(row["candidate_key"]) not in seen)
    return sorted(
        pool,
        key=lambda row: (
            _finite_or_penalty(row.get("worst_step_overshoot_fraction"), 1000.0) > maximum,
            abs(_finite_or_penalty(row.get("worst_step_overshoot_fraction"), 1000.0) - maximum),
            _finite_or_penalty(row.get("worst_settling_time_s"), 1000.0),
            _finite_or_penalty(row.get("worst_overshoot_band_distance"), 1000.0),
            _finite_or_penalty(row.get("worst_rise_time_s"), 1000.0),
            str(row.get("candidate_key", "")),
        ),
    )[:parent_count]


def targeted_refinement_candidates(
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    quick: bool = False,
) -> tuple[list[VariantCandidate], list[dict[str, Any]]]:
    parents = _targeted_parent_rows(rows, spec)
    if not parents:
        return [], []
    ranges = spec["ranges"]
    active = _active_parameters(variant)
    fraction = float(spec["search"]["targeted_refinement_fraction"])
    count = 12 if quick else int(spec["search"]["targeted_adaptive_samples"])
    variant_seed = 1009 if variant == "vane_only" else 2027
    points = halton_points(count * 3, len(active), seed=int(spec["seed"]) + variant_seed)
    candidates: dict[str, VariantCandidate] = {}
    for index, point in enumerate(points):
        parent = parents[index % len(parents)]
        values: dict[str, float] = {}
        for name, coordinate in zip(active, point):
            low, high = map(float, ranges[name])
            half = fraction * (high - low)
            values[name] = min(high, max(low, float(parent[name]) + (2.0 * coordinate - 1.0) * half))
        candidate = _candidate_from_parameters(variant, "targeted_adaptive", values)
        candidates[candidate.key] = candidate
        if len(candidates) >= count:
            break
    return list(candidates.values()), parents


def targeted_joint_refinement_candidates(
    variant: str,
    best: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    quick: bool = False,
) -> list[VariantCandidate]:
    ranges = spec["ranges"]
    active = _active_parameters(variant)
    fraction = float(spec["search"]["targeted_joint_refinement_fraction"])
    count = 8 if quick else int(spec["search"]["targeted_joint_refinement_samples"])
    variant_seed = 4099 if variant == "vane_only" else 5003
    points = halton_points(count * 3, len(active), seed=int(spec["seed"]) + variant_seed)
    candidates: dict[str, VariantCandidate] = {}
    for point in points:
        values = {}
        for name, coordinate in zip(active, point):
            low, high = map(float, ranges[name])
            half = fraction * (high - low)
            values[name] = min(high, max(low, float(best[name]) + (2.0 * coordinate - 1.0) * half))
        candidate = _candidate_from_parameters(variant, "targeted_joint", values)
        candidates[candidate.key] = candidate
        if len(candidates) >= count:
            break
    return list(candidates.values())


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


def _simulation_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    targets = spec["targets"]
    return {
        "seed": int(spec["seed"]),
        "source_profile": str(spec["source_profile"]),
        "fixed": dict(spec["fixed"]),
        "fast_screen_duration_s": float(spec["search"]["fast_screen_duration_s"]),
        "targets": {
            "preferred_overshoot_fraction": list(map(float, targets["preferred_overshoot_fraction"])),
            "hard_maximum_overshoot_fraction": float(targets["hard_maximum_overshoot_fraction"]),
            "position_settling_band_m": float(targets["position_settling_band_m"]),
            "velocity_settling_band_m_s": float(targets["velocity_settling_band_m_s"]),
            "required_continuous_settling_duration_s": float(
                targets["required_continuous_settling_duration_s"]
            ),
            "steady_state_position_error_m": float(targets["steady_state_position_error_m"]),
            "hard_maximum_target_crossings": int(
                targets.get("hard_maximum_target_crossings", targets.get("maximum_target_crossings", -1))
            ),
            "target_crossing_deadband_m": float(targets["target_crossing_deadband_m"]),
        },
    }


def _cache_metrics_compatible(
    previous_metadata: Mapping[str, Any] | None, spec: Mapping[str, Any]
) -> bool:
    """Allow policy-only reranking while rejecting changed traces or hard gates."""
    if not previous_metadata:
        return False
    payload = previous_metadata.get("fingerprint_payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("spec"), Mapping):
        return False
    if _simulation_contract(payload["spec"]) != _simulation_contract(spec):
        return False
    previous_sources = payload.get("sources")
    if not isinstance(previous_sources, Mapping):
        return False
    simulation_sources = (
        REPO_ROOT / "analysis" / "headless_loiter.py",
        REPO_ROOT / "analysis" / "pitch_damping_retune.py",
        REPO_ROOT / "analysis" / "moving_mass_gain_resweep.py",
        REPO_ROOT / str(spec["source_profile"]),
    )
    return all(
        previous_sources.get(path.relative_to(REPO_ROOT).as_posix()) == sha256_file(path)
        for path in simulation_sources
    )


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
                "selection_method": (
                    "hard gates, settling, <=8% step overshoot, <=1 meaningful target crossing, "
                    "then settling/band-distance/rise/error/effort/robustness lexicographic ranking"
                ),
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
                failure_list = [
                    reason
                    for reason in str(metrics.get("rejection_reasons", "")).split("; ")
                    if reason
                ]
                if bool(metrics.get("is_position_step", False)):
                    if float(metrics.get("overshoot_fraction", math.inf)) > float(
                        spec["targets"]["selection_maximum_overshoot_fraction"]
                    ) + 1e-12:
                        failure_list.append("selection_overshoot_limit")
                    if int(metrics.get("target_crossing_count", 10**9)) > int(
                        spec["targets"]["selection_maximum_target_crossings"]
                    ):
                        failure_list.append("selection_target_crossing_limit")
                failures = "; ".join(dict.fromkeys(failure_list))
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
    cache: CandidateCache,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    reference_candidates = (
        _candidate_from_parameters("vane_only", "pr24_shared_reference", {**PR24_CONTROLLER, "moving_mass_assist_gain_m_per_Nm": 0.0}),
        _candidate_from_parameters("moving_mass_assist", "pr24_shared_reference", {**PR24_CONTROLLER, "moving_mass_assist_gain_m_per_Nm": PR24_ASSIST_GAIN}),
    )
    rows = [
        apply_corrected_selection_fields(row, spec)
        for row in evaluate_candidates(
            reference_candidates, spec, fingerprint, cache, workers=workers
        )
    ]
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
    previous_selected: Mapping[str, Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    candidate_count: int,
    search_stats: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        "# Corrected independent variant-controller optimization",
        "",
        "This is deterministic 2D simulation research only. It makes no real-flight, Pixhawk, Raspberry Pi, HIL, or hardware-safety claim, and it does not modify rigid-body physics.",
        "",
        "## Corrected reranking outcome",
        "",
        "All cached simulation metrics were reranked before any new simulation. A candidate is selectable only when all prior hard gates pass, both steps settle, both overshoots are at most 8%, and both steps have at most one meaningful target crossing.",
        "",
        "| variant | cached candidates | existing eligible | newly evaluated | corrected result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for variant in VARIANTS:
        stats = search_stats[variant]
        lines.append(
            f"| {variant} | {int(stats['existing_candidate_count'])} | {int(stats['existing_eligible_candidate_count'])} | {int(stats['newly_evaluated_candidates'])} | {stats['selection_status']} |"
        )
    lines.extend(
        [
            "",
            f"Total candidate rows: **{candidate_count}**. Selected validation cases per rerun: **{len(validation['rows'])}**. Two-rerun deterministic validation: **{validation['deterministic']}**. All finalist gates passed: **{validation['passed']}**.",
            "",
            "## Historical previous selections",
            "",
            "The prior 11-12% results are retained below as historical output and are not final under the corrected objective.",
            "",
            "| variant | previous candidate | worst overshoot | crossings (+/-) | corrected class |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for variant in VARIANTS:
        row = previous_selected.get(variant)
        if row is None:
            lines.append(f"| {variant} | not available | - | - | - |")
        else:
            lines.append(
                f"| {variant} | `{row['candidate_key']}` | {float(row['worst_step_overshoot_fraction']):.6f} | {int(row['positive_step_target_crossing_count'])}/{int(row['negative_step_target_crossing_count'])} | {row['corrected_selection_class']} |"
            )
    lines.extend(
        [
            "",
            "## Final selected controllers",
            "",
            "| variant | Rate P | Rate D | Angle P | Position P | Velocity P | mass gain (m/Nm) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in VARIANTS:
        row = selected[variant]
        lines.append(
            f"| {variant} | {float(row['atc_rat_pit_p']):.8f} | {float(row['atc_rat_pit_d']):.8f} | {float(row['atc_ang_pit_p']):.6f} | {float(row['psc_ne_pos_p']):.8f} | {float(row['psc_ne_vel_p']):.8f} | {float(row['moving_mass_assist_gain_m_per_Nm']):.8f} |"
        )
    lines.extend(
        [
            "",
            "| variant | settle + / - (s) | rise + / - (s) | worst overshoot | crossings + / - | worst final error (m) | effort index | vane / servo / mixer sat. (%) | mass limiter max (%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in VARIANTS:
        row = selected[variant]
        lines.append(
            f"| {variant} | {float(row['positive_step_settling_time_s']):.4f} / {float(row['negative_step_settling_time_s']):.4f} | {float(row['positive_step_rise_time_s']):.4f} / {float(row['negative_step_rise_time_s']):.4f} | {float(row['worst_step_overshoot_fraction']):.6f} | {int(row['positive_step_target_crossing_count'])} / {int(row['negative_step_target_crossing_count'])} | {float(row['worst_final_abs_position_error_m']):.8f} | {float(row['actuator_effort_index']):.6f} | {float(row['worst_vane_saturation_percent']):.3f} / {float(row['worst_servo_rate_saturation_percent']):.3f} / {float(row['worst_mixer_saturation_percent']):.3f} | {float(row['worst_moving_mass_limiter_duty_percent']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Exact selection rationale",
            "",
            "The executable optimizer uses this strict lexicographic order: (1) every pre-existing hard gate, (2) both steps settle, (3) both step overshoots <= 0.08, (4) both target-crossing counts <= 1, (5) shortest worst settling time, (6) smallest worst distance from the 0.02-0.05 preferred overshoot band, (7) shortest worst rise time, (8) smallest worst final error, (9) lower actuator effort and limiter use, and (10) better symmetry and robustness. Zero overshoot remains eligible but has a 0.02 band-distance penalty. The 0.12 overshoot hard failure and pre-existing two-crossing hard failure remain unchanged.",
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
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[float]],
    dict[str, Any],
]:
    existing_rows = [apply_corrected_selection_fields(row, spec) for row in cache.all_rows(variant)]
    all_rows = {str(row["candidate_key"]): row for row in existing_rows}
    existing_result = corrected_selection_result(existing_rows, spec)
    existing_eligible_count = sum(bool(row.get("selection_eligible")) for row in existing_result["ranked"])
    start_new_count = cache.newly_recorded_count
    ranges = {name: list(map(float, bounds)) for name, bounds in spec["ranges"].items()}
    diagnostics: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "variant": variant,
        "existing_candidate_count": len(existing_rows),
        "existing_eligible_candidate_count": existing_eligible_count,
        "newly_evaluated_candidates": 0,
        "targeted_parent_count": 0,
        "targeted_adaptive_candidates": 0,
        "targeted_joint_candidates": 0,
        "selection_status": existing_result["status"],
    }
    if existing_result["selected"] is not None:
        diagnostics.append(
            {
                "variant": variant,
                "round": 0,
                "action": "corrected_rerank_existing_metrics",
                "existing_candidates": len(existing_rows),
                "eligible_candidates": existing_eligible_count,
            }
        )
        return existing_result["selected"], existing_result["ranked"], diagnostics, ranges, stats

    # A fresh output directory still needs the ordinary broad search before
    # near-boundary refinement can choose meaningful parents.
    if not existing_rows:
        for row in evaluate_candidates(
            coarse_candidates(variant, spec, quick=quick), spec, fingerprint, cache, workers=workers
        ):
            all_rows[str(row["candidate_key"])] = apply_corrected_selection_fields(row, spec)
        ranked = rank_candidates(all_rows.values())
        hard_pass = [row for row in ranked if bool(row.get("all_hard_gates_pass"))]
        if hard_pass:
            for row in evaluate_candidates(
                local_refinement_candidates(variant, hard_pass, spec, quick=quick),
                spec,
                fingerprint,
                cache,
                workers=workers,
            ):
                all_rows[str(row["candidate_key"])] = apply_corrected_selection_fields(row, spec)
            ranked = rank_candidates(all_rows.values())
            hard_pass = [row for row in ranked if bool(row.get("all_hard_gates_pass"))]
        if hard_pass:
            for row in evaluate_candidates(
                joint_refinement_candidates(variant, hard_pass[0], spec, quick=quick),
                spec,
                fingerprint,
                cache,
                workers=workers,
            ):
                all_rows[str(row["candidate_key"])] = apply_corrected_selection_fields(row, spec)

    ranked = rank_candidates(all_rows.values())
    targeted, parents = targeted_refinement_candidates(variant, ranked, spec, quick=quick)
    stats["targeted_parent_count"] = len(parents)
    stats["targeted_adaptive_candidates"] = len(targeted)
    diagnostics.append(
        {
            "variant": variant,
            "round": 0,
            "action": "targeted_adaptive_refinement",
            "parent_count": len(parents),
            "candidate_count": len(targeted),
        }
    )
    for row in evaluate_candidates(targeted, spec, fingerprint, cache, workers=workers):
        all_rows[str(row["candidate_key"])] = apply_corrected_selection_fields(row, spec)

    ranked = rank_candidates(all_rows.values())
    refined_parents = _targeted_parent_rows(ranked, spec)
    if refined_parents:
        joint = targeted_joint_refinement_candidates(variant, refined_parents[0], spec, quick=quick)
    else:
        joint = []
    stats["targeted_joint_candidates"] = len(joint)
    diagnostics.append(
        {
            "variant": variant,
            "round": 0,
            "action": "targeted_joint_refinement",
            "candidate_count": len(joint),
        }
    )
    for row in evaluate_candidates(joint, spec, fingerprint, cache, workers=workers):
        all_rows[str(row["candidate_key"])] = apply_corrected_selection_fields(row, spec)

    maximum_rounds = 1 if quick else int(spec["search"]["maximum_boundary_extension_rounds"])
    for round_index in range(1, maximum_rounds + 1):
        result = corrected_selection_result(all_rows.values(), spec)
        selected = result["selected"]
        if selected is None:
            break
        axes = _boundary_axes(selected, variant, ranges)
        if not axes:
            diagnostics.append({"variant": variant, "round": round_index, "action": "selected_interior"})
            break
        extension, round_diagnostics = boundary_extension_candidates(
            variant, selected, ranges, axes, spec, round_index, quick=quick
        )
        diagnostics.extend(round_diagnostics)
        if not extension:
            break
        extended_spec = json.loads(json.dumps(spec))
        extended_spec["ranges"].update(ranges)
        for row in evaluate_candidates(extension, extended_spec, fingerprint, cache, workers=workers):
            all_rows[str(row["candidate_key"])] = apply_corrected_selection_fields(row, spec)

    final_result = corrected_selection_result(all_rows.values(), spec)
    stats["newly_evaluated_candidates"] = cache.newly_recorded_count - start_new_count
    stats["selection_status"] = final_result["status"]
    if final_result["selected"] is None:
        raise NoEligibleCandidateError(
            variant, final_result["ranked"], diagnostics, ranges, stats
        )
    return final_result["selected"], final_result["ranked"], diagnostics, ranges, stats


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
    previous_metadata_path = output_dir / "run_metadata.json"
    previous_metadata = (
        json.loads(previous_metadata_path.read_text(encoding="utf-8"))
        if previous_metadata_path.exists()
        else None
    )
    previous_selected_raw = {
        name: json.loads((output_dir / f"{name}_selected.json").read_text(encoding="utf-8"))
        for name in VARIANTS
        if (output_dir / f"{name}_selected.json").exists()
    }
    historical_path = output_dir / "historical_previous_selections.json"
    existing_history = (
        json.loads(historical_path.read_text(encoding="utf-8"))
        if historical_path.exists()
        else None
    )
    if isinstance(existing_history, Mapping) and isinstance(existing_history.get("selections"), Mapping):
        previous_selected_raw = dict(existing_history["selections"])
    historical_source_fingerprint = (
        existing_history.get("source_workflow_fingerprint")
        if isinstance(existing_history, Mapping)
        else (previous_metadata.get("workflow_fingerprint") if previous_metadata else None)
    )
    previous_selected = {
        name: apply_corrected_selection_fields(row, spec)
        for name, row in previous_selected_raw.items()
    }
    fingerprint_payload, fingerprint = _source_fingerprint(spec)
    preserved_before = {**_tree_hashes(PR24_RESULTS), PR24_PROFILE.relative_to(REPO_ROOT).as_posix(): sha256_file(PR24_PROFILE)}
    atomic_json(output_dir / "search_space.json", spec)
    compatible_reuse = resume and _cache_metrics_compatible(previous_metadata, spec)
    cache = CandidateCache(
        output_dir / "candidate_cache.json",
        fingerprint,
        resume=resume,
        compatible_reuse=compatible_reuse,
    )
    if cache.rows:
        cache.replace_rows(apply_corrected_selection_fields(row, spec) for row in cache.all_rows())
    atomic_json(
        historical_path,
        {
            "schema_version": 1,
            "source_workflow_fingerprint": historical_source_fingerprint,
            "status": "historical_not_final_under_corrected_objective",
            "selections": previous_selected,
        },
    )
    requested = VARIANTS if variant == "both" else (variant,)
    selected: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    actual_ranges: dict[str, Any] = {}
    search_stats: dict[str, dict[str, Any]] = {}
    for name in requested:
        try:
            chosen, ranked, diagnostics, ranges, stats = run_variant_search(
                name,
                spec,
                fingerprint,
                cache,
                workers=workers,
                quick=quick,
            )
        except NoEligibleCandidateError as error:
            atomic_json(
                output_dir / "blocked_selection.json",
                {
                    "schema_version": 1,
                    "status": "blocked",
                    "variant": error.variant,
                    "reason": str(error),
                    "search_stats": error.stats,
                    "actual_search_ranges": error.ranges,
                },
            )
            atomic_csv(output_dir / f"{name}_candidate_results.csv", error.ranked)
            atomic_csv(output_dir / "boundary_diagnostics.csv", error.diagnostics)
            cache.replace_rows(
                apply_corrected_selection_fields(row, spec) for row in cache.all_rows()
            )
            raise
        selected[name] = chosen
        all_rows.extend(ranked)
        boundary_rows.extend(diagnostics)
        actual_ranges[name] = ranges
        search_stats[name] = stats
        atomic_json(output_dir / f"{name}_selected.json", chosen)
        atomic_json(output_dir / "profiles" / f"{name}.json", _profile_payload(_selected_candidate(chosen), spec, chosen))
    all_rows = sorted(
        all_rows,
        key=lambda row: (str(row["variant"]), int(row["corrected_lexicographic_rank"])),
    )
    cache.replace_rows(
        apply_corrected_selection_fields(row, spec) for row in cache.all_rows()
    )
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
        comparison_new_start = cache.newly_recorded_count
        comparison = _comparison_rows(
            selected, spec, fingerprint, cache, workers=workers
        )
        comparison_new = cache.newly_recorded_count - comparison_new_start
        if comparison_new:
            search_stats["comparison_reference"] = {
                "newly_evaluated_candidates": comparison_new
            }
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
    summary = _summary_markdown(
        selected,
        previous_selected,
        comparison,
        validation,
        len(all_rows),
        search_stats,
    )
    atomic_text(output_dir / "summary.md", summary)
    atomic_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": 2,
            "workflow_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "variant": variant,
            "workers": workers,
            "resume": resume,
            "quick": quick,
            "candidate_count": len(all_rows),
            "cache_policy_only_compatible_reuse": compatible_reuse,
            "reused_candidate_count": cache.reused_candidate_count,
            "newly_evaluated_candidate_count": cache.newly_recorded_count,
            "search_stats": search_stats,
            "selection_policy": {
                "maximum_overshoot_fraction": spec["targets"]["selection_maximum_overshoot_fraction"],
                "maximum_target_crossings": spec["targets"]["selection_maximum_target_crossings"],
                "hard_maximum_overshoot_fraction": spec["targets"]["hard_maximum_overshoot_fraction"],
                "hard_maximum_target_crossings": spec["targets"]["hard_maximum_target_crossings"],
                "lexicographic_order": [
                    "all_hard_gates_pass",
                    "all_step_scenarios_settled",
                    "selection_overshoot_eligible",
                    "selection_crossing_eligible",
                    "worst_settling_time_s",
                    "worst_overshoot_band_distance",
                    "worst_rise_time_s",
                    "worst_final_abs_position_error_m",
                    "actuator_effort_index",
                    "robustness_index",
                ],
            },
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
