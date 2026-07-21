from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .headless_loiter import LoiterRunResult, run_headless_loiter, save_loiter_timeseries
from .pitch_damping_retune import (
    CHATTER_THRESHOLDS,
    GROUP_WEIGHTS,
    HARD_GATE_THRESHOLDS,
    SCORE_WEIGHTS,
    ScenarioDefinition,
    _boolean,
    _canonical_json,
    _number,
    _sha256_bytes,
    _strict_settling_time,
    asymmetry_fraction,
    compute_metrics,
    git_revision,
    required_scenarios,
    sha256_file,
    source_hashes,
)
from params import load_interactive_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = REPO_ROOT / "params" / "pitch_damping_retune_provisional.json"
PROVISIONAL_PROFILE = REPO_ROOT / "params" / "moving_mass_gain_resweep_provisional.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "moving_mass_gain_resweep"
SEED = 0

FIXED_CONTROLLER = {
    "atc_rat_pit_p": 0.09375,
    "atc_rat_pit_i": 0.0,
    "atc_rat_pit_d": 0.02100,
    "atc_ang_pit_p": 25.0,
    "psc_ne_pos_p": 0.55,
    "psc_ne_vel_p": 0.70,
    "loit_brk_delay_s": 0.50,
    "loit_brk_acc_mss": 1.00,
    "loit_brk_jerk_msss": 3.00,
    "loit_capture_vx_threshold_ms": 0.08,
    "loit_capture_desired_vx_threshold_ms": 0.02,
    "loit_capture_persistent": True,
    "loit_shaper_clamp_target": True,
    "loit_capture_without_jump": True,
}
STAGE1_GAINS = tuple(round(value * 0.025, 10) for value in range(9)) + (0.1325,)
OLD_PR20_REFERENCE_GAIN = 0.1325
MAX_GAIN = 0.300
STAGE2_HALF_WIDTH = 0.025
STAGE2_STEP = 0.0025
STAGE3_HALF_WIDTH = 0.005
STAGE3_STEP = 0.0005
MIN_MEANINGFUL_SCORE_IMPROVEMENT_PERCENT = 1.0
MOVING_MASS_CHATTER_THRESHOLDS = {
    "command_deadband_m": 0.0005,
    "meaningful_rate_m_s": 0.01,
    "max_meaningful_direction_changes": 30,
    "max_total_travel_per_second_m_s": 0.50,
    "max_tail_high_frequency_energy_m2": 2.5e-5,
    "high_frequency_window_s": 0.10,
}
MOVING_MASS_LIMITER_HARD_GATES = {
    "command_clipping": {"max_duty_percent": 5.0, "max_continuous_duration_s": 0.5},
    "rail_contact": {"max_duty_percent": 5.0, "max_continuous_duration_s": 0.5},
    "rate_limiter": {"max_duty_percent": 20.0, "max_continuous_duration_s": 1.0},
    "acceleration_limiter": {"max_duty_percent": 30.0, "max_continuous_duration_s": 1.0},
}
MOVING_MASS_LIMITER_SCORE_PENALTIES = {
    "command_clipping": 0.25,
    "rail_contact": 0.25,
    "rate_limiter": 0.10,
    "acceleration_limiter": 0.10,
}
MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE = 1e-9
SYMMETRY_METRICS = (
    "tail_rms_pitch_deg",
    "tail_rms_pitch_rate_deg_s",
    "tail_rms_horizontal_velocity_m_s",
    "tail_path_length_m",
    "position_overshoot_m",
    "final_abs_position_error_m",
    "vane_command_rms_deg",
    "moving_mass_rms_displacement_m",
)
SYMMETRY_PAIRS = (
    ("loiter_positive_disturbance", "loiter_negative_disturbance"),
    ("forward_1m", "backward_1m"),
    ("pitch_positive_recovery", "pitch_negative_recovery"),
)


@dataclass(frozen=True)
class GainCandidate:
    stage: str
    gain: float

    @property
    def key(self) -> str:
        return f"gain={self.gain:.8f}"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, content: str) -> None:
    _atomic_bytes(path, content.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer): return int(value)
    return value


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        atomic_text(path, stream.read())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class ScenarioCache:
    def __init__(self, path: Path, fingerprint: str, *, resume: bool):
        self.path = path
        self.fingerprint = fingerprint
        self.rows: list[dict[str, Any]] = []
        self.by_key: dict[str, dict[str, Any]] = {}
        if resume and path.exists():
            for row in read_csv(path):
                if row.get("workflow_fingerprint") != fingerprint:
                    raise ValueError("stale gain-sweep cache fingerprint; use --no-resume")
                key = row.get("run_key", "")
                if not key or key in self.by_key:
                    raise ValueError("invalid duplicate or missing run key")
                self.rows.append(row)
                self.by_key[key] = row

    def save(self) -> None:
        atomic_csv(self.path, self.rows)


def _array(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _duty_and_longest_duration(flags: np.ndarray, time_values: np.ndarray) -> tuple[float, float]:
    """Return sampled duty (%) and maximum contiguous active time (s)."""
    active = np.asarray(flags, dtype=bool)
    if not active.size:
        return 0.0, 0.0
    dt = float(np.median(np.diff(time_values))) if time_values.size > 1 else 0.0
    longest = current = 0
    for value in active:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(100.0 * np.mean(active)), float(longest * dt)


def _gain_key(gain: float, definition: ScenarioDefinition, fingerprint: str) -> str:
    scenario_hash = _sha256_bytes(_canonical_json(asdict(definition)).encode())
    return f"gain={gain:.8f}:scenario={definition.key}:scenario_hash={scenario_hash}:workflow={fingerprint}"


def _controller_overrides() -> dict[str, float | bool]:
    return {**FIXED_CONTROLLER, "enable_noise": False, "random_seed": SEED}


def _moving_mass_metrics(result: LoiterRunResult) -> dict[str, Any]:
    rows = result.rows
    time_values = _array(rows, "time")
    offset = _array(rows, "moving_mass_offset_m")
    target = _array(rows, "moving_mass_target_m")
    velocity = _array(rows, "moving_mass_velocity_m_s")
    dt = float(np.median(np.diff(time_values)))
    acceleration = np.diff(velocity, prepend=velocity[0]) / dt
    limiter_fields = {
        "command_clipping": "moving_mass_command_clipped",
        "rail_contact": "moving_mass_rail_contact",
        "rate_limiter": "moving_mass_rate_limited",
        "acceleration_limiter": "moving_mass_acceleration_limited",
    }
    limiter_metrics = {}
    for name, field in limiter_fields.items():
        duty, longest = _duty_and_longest_duration(_array(rows, field) > 0.5, time_values)
        limiter_metrics[f"moving_mass_{name}_duty_percent"] = duty
        limiter_metrics[f"moving_mass_{name}_longest_continuous_duration_s"] = longest
    travel = float(np.sum(np.abs(np.diff(offset))))
    meaningful = 0
    for index in range(len(velocity) - 1):
        if (
            velocity[index] * velocity[index + 1] < 0.0
            and max(abs(offset[index]), abs(offset[index + 1]))
            >= MOVING_MASS_CHATTER_THRESHOLDS["command_deadband_m"]
            and max(abs(velocity[index]), abs(velocity[index + 1]))
            >= MOVING_MASS_CHATTER_THRESHOLDS["meaningful_rate_m_s"]
        ):
            meaningful += 1
    tail = time_values >= max(0.0, float(time_values[-1]) - 2.5) - 1e-12
    window = max(3, int(round(MOVING_MASS_CHATTER_THRESHOLDS["high_frequency_window_s"] / dt)))
    smoothed = np.convolve(offset, np.ones(window) / window, mode="same")
    return {
        "moving_mass_assist_gain_m_per_Nm": float(result.scenario.moving_mass_assist_gain_m_per_Nm),
        "moving_mass_rms_displacement_m": _rms(offset),
        "moving_mass_total_travel_m": travel,
        "moving_mass_rate_rms_m_s": _rms(velocity),
        "moving_mass_acceleration_rms_m_s2": _rms(acceleration),
        "moving_mass_max_abs_offset_m": float(np.max(np.abs(offset))),
        "moving_mass_max_abs_target_m": float(np.max(np.abs(target))),
        "moving_mass_max_abs_velocity_m_s": float(np.max(np.abs(velocity))),
        "moving_mass_max_abs_acceleration_m_s2": float(np.max(np.abs(acceleration))),
        "moving_mass_saturation_percent": float(100.0 * np.mean(_array(rows, "moving_mass_saturated") > 0.5)),
        "meaningful_moving_mass_direction_change_count": meaningful,
        "moving_mass_total_travel_per_second_m_s": travel / max(float(time_values[-1]), dt),
        "tail_high_frequency_moving_mass_energy_m2": float(
            np.mean((offset[tail] - smoothed[tail]) ** 2)
        ),
        **limiter_metrics,
    }


def _hard_gate_reasons(definition: ScenarioDefinition, metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not metrics.get("finite", False): reasons.append("non_finite")
    if metrics.get("crash"): reasons.append("crash")
    if metrics.get("ground_contact"): reasons.append("ground_contact")
    if _number(metrics.get("peak_abs_pitch_deg"), math.inf) > HARD_GATE_THRESHOLDS["pitch_divergence_deg"]: reasons.append("pitch_divergence")
    for key in ("premature_pause", "early_velocity_reversal", "second_acceleration_lobe_after_full_pause", "capture_discontinuity", "shaped_velocity_sign_reversal_after_release"):
        if _boolean(metrics.get(key)): reasons.append(key)
    if definition.requires_capture_gates and int(_number(metrics.get("target_capture_count"), 0)) != 1: reasons.append("capture_count_not_one")
    for metric, threshold, reason in (
        ("vane_saturation_percent", HARD_GATE_THRESHOLDS["vane_saturation_percent"], "excessive_vane_saturation"),
        ("servo_rate_saturation_percent", HARD_GATE_THRESHOLDS["servo_rate_saturation_percent"], "excessive_servo_rate_saturation"),
        ("mixer_saturation_percent", HARD_GATE_THRESHOLDS["mixer_saturation_percent"], "excessive_mixer_saturation"),
    ):
        if _number(metrics.get(metric), math.inf) > threshold: reasons.append(reason)
    if (
        int(_number(metrics.get("meaningful_vane_sign_change_count"), 0)) > CHATTER_THRESHOLDS["max_meaningful_sign_changes"]
        or _number(metrics.get("vane_total_variation_per_second_deg_s"), math.inf) > CHATTER_THRESHOLDS["max_total_variation_per_second_deg_s"]
        or _number(metrics.get("tail_high_frequency_vane_energy_deg2"), math.inf) > CHATTER_THRESHOLDS["max_tail_high_frequency_energy_deg2"]
    ): reasons.append("vane_chatter")
    for name, limits in MOVING_MASS_LIMITER_HARD_GATES.items():
        if _number(metrics.get(f"moving_mass_{name}_duty_percent"), math.inf) > limits["max_duty_percent"]:
            reasons.append(f"excessive_moving_mass_{name}_duty")
        if _number(metrics.get(f"moving_mass_{name}_longest_continuous_duration_s"), math.inf) > limits["max_continuous_duration_s"]:
            reasons.append(f"excessive_moving_mass_{name}_continuous_duration")
    if _number(metrics.get("moving_mass_max_abs_offset_m"), math.inf) > _number(metrics.get("effective_moving_mass_max_offset_m"), 0.0) + MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE:
        reasons.append("moving_mass_actual_offset_physical_limit_violation")
    if _number(metrics.get("moving_mass_max_abs_velocity_m_s"), math.inf) > _number(metrics.get("effective_moving_mass_max_rate_m_s"), 0.0) + MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE:
        reasons.append("moving_mass_actual_rate_physical_limit_violation")
    if _number(metrics.get("moving_mass_max_abs_acceleration_m_s2"), math.inf) > _number(metrics.get("effective_moving_mass_max_accel_m_s2"), 0.0) + MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE:
        reasons.append("moving_mass_actual_acceleration_physical_limit_violation")
    if (
        int(_number(metrics.get("meaningful_moving_mass_direction_change_count"), 0)) > MOVING_MASS_CHATTER_THRESHOLDS["max_meaningful_direction_changes"]
        or _number(metrics.get("moving_mass_total_travel_per_second_m_s"), math.inf) > MOVING_MASS_CHATTER_THRESHOLDS["max_total_travel_per_second_m_s"]
        or _number(metrics.get("tail_high_frequency_moving_mass_energy_m2"), math.inf) > MOVING_MASS_CHATTER_THRESHOLDS["max_tail_high_frequency_energy_m2"]
    ): reasons.append("moving_mass_chatter")
    for key, expected in FIXED_CONTROLLER.items():
        effective = f"effective_{key}"
        if effective in metrics and not math.isclose(_number(metrics[effective]), float(expected), rel_tol=0.0, abs_tol=1e-12): reasons.append(f"effective_parameter_mismatch:{key}")
    for key, expected in (("total_mass_kg", 2.0), ("physical_moving_mass_kg", 0.5)):
        if not math.isclose(_number(metrics.get(key)), expected, rel_tol=0.0, abs_tol=1e-12): reasons.append(f"effective_parameter_mismatch:{key}")
    if not _boolean(metrics.get("moving_mass_enabled")): reasons.append("physical_moving_mass_disabled")
    if not _boolean(metrics.get("total_com_geometry_active")): reasons.append("total_com_geometry_disabled")
    if _boolean(metrics.get("legacy_gravity_offset_active")): reasons.append("legacy_gravity_offset_active")
    if _number(metrics.get("moving_mass_assist_gain_m_per_Nm"), math.nan) == 0.0 and (
        _number(metrics.get("moving_mass_max_abs_offset_m"), math.inf) != 0.0
        or _number(metrics.get("moving_mass_max_abs_target_m"), math.inf) != 0.0
    ): reasons.append("gain_zero_displacement_not_locked")
    return list(dict.fromkeys(reasons))


def _run_gain(gain: float, definition: ScenarioDefinition, cache: ScenarioCache, fingerprint: str, keep_result: bool = False) -> tuple[dict[str, Any], LoiterRunResult | None]:
    key = _gain_key(gain, definition, fingerprint)
    if key in cache.by_key:
        return dict(cache.by_key[key]), None
    config = asdict(definition.config)
    config.update({"moving_mass_enabled": True, "moving_mass_target_m": 0.0, "moving_mass_assist_gain_m_per_Nm": gain})
    scenario = type(definition.config)(**config)
    result = run_headless_loiter(SOURCE_PROFILE, scenario, controller_overrides=_controller_overrides())
    metrics = compute_metrics(definition, result, quick=False)
    metrics.update(_moving_mass_metrics(result))
    metrics.update({
        "effective_moving_mass_max_offset_m": result.metrics["effective_moving_mass_max_offset_m"],
        "effective_moving_mass_max_rate_m_s": result.metrics["effective_moving_mass_max_rate_m_s"],
        "effective_moving_mass_max_accel_m_s2": result.metrics["effective_moving_mass_max_accel_m_s2"],
    })
    reasons = _hard_gate_reasons(definition, metrics)
    metrics["rejected"] = bool(reasons)
    metrics["rejection_reasons"] = "; ".join(reasons)
    row = {"run_key": key, "workflow_fingerprint": fingerprint, "gain": gain, **metrics}
    cache.rows.append(row); cache.by_key[key] = row; cache.save()
    return row, result if keep_result else None


def _scenario_rows(gain: float, scenarios: Sequence[ScenarioDefinition], cache: ScenarioCache, fingerprint: str) -> list[dict[str, Any]]:
    return [dict(cache.by_key[_gain_key(gain, definition, fingerprint)]) for definition in scenarios]


def _scenario_score(row: dict[str, Any], baseline: dict[str, Any]) -> float:
    performance_score = sum(
        weight * _number(row[metric]) / max(abs(_number(baseline[metric])), 1e-12)
        for metric, weight in SCORE_WEIGHTS.items()
    )
    limiter_penalty = sum(
        weight * _number(row.get(f"moving_mass_{name}_duty_percent"), 0.0) / 100.0
        for name, weight in MOVING_MASS_LIMITER_SCORE_PENALTIES.items()
    )
    return performance_score + limiter_penalty


def aggregate_gain(gain: float, scenarios: Sequence[ScenarioDefinition], cache: ScenarioCache, fingerprint: str, baseline: dict[str, dict[str, Any]], stage: str) -> dict[str, Any]:
    rows = _scenario_rows(gain, scenarios, cache, fingerprint)
    reasons = [reason for row in rows for reason in str(row.get("rejection_reasons", "")).split("; ") if reason]
    scores = {row["scenario_name"]: _scenario_score(row, baseline[row["scenario_name"]]) for row in rows}
    symmetry = {}
    by_name = {str(row["scenario_name"]): row for row in rows}
    for positive, negative in SYMMETRY_PAIRS:
        for metric in SYMMETRY_METRICS:
            symmetry[f"{positive}__{negative}__{metric}"] = asymmetry_fraction(_number(by_name[positive][metric]), _number(by_name[negative][metric]))
    worst_symmetry = max(symmetry.values(), default=0.0)
    if worst_symmetry > HARD_GATE_THRESHOLDS["symmetry_max_fraction"]: reasons.append("severe_mirrored_scenario_asymmetry")
    group_scores = {}
    for group in GROUP_WEIGHTS:
        vals = [scores[definition.key] for definition in scenarios if definition.group == group]
        group_scores[group] = float(np.mean(vals) + 0.5 * np.max(vals))
    final_score = sum(GROUP_WEIGHTS[group] * group_scores[group] for group in GROUP_WEIGHTS)
    return {
        "stage": stage, "gain": gain, "candidate_key": f"gain={gain:.8f}", "scenario_count": len(rows), "rejected": bool(reasons), "rejection_reasons": "; ".join(dict.fromkeys(reasons)),
        "final_score": math.inf if reasons else final_score, "inner_loop_aggregate": group_scores["inner_loop"], "integrated_loiter_aggregate": group_scores["integrated_loiter"], "worst_scenario_score": max(scores.values()), "scenario_mean_score": float(np.mean(list(scores.values()))),
        "mean_vane_command_rms_deg": float(np.mean([_number(row["vane_command_rms_deg"]) for row in rows])), "mean_vane_total_variation_deg": float(np.mean([_number(row["vane_command_total_variation_deg"]) for row in rows])),
        "mean_moving_mass_rms_displacement_m": float(np.mean([_number(row["moving_mass_rms_displacement_m"]) for row in rows])), "mean_moving_mass_total_travel_m": float(np.mean([_number(row["moving_mass_total_travel_m"]) for row in rows])), "worst_asymmetry_fraction": worst_symmetry,
        **{f"mean_moving_mass_{name}_duty_percent": float(np.mean([_number(row[f"moving_mass_{name}_duty_percent"]) for row in rows])) for name in MOVING_MASS_LIMITER_HARD_GATES},
        **{f"max_moving_mass_{name}_longest_continuous_duration_s": float(np.max([_number(row[f"moving_mass_{name}_longest_continuous_duration_s"]) for row in rows])) for name in MOVING_MASS_LIMITER_HARD_GATES},
        "scenario_scores_json": _canonical_json(scores), "symmetry_json": _canonical_json(symmetry),
    }


def _rank(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (_boolean(row["rejected"]), _number(row["final_score"], math.inf), _number(row["gain"])))
    for index, row in enumerate(ranked, 1): row["rank"] = index if not _boolean(row["rejected"]) else ""
    return ranked


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not _boolean(row["rejected"])]
    if not valid: raise RuntimeError("no valid moving-mass gain candidate")
    return min(valid, key=lambda row: _number(row["final_score"]))


def _grid(center: float, half: float, step: float) -> list[float]:
    low, high = max(0.0, center - half), min(MAX_GAIN, center + half)
    count = int(round((high - low) / step))
    return sorted({round(low + index * step, 10) for index in range(count + 1)} | {round(center, 10)})


def _fingerprint() -> tuple[dict[str, Any], str]:
    payload = {
        "schema_version": 1, "base_sha": git_revision("rev-parse", "HEAD"), "source_profile": SOURCE_PROFILE.relative_to(REPO_ROOT).as_posix(), "source_profile_sha256": sha256_file(SOURCE_PROFILE),
        "fixed_controller": FIXED_CONTROLLER, "stage1_gains": STAGE1_GAINS, "stage2": {"half_width": STAGE2_HALF_WIDTH, "step": STAGE2_STEP}, "stage3": {"half_width": STAGE3_HALF_WIDTH, "step": STAGE3_STEP, "max_gain": MAX_GAIN},
        "scenarios": [asdict(item) for item in required_scenarios(False)], "score_weights": SCORE_WEIGHTS, "group_weights": GROUP_WEIGHTS, "hard_gate_thresholds": HARD_GATE_THRESHOLDS, "moving_mass_chatter_thresholds": MOVING_MASS_CHATTER_THRESHOLDS, "moving_mass_limiter_hard_gates": MOVING_MASS_LIMITER_HARD_GATES, "moving_mass_limiter_score_penalties": MOVING_MASS_LIMITER_SCORE_PENALTIES, "moving_mass_physical_limit_tolerance": MOVING_MASS_PHYSICAL_LIMIT_TOLERANCE, "source_hashes": source_hashes(),
    }
    return payload, _sha256_bytes(_canonical_json(payload).encode())


def _validate_source_profile() -> None:
    rb, _ui, controller = load_interactive_config(SOURCE_PROFILE)
    for key, expected in FIXED_CONTROLLER.items():
        actual = getattr(controller, key)
        if actual != expected: raise ValueError(f"merged pitch controller mismatch: {key}={actual!r}, expected {expected!r}")
    moving = rb.moving_mass
    if not (moving.enabled and moving.use_total_com_geometry and not moving.use_legacy_gravity_offset_moment): raise ValueError("source profile does not preserve total-COM simulation-only moving mass configuration")


def _write_timeseries(output_dir: Path, label: str, results: dict[str, LoiterRunResult]) -> None:
    for name, result in results.items(): save_loiter_timeseries(result.rows, output_dir / "validation" / label / f"{name}_timeseries.csv")


def _run_fresh(gain: float, scenarios: Sequence[ScenarioDefinition]) -> tuple[list[dict[str, dict[str, Any]]], dict[str, LoiterRunResult], list[str]]:
    runs, first_results, digests = [], {}, []
    for rerun in range(2):
        metrics_by_scenario = {}
        for definition in scenarios:
            config = asdict(definition.config); config.update({"moving_mass_enabled": True, "moving_mass_target_m": 0.0, "moving_mass_assist_gain_m_per_Nm": gain})
            result = run_headless_loiter(SOURCE_PROFILE, type(definition.config)(**config), controller_overrides=_controller_overrides())
            metrics = compute_metrics(definition, result, quick=False); metrics.update(_moving_mass_metrics(result))
            metrics.update({"effective_moving_mass_max_offset_m": result.metrics["effective_moving_mass_max_offset_m"], "effective_moving_mass_max_rate_m_s": result.metrics["effective_moving_mass_max_rate_m_s"], "effective_moving_mass_max_accel_m_s2": result.metrics["effective_moving_mass_max_accel_m_s2"]})
            reasons = _hard_gate_reasons(definition, metrics); metrics["rejected"] = bool(reasons); metrics["rejection_reasons"] = "; ".join(reasons)
            metrics_by_scenario[definition.key] = metrics
            if rerun == 0: first_results[definition.key] = result
        runs.append(metrics_by_scenario); digests.append(_sha256_bytes(_canonical_json(metrics_by_scenario).encode()))
    if len(set(digests)) != 1: raise RuntimeError(f"deterministic rerun mismatch: {digests}")
    return runs, first_results, digests


def _comparison(baseline: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]], selected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = ("tail_rms_pitch_deg", "tail_rms_pitch_rate_deg_s", "tail_rms_horizontal_velocity_m_s", "tail_path_length_m", "final_abs_position_error_m", "position_overshoot_m", "recovery_excursion_m", "strict_settling_time_s", "vane_command_rms_deg", "vane_command_total_variation_deg", "vane_command_rate_rms_deg_s", "moving_mass_rms_displacement_m", "moving_mass_total_travel_m", "moving_mass_rate_rms_m_s", "moving_mass_acceleration_rms_m_s2", "moving_mass_max_abs_offset_m", "moving_mass_max_abs_velocity_m_s", "moving_mass_max_abs_acceleration_m_s2", "moving_mass_command_clipping_duty_percent", "moving_mass_rail_contact_duty_percent", "moving_mass_rate_limiter_duty_percent", "moving_mass_acceleration_limiter_duty_percent")
    result = {"aggregate": {}, "per_scenario": {}}
    for metric in metrics:
        values = {label: float(np.mean([_number(row[metric]) for row in rows.values()])) for label, rows in (("gain_zero_baseline", baseline), ("old_pr20_reference_0_1325", reference), ("selected", selected))}
        baseline_value = values["gain_zero_baseline"]
        result["aggregate"][metric] = {
            **values,
            "selected_minus_baseline": values["selected"] - baseline_value,
            "selected_vs_baseline_percent": (
                100.0 * (values["selected"] - baseline_value) / abs(baseline_value)
                if abs(baseline_value) > 1e-12 else None
            ),
        }
    for name in selected:
        result["per_scenario"][name] = {"gain_zero_baseline": baseline[name], "old_pr20_reference_0_1325": reference[name], "selected": selected[name]}
    return result


def _comparison_csv_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, values in comparison["aggregate"].items():
        rows.append({"scope": "aggregate_mean", "scenario": "all", "metric": metric, **values})
    for scenario, values in comparison["per_scenario"].items():
        for controller_set, metrics in values.items():
            rows.append({
                "scope": "scenario_hard_gates", "scenario": scenario, "controller_set": controller_set,
                "rejected": metrics["rejected"], "rejection_reasons": metrics["rejection_reasons"],
                "early_velocity_reversal": metrics["early_velocity_reversal"], "premature_pause": metrics["premature_pause"],
                "second_acceleration_lobe_after_full_pause": metrics["second_acceleration_lobe_after_full_pause"],
                "target_capture_count": metrics["target_capture_count"], "capture_discontinuity": metrics["capture_discontinuity"],
                "shaped_velocity_sign_reversal_after_release": metrics["shaped_velocity_sign_reversal_after_release"],
                "vane_saturation_percent": metrics["vane_saturation_percent"], "servo_rate_saturation_percent": metrics["servo_rate_saturation_percent"],
                "mixer_saturation_percent": metrics["mixer_saturation_percent"], "moving_mass_saturation_percent": metrics["moving_mass_saturation_percent"],
                "moving_mass_command_clipping_duty_percent": metrics["moving_mass_command_clipping_duty_percent"], "moving_mass_command_clipping_longest_continuous_duration_s": metrics["moving_mass_command_clipping_longest_continuous_duration_s"],
                "moving_mass_rail_contact_duty_percent": metrics["moving_mass_rail_contact_duty_percent"], "moving_mass_rail_contact_longest_continuous_duration_s": metrics["moving_mass_rail_contact_longest_continuous_duration_s"],
                "moving_mass_rate_limiter_duty_percent": metrics["moving_mass_rate_limiter_duty_percent"], "moving_mass_rate_limiter_longest_continuous_duration_s": metrics["moving_mass_rate_limiter_longest_continuous_duration_s"],
                "moving_mass_acceleration_limiter_duty_percent": metrics["moving_mass_acceleration_limiter_duty_percent"], "moving_mass_acceleration_limiter_longest_continuous_duration_s": metrics["moving_mass_acceleration_limiter_longest_continuous_duration_s"],
                "moving_mass_max_abs_offset_m": metrics["moving_mass_max_abs_offset_m"], "moving_mass_max_abs_velocity_m_s": metrics["moving_mass_max_abs_velocity_m_s"], "moving_mass_max_abs_acceleration_m_s2": metrics["moving_mass_max_abs_acceleration_m_s2"],
                "meaningful_vane_sign_change_count": metrics["meaningful_vane_sign_change_count"], "meaningful_moving_mass_direction_change_count": metrics["meaningful_moving_mass_direction_change_count"],
                "worst_asymmetry_fraction": "see_candidate_results", "finite": metrics["finite"], "crash": metrics["crash"], "ground_contact": metrics["ground_contact"],
            })
    return rows


def _selection_markdown(summary: dict[str, Any], comparison: dict[str, Any]) -> str:
    rows = [
        "# Moving-mass gain selection",
        "",
        "The merged PR #22 Pitch controller is fixed. This is a simulation-only virtual-actuator sweep; no hardware, HIL, Pixhawk, Raspberry Pi, or real-flight claim is made.",
        "",
        "Stage 0 gain `0.0` is a **VALID Vane-only comparison baseline**: it passes all seven full-duration scenarios with exactly zero moving-mass target and actual displacement.",
        "",
        f"Raw-score rank 1 is gain `{_number(summary['raw_score_rank1']['gain']):.5f}` with score `{_number(summary['raw_score_rank1']['final_score']):.12f}`. Decision: **{summary['decision']}**.",
        "",
        "Normal Moving-mass limiter engagement is recorded and penalized, not automatically rejected. Actual offset/rate/acceleration violations use a +1e-9 numerical tolerance.",
        "",
        "| metric | gain 0 Vane-only | old PR #20 reference 0.1325 | selected | selected vs gain 0 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in comparison["aggregate"].items():
        reference = values["old_pr20_reference_0_1325"]
        reference_text = f"{reference:.9f}" if math.isfinite(reference) else "hard-rejected"
        change = values["selected_vs_baseline_percent"]
        change_text = f"{change:.3f}%" if change is not None else f"new; delta {values['selected_minus_baseline']:.9f}"
        rows.append(f"| {metric} | {values['gain_zero_baseline']:.9f} | {reference_text} | {values['selected']:.9f} | {change_text} |")
    worse = [metric for metric, values in comparison["aggregate"].items() if values["selected_minus_baseline"] > 1e-12]
    selected_rows = [values["selected"] for values in comparison["per_scenario"].values()]
    rows.extend(["", "Metrics worse than gain zero: " + (", ".join(worse) if worse else "none") + ".", "", "| selected limiter diagnostic | maximum duty | longest continuous duration | hard-gate duty / duration |", "| --- | ---: | ---: | ---: |"])
    for name, limits in MOVING_MASS_LIMITER_HARD_GATES.items():
        rows.append(f"| {name} | {max(_number(item[f'moving_mass_{name}_duty_percent']) for item in selected_rows):.3f}% | {max(_number(item[f'moving_mass_{name}_longest_continuous_duration_s']) for item in selected_rows):.3f} s | {limits['max_duty_percent']:.1f}% / {limits['max_continuous_duration_s']:.1f} s |")
    rows.extend(["", "Actual selected maxima remain within the unchanged physical limits (offset 0.050 m, rate 0.200 m/s, acceleration 1.000 m/s²), using the required +1e-9 tolerance for violation checks.", "The comparison CSV/JSON preserves every scenario-level hard-gate, chatter, saturation, symmetry, and transient field.", ""])
    return "\n".join(rows)


def _plots(output_dir: Path, baseline: dict[str, LoiterRunResult], selected: dict[str, LoiterRunResult]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = (("baseline_vs_selected_loiter.png", ("loiter_positive_disturbance", "loiter_negative_disturbance")), ("baseline_vs_selected_forward_1m.png", ("forward_1m", "backward_1m")), ("baseline_vs_selected_pitch_recovery.png", ("pitch_positive_recovery", "pitch_negative_recovery")), ("baseline_vs_selected_stick_release.png", ("stick_release",)))
    destination = output_dir / "plots"; destination.mkdir(parents=True, exist_ok=True)
    for filename, names in specs:
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for name in names:
            for label, source, style in (("gain 0", baseline, "--"), ("selected", selected, "-")):
                rows = source[name].rows; t = _array(rows, "time")
                axes[0].plot(t, _array(rows, "x"), style, label=f"{label}: {name}")
                axes[1].plot(t, np.rad2deg(_array(rows, "theta")), style, label=f"{label}: {name}")
                axes[2].plot(t, _array(rows, "moving_mass_offset_m"), style, label=f"{label}: {name}")
        for axis, ylabel in zip(axes, ("x (m)", "pitch (deg)", "moving-mass offset (m)")):
            axis.set_ylabel(ylabel); axis.grid(alpha=0.25); axis.legend(fontsize=7, ncol=2)
        axes[-1].set_xlabel("time (s)"); figure.tight_layout(); figure.savefig(destination / filename, dpi=160); plt.close(figure)


def run_resweep(output_dir: Path = DEFAULT_OUTPUT_DIR, *, resume: bool = True) -> dict[str, Any]:
    started = time.perf_counter(); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    _validate_source_profile(); scenarios = required_scenarios(False); fingerprint_payload, fingerprint = _fingerprint(); cache = ScenarioCache(output_dir / "scenario_results.csv", fingerprint, resume=resume)
    def evaluate(gains: Sequence[float], stage: str) -> list[dict[str, Any]]:
        for index, gain in enumerate(gains, 1):
            for definition in scenarios: _run_gain(gain, definition, cache, fingerprint)
            print(f"{stage}: {index}/{len(gains)} gains", flush=True)
        return _rank([aggregate_gain(gain, scenarios, cache, fingerprint, baseline_by_scenario, stage) for gain in gains])

    baseline_rows = []
    for definition in scenarios: baseline_rows.append(_run_gain(0.0, definition, cache, fingerprint)[0])
    baseline_by_scenario = {str(row["scenario_name"]): row for row in baseline_rows}
    baseline = aggregate_gain(0.0, scenarios, cache, fingerprint, baseline_by_scenario, "stage0_gain_zero")
    if _boolean(baseline["rejected"]): raise RuntimeError("gain-0 Vane-only baseline failed: " + str(baseline["rejection_reasons"]))
    stage1 = evaluate(sorted(set(STAGE1_GAINS)), "stage1_coarse")
    stage2 = evaluate(_grid(_number(_best(stage1)["gain"]), STAGE2_HALF_WIDTH, STAGE2_STEP), "stage2_fine")
    stage3_gains = _grid(_number(_best(stage2)["gain"]), STAGE3_HALF_WIDTH, STAGE3_STEP)
    stage3 = evaluate(stage3_gains, "stage3_local")
    boundary_rows = []
    while True:
        best = _best(stage3); upper = math.isclose(_number(best["gain"]), max(stage3_gains), abs_tol=1e-12)
        boundary_rows.append({"round": len(boundary_rows), "gain_min": min(stage3_gains), "gain_max": max(stage3_gains), "best_gain": best["gain"], "best_score": best["final_score"], "best_at_upper_boundary": upper})
        if not upper: break
        next_gain = round(max(stage3_gains) + STAGE3_STEP, 10)
        if next_gain > MAX_GAIN + 1e-12: raise RuntimeError("best gain reached 0.300 m/Nm boundary; authorization required before extending")
        stage3_gains.append(next_gain); stage3_gains.sort(); stage3 = evaluate(stage3_gains, "stage3_local")
    raw_best = _best(stage3); old_reference = next(row for row in stage1 if math.isclose(_number(row["gain"]), OLD_PR20_REFERENCE_GAIN, abs_tol=1e-12))
    improvement_percent = 100.0 * (_number(baseline["final_score"]) - _number(raw_best["final_score"])) / _number(baseline["final_score"])
    adopt_nonzero = _number(raw_best["gain"]) > 0.0 and improvement_percent >= MIN_MEANINGFUL_SCORE_IMPROVEMENT_PERCENT
    selected = raw_best if adopt_nonzero or math.isclose(_number(raw_best["gain"]), 0.0, abs_tol=1e-12) else baseline
    if _number(raw_best["gain"]) == 0.0:
        decision = "retain_gain_zero_valid_raw_score_rank1"
    elif selected is raw_best:
        decision = "adopt_valid_raw_score_rank1"
    else:
        decision = "retain_gain_zero_practically_marginal"
    validation_runs, selected_results, digests = _run_fresh(_number(selected["gain"]), scenarios)
    if any(_boolean(metrics["rejected"]) for metrics in validation_runs[0].values()): raise RuntimeError("selected deterministic validation failed hard gates")
    baseline_runs, baseline_results, _baseline_digests = _run_fresh(0.0, scenarios)
    reference_runs, _reference_results, _reference_digests = _run_fresh(OLD_PR20_REFERENCE_GAIN, scenarios)
    comparison = _comparison(baseline_runs[0], reference_runs[0], validation_runs[0])
    _write_timeseries(output_dir, "selected", selected_results); _write_timeseries(output_dir, "gain_zero", baseline_results); _plots(output_dir, baseline_results, selected_results)
    aggregates = [baseline, *stage1, *stage2, *stage3]
    summary = {"schema_version": 1, "decision": decision, "selected": selected, "raw_score_rank1": raw_best, "gain_zero_baseline": baseline, "old_pr20_reference_gain": old_reference, "raw_score_improvement_vs_gain_zero_percent": improvement_percent, "practically_marginal": not adopt_nonzero and _number(raw_best["gain"]) > 0.0, "candidate_counts": {"stage1": len(stage1), "stage2": len(stage2), "stage3": len(stage3)}, "unique_gain_count": len({float(row["gain"]) for row in cache.rows}), "scenario_run_count": len(cache.rows), "deterministic_digests": digests, "fixed_controller": FIXED_CONTROLLER, "stage0_status": "VALID Vane-only comparison baseline", "old_pr20_note": "PR #20 uses the obsolete pre-PR-22 Pitch controller and is comparison-only.", "runtime_s": time.perf_counter() - started}
    atomic_csv(output_dir / "candidate_results.csv", aggregates); atomic_csv(output_dir / "stage1_coarse.csv", stage1); atomic_csv(output_dir / "stage2_fine.csv", stage2); atomic_csv(output_dir / "stage3_local.csv", stage3); atomic_csv(output_dir / "boundary_diagnostics.csv", boundary_rows); atomic_csv(output_dir / "baseline" / "baseline_scenario_results.csv", baseline_rows); atomic_json(output_dir / "selection_comparison.json", comparison); atomic_csv(output_dir / "selection_comparison.csv", _comparison_csv_rows(comparison)); atomic_json(output_dir / "validation" / "deterministic_reruns.json", {"digests": digests, "runs": validation_runs, "passed": True})
    if _number(raw_best["gain"]) == 0.0:
        decision_note = "Gain zero is the valid raw-score rank-1 result, so no nonzero moving-mass gain is adopted."
    elif summary["practically_marginal"]:
        decision_note = "The nonzero score improvement is practically marginal, so gain zero is retained pending authorization."
    else:
        decision_note = "The selected nonzero gain satisfies the required meaningful-improvement threshold."
    report = f"""# Moving-mass assist-gain resweep\n\nThe merged Pitch controller is fixed at P/I/D `{FIXED_CONTROLLER['atc_rat_pit_p']:.5f} / 0.0 / {FIXED_CONTROLLER['atc_rat_pit_d']:.5f}`, Angle P `{FIXED_CONTROLLER['atc_ang_pit_p']:.1f}`. Only the simulation-only `moving_mass_assist_gain_m_per_Nm` changed.\n\nStage 0 gain `0.0` is the valid Vane-only comparison baseline: it passes all seven full-duration scenarios and maintains exactly zero moving-mass actual and target displacement.\n\nRaw-score rank 1 is gain `{_number(raw_best['gain']):.5f}` with score `{_number(raw_best['final_score']):.12f}`. Gain-zero score is `{_number(baseline['final_score']):.12f}`; improvement is `{improvement_percent:.3f}%`. Decision: **{decision}**. {decision_note}\n\nThe old PR #20 gain `0.1325` is included only as a reference. PR #20 uses the obsolete pre-PR-22 Pitch controller and was not modified.\n\nAll selected hard gates, symmetry, chatter, saturation, and moving-mass physical-limit results are in `selection_comparison.json` and the scenario results. No hardware, HIL, Pixhawk, Raspberry Pi, or real-flight claim is made.\n"""
    atomic_text(output_dir / "selection_report.md", report + "\n" + _selection_markdown(summary, comparison))
    atomic_text(output_dir / "methodology.md", "# Methodology\n\nSeven full-duration mirrored LOITER/Pitch/stick-release scenarios were evaluated deterministically with the merged PR #22 Pitch controller fixed. Stage 1: 0.000..0.200 m/Nm by 0.025 plus 0.1325. Stage 2: best +/-0.025 by 0.0025. Stage 3: best +/-0.005 by 0.0005, extending only an upper boundary and never above 0.300 without authorization. Raw aggregate score uses the pre-existing pitch-retune scenario weights and group aggregation plus predeclared Moving-mass limiter-duty penalties; no near-equivalent tie-break is used. Command clipping and rail contact hard-reject above 5% duty or 0.5 s continuous; rate limiting above 20% or 1.0 s; acceleration limiting above 30% or 1.0 s. Actual offset, rate, and acceleration reject only above their unchanged physical limits plus 1e-9.\n")
    profile = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8")); profile.setdefault("analysis", {}).update({"profile_status": "provisional", "profile_notes": "Simulation-only moving-mass assist gain resweep; not hardware or flight validation.", "moving_mass_assist_gain_m_per_Nm": _number(selected["gain"]), "gain_resweep_summary": summary}); atomic_json(PROVISIONAL_PROFILE, profile)
    atomic_json(output_dir / "summary.json", summary)
    rejection_counts = Counter(reason for row in aggregates if _boolean(row["rejected"]) for reason in str(row["rejection_reasons"]).split("; ") if reason)
    atomic_csv(output_dir / "rejection_summary.csv", [{"rejection_category": reason, "candidate_count": count} for reason, count in sorted(rejection_counts.items())])
    artifacts = {}
    for path in sorted(file for file in output_dir.rglob("*") if file.is_file() and file.name != "manifest.json"): artifacts[path.relative_to(output_dir).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {"schema_version": 1, "deterministic": True, "summary": summary, "fingerprint_payload": fingerprint_payload, "artifacts": artifacts, "provisional_profile_sha256": sha256_file(PROVISIONAL_PROFILE)}; atomic_json(output_dir / "manifest.json", manifest)
    return summary


def refresh_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Re-hash final publication artifacts added after the simulation run."""
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {}
    for path in sorted(file for file in output_dir.rglob("*") if file.is_file() and file.name != "manifest.json"):
        artifacts[path.relative_to(output_dir).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest["artifacts"] = artifacts
    manifest["provisional_profile_sha256"] = sha256_file(PROVISIONAL_PROFILE)
    atomic_json(manifest_path, manifest)
