from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from ..config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from ..params import apply_dataclass_overrides, load_interactive_config
    from .headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter
except ImportError:  # pragma: no cover - direct script execution from repository root
    from config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from params import apply_dataclass_overrides, load_interactive_config
    from analysis.headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter


REPO_ROOT = Path(__file__).resolve().parents[1]
SEMINAR_PARAMETER_FILE = REPO_ROOT / "params" / "loiter_transient_provisional.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "seminar_videos"
SOURCE_FILES = tuple(
    REPO_ROOT / relative
    for relative in (
        "generate_seminar_videos.py",
        "analysis/seminar_video_scenarios.py",
        "analysis/seminar_moving_mass_gain_sweep.py",
        "analysis/seminar_video_renderer.py",
        "analysis/headless_loiter.py",
        "interactive_sim.py",
        "interactive_logging.py",
        "cascaded_controller.py",
        "actuators.py",
        "singlecopter_mixer.py",
        "math_utils.py",
        "rigid_body_model.py",
        "safety.py",
        "thrust_curve.py",
        "config.py",
        "params.py",
    )
)
TAIL_WINDOW_S = 2.0
ASSIST_GAIN_M_PER_NM = 0.1325
VANE_SATURATION_REJECTION_PERCENT = 5.0

SELECTED_CONTROLLER_VALUES = {
    "atc_rat_pit_p": 0.07,
    "atc_rat_pit_i": 0.0,
    "atc_rat_pit_d": 0.008,
    "atc_ang_pit_p": 10.0,
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

SHARED_RIGID_BODY_OVERRIDES = {
    "H": 0.5,
    "m": 2.0,
    "moving_mass": {
        "enabled": True,
        "mass_kg": 0.5,
        "max_offset_m": 0.05,
        "max_rate_m_s": 0.2,
        "max_accel_m_s2": 1.0,
        "initial_offset_m": 0.0,
        "moving_mass_body_up_offset_m": 0.12,
        "use_total_com_geometry": True,
        "use_legacy_gravity_offset_moment": False,
    },
}

METRIC_COLUMNS = [
    "scenario",
    "variant",
    "assist_gain_m_per_Nm",
    "tail_rms_x_m",
    "tail_rms_vx_m_s",
    "tail_peak_to_peak_x_m",
    "tail_path_length_m",
    "final_abs_x_error_m",
    "position_overshoot_m",
    "settling_time_s",
    "peak_abs_theta_deg",
    "tail_rms_theta_deg",
    "vane_command_rms_deg",
    "vane_command_max_deg",
    "moving_mass_max_offset_m",
    "moving_mass_tracking_rms_m",
    "premature_pause",
    "second_acceleration_lobe_after_full_pause",
    "early_velocity_reversal",
    "moving_mass_rail_saturation",
    "vane_saturation_percent",
    "target_capture_discontinuity",
    "ground_contact",
    "rejected",
    "rejection_reasons",
    "settled",
    "tail_window_s",
    "duration_s",
    "physics_dt_s",
    "controller_dt_s",
    "simulation_sample_count",
    "tail_rms_x_change_pct",
    "tail_rms_vx_change_pct",
    "tail_peak_to_peak_x_change_pct",
    "tail_path_length_change_pct",
    "final_abs_x_error_change_pct",
    "position_overshoot_change_pct",
    "peak_abs_theta_change_pct",
    "tail_rms_theta_change_pct",
    "vane_command_rms_change_pct",
    "moving_mass_max_offset_delta_mm",
]

PAIRWISE_PERCENT_METRICS = {
    "tail_rms_x_change_pct": "tail_rms_x_m",
    "tail_rms_vx_change_pct": "tail_rms_vx_m_s",
    "tail_peak_to_peak_x_change_pct": "tail_peak_to_peak_x_m",
    "tail_path_length_change_pct": "tail_path_length_m",
    "final_abs_x_error_change_pct": "final_abs_x_error_m",
    "position_overshoot_change_pct": "position_overshoot_m",
    "peak_abs_theta_change_pct": "peak_abs_theta_deg",
    "tail_rms_theta_change_pct": "tail_rms_theta_deg",
    "vane_command_rms_change_pct": "vane_command_rms_deg",
}


@dataclass(frozen=True)
class SeminarScenarioDefinition:
    key: str
    display_name: str
    config: LoiterScenarioConfig
    settling_reference_time_s: float
    response_kind: str
    motion_direction: int


@dataclass(frozen=True)
class SeminarVariantDefinition:
    key: str
    display_name: str
    subtitle: str
    assist_gain_m_per_Nm: float


@dataclass(frozen=True)
class SeminarRunResult:
    scenario: SeminarScenarioDefinition
    variant: SeminarVariantDefinition
    run: LoiterRunResult
    metrics: dict[str, Any]
    rb_config: RigidBodyConfig
    ui_config: InteractiveSimConfig
    controller_config: ControllerConfig

    @property
    def key(self) -> tuple[str, str]:
        return self.scenario.key, self.variant.key

    @property
    def artifact_stem(self) -> str:
        return f"{self.scenario.key}_{self.variant.key}"


def seminar_scenarios(duration_s: float = 8.0) -> tuple[SeminarScenarioDefinition, ...]:
    """Return the two deterministic seminar definitions.

    The 8 N disturbance matches the existing validated
    ``horizontal_impulse_recovery`` magnitude. Only its requested timing is
    changed here, from 1.0--1.2 s to 1.5--1.7 s.
    """
    return (
        SeminarScenarioDefinition(
            key="loiter",
            display_name="LOITER disturbance recovery",
            config=LoiterScenarioConfig(
                name="seminar_loiter_horizontal_disturbance",
                duration_s=float(duration_s),
                initial_x=0.0,
                initial_z=1.0,
                initial_theta_deg=0.0,
                target_x=0.0,
                target_z=1.0,
                disturbance_start_s=1.5,
                disturbance_duration_s=0.2,
                disturbance_force_x=8.0,
                notes=(
                    "Stable LOITER hold followed by the existing validated 8 N "
                    "world-frame horizontal disturbance magnitude from 1.5 to 1.7 s."
                ),
            ),
            settling_reference_time_s=1.7,
            response_kind="disturbance",
            motion_direction=-1,
        ),
        SeminarScenarioDefinition(
            key="forward_1m",
            display_name="+1 m position command and hold",
            config=LoiterScenarioConfig(
                name="seminar_forward_1m_step",
                duration_s=float(duration_s),
                initial_x=0.0,
                initial_z=1.0,
                initial_theta_deg=0.0,
                target_x=0.0,
                target_z=1.0,
                target_step_time_s=1.0,
                target_step_x=1.0,
                notes="Absolute LOITER x target steps from 0 to +1.0 m at exactly t=1.0 s and is held.",
            ),
            settling_reference_time_s=1.0,
            response_kind="target_step",
            motion_direction=1,
        ),
    )


def seminar_variants() -> tuple[SeminarVariantDefinition, ...]:
    return (
        SeminarVariantDefinition("locked", "Vane-only", "Mass locked at center", 0.0),
        SeminarVariantDefinition(
            "assist", "Moving-mass assist", "Active moving mass", ASSIST_GAIN_M_PER_NM
        ),
    )


def _effective_configs() -> tuple[RigidBodyConfig, InteractiveSimConfig, ControllerConfig]:
    rb_cfg, ui_cfg, controller_cfg = load_interactive_config(SEMINAR_PARAMETER_FILE)
    rb_cfg = apply_dataclass_overrides(rb_cfg, SHARED_RIGID_BODY_OVERRIDES, "seminar rigid body")
    return rb_cfg, ui_cfg, controller_cfg


def validate_parameter_sources() -> None:
    """Ensure the authoritative PR #19 profile carries the complete controller bundle."""
    if not SEMINAR_PARAMETER_FILE.is_file():
        raise FileNotFoundError(SEMINAR_PARAMETER_FILE)
    _rb, _ui, controller = load_interactive_config(SEMINAR_PARAMETER_FILE)
    for key, expected in SELECTED_CONTROLLER_VALUES.items():
        actual = getattr(controller, key)
        matches = actual is expected if isinstance(expected, bool) else math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
        )
        if not matches:
            raise ValueError(
                f"{SEMINAR_PARAMETER_FILE}: {key}={actual!r}, expected {expected!r}"
            )


def run_seminar_variant(
    scenario: SeminarScenarioDefinition,
    variant: SeminarVariantDefinition,
) -> SeminarRunResult:
    validate_parameter_sources()
    effective_scenario = replace(
        scenario.config,
        moving_mass_enabled=True,
        moving_mass_target_m=0.0,
        moving_mass_assist_gain_m_per_Nm=variant.assist_gain_m_per_Nm,
    )
    run = run_headless_loiter(
        SEMINAR_PARAMETER_FILE,
        effective_scenario,
        rb_overrides=SHARED_RIGID_BODY_OVERRIDES,
    )
    rb_cfg, ui_cfg, controller_cfg = _effective_configs()
    metric_row = compute_seminar_metrics(scenario, variant, run)
    return SeminarRunResult(
        scenario=scenario,
        variant=variant,
        run=run,
        metrics=metric_row,
        rb_config=rb_cfg,
        ui_config=ui_cfg,
        controller_config=controller_cfg,
    )


def run_all_scenarios(duration_s: float = 8.0) -> list[SeminarRunResult]:
    results = [
        run_seminar_variant(scenario, variant)
        for scenario in seminar_scenarios(duration_s)
        for variant in seminar_variants()
    ]
    _add_pairwise_percentages(results)
    validate_result_set(results)
    return results


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _settling_metric(
    times: np.ndarray,
    x_error: np.ndarray,
    vx: np.ndarray,
    reference_time_s: float,
    duration_s: float,
) -> tuple[float, bool]:
    mask = times >= reference_time_s - 1e-12
    if not np.any(mask):
        return max(0.0, duration_s - reference_time_s), False
    event_times = times[mask]
    event_error = np.abs(x_error[mask])
    event_vx = np.abs(vx[mask])
    suffix_error = np.maximum.accumulate(event_error[::-1])[::-1]
    suffix_vx = np.maximum.accumulate(event_vx[::-1])[::-1]
    candidates = np.flatnonzero((suffix_error <= 0.05) & (suffix_vx <= 0.05))
    if candidates.size:
        return float(max(0.0, event_times[candidates[0]] - reference_time_s)), True
    return max(0.0, duration_s - reference_time_s), False


def detect_response_transient_gates(
    scenario: SeminarScenarioDefinition,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the direction-aware PR #19 pause/lobe/reversal definitions."""
    times = _array(rows, "sim_time")
    vx = _array(rows, "vx")
    x_error = scenario.motion_direction * _array(rows, "x_error")
    aligned_vx = scenario.motion_direction * vx
    sample_dt = float(np.median(np.diff(times))) if len(times) > 1 else float("nan")
    required = max(1, int(math.ceil(0.15 / sample_dt - 1e-9)))
    started = np.flatnonzero(
        (times >= scenario.settling_reference_time_s - 1e-12) & (aligned_vx >= 0.10)
    )
    if not started.size:
        return {
            "premature_pause": False,
            "second_acceleration_lobe_after_full_pause": False,
            "early_velocity_reversal": False,
            "motion_start_detected": False,
        }

    start = int(started[0])
    reached = np.flatnonzero(
        (np.arange(len(times)) >= start) & (x_error <= 0.02)
    )
    end = int(reached[0]) if reached.size else len(times)
    early_reversal = any(
        aligned_vx[index] * aligned_vx[index + 1] < 0.0
        for index in range(start, max(start, end - 1))
    )

    pause_mask = (x_error > 0.10) & (np.abs(vx) < 0.03)
    full_pause_mask = np.abs(vx) < 0.03
    premature_pause = False
    second_lobe = False
    for index in range(start, len(times) - required + 1):
        stop = index + required
        if bool(np.all(pause_mask[index:stop])):
            premature_pause = True
        if bool(np.all(full_pause_mask[index:stop])) and bool(
            np.any(aligned_vx[stop:] >= 0.10)
        ):
            second_lobe = True
        if premature_pause and second_lobe:
            break
    return {
        "premature_pause": bool(premature_pause),
        "second_acceleration_lobe_after_full_pause": bool(second_lobe),
        "early_velocity_reversal": bool(early_reversal),
        "motion_start_detected": True,
    }


def hard_gate_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not metrics["finite"]:
        reasons.append("non_finite")
    if metrics["crash_reason"]:
        reasons.append(f"crash:{metrics['crash_reason']}")
    if metrics["ground_contact"]:
        reasons.append("ground_contact")
    if metrics["premature_pause"]:
        reasons.append("premature_pause")
    if metrics["second_acceleration_lobe_after_full_pause"]:
        reasons.append("second_lobe_after_full_pause")
    if metrics["early_velocity_reversal"]:
        reasons.append("early_velocity_reversal")
    if metrics["moving_mass_rail_saturation"]:
        reasons.append("moving_mass_rail_saturation")
    if metrics["vane_saturation_percent"] > VANE_SATURATION_REJECTION_PERCENT:
        reasons.append("excessive_vane_saturation")
    if metrics["target_capture_discontinuity"]:
        reasons.append("capture_target_discontinuity")
    return reasons


def compute_seminar_metrics(
    scenario: SeminarScenarioDefinition,
    variant: SeminarVariantDefinition,
    run: LoiterRunResult,
) -> dict[str, Any]:
    rows = run.rows
    if not rows:
        raise ValueError(f"{scenario.key}/{variant.key}: simulation produced no rows")

    times = _array(rows, "sim_time")
    x = _array(rows, "x_cg")
    z = _array(rows, "z_cg")
    vx = _array(rows, "vx")
    theta_deg = np.rad2deg(_array(rows, "theta"))
    vane_command_deg = np.rad2deg(_array(rows, "vane_angle_cmd"))
    moving_mass = _array(rows, "moving_mass_offset_m")
    moving_mass_target = _array(rows, "moving_mass_target_m")
    x_error = _array(rows, "x_error")
    tail_start = max(0.0, scenario.config.duration_s - TAIL_WINDOW_S)
    tail = times >= tail_start - 1e-12

    tail_dx = np.diff(x[tail])
    tail_dz = np.diff(z[tail])
    settling_time_s, settled = _settling_metric(
        times,
        x_error,
        vx,
        scenario.settling_reference_time_s,
        scenario.config.duration_s,
    )

    if scenario.response_kind == "target_step":
        post_event = times >= scenario.settling_reference_time_s - 1e-12
        target_x = float(scenario.config.target_step_x or 0.0)
        position_overshoot = float(
            max(0.0, np.max(scenario.motion_direction * (x[post_event] - target_x)))
        ) if np.any(post_event) else 0.0
    else:
        post_event = times >= scenario.settling_reference_time_s - 1e-12
        position_overshoot = (
            float(np.max(np.abs(x_error[post_event]))) if np.any(post_event) else 0.0
        )

    crash_reasons = {str(row.get("crash_reason", "")) for row in rows} - {""}
    ground_contact = any(
        str(row.get("crash_reason", "")) == "ground contact"
        or float(row.get("min_body_z", 1.0)) <= 0.0
        for row in rows
    )
    numeric_series = np.concatenate(
        [times, x, z, vx, theta_deg, vane_command_deg, moving_mass, moving_mass_target, x_error]
    )
    finite = bool(np.all(np.isfinite(numeric_series)))
    transient = detect_response_transient_gates(scenario, rows)
    rail_limit = float(SHARED_RIGID_BODY_OVERRIDES["moving_mass"]["max_offset_m"])
    rail_saturation = bool(
        np.any(np.abs(moving_mass) >= rail_limit - 1e-9)
        or np.any(np.abs(moving_mass_target) >= rail_limit - 1e-9)
    )
    vane_saturation = np.maximum.reduce(
        (
            _array(rows, "servo_angle_saturated"),
            _array(rows, "servo_rate_saturated"),
            _array(rows, "mixer_saturated"),
        )
    )
    target = _array(rows, "target_x")
    capture_event = _array(rows, "target_capture_event") > 0.5
    target_jump = np.r_[0.0, np.abs(np.diff(target))]
    capture_discontinuity = bool(np.any(capture_event & (target_jump > 0.02)))

    metric = {
        "scenario": scenario.key,
        "variant": variant.key,
        "assist_gain_m_per_Nm": float(variant.assist_gain_m_per_Nm),
        "tail_rms_x_m": _rms(x_error[tail]),
        "tail_rms_vx_m_s": _rms(vx[tail]),
        "tail_peak_to_peak_x_m": float(np.ptp(x[tail])) if np.any(tail) else 0.0,
        "tail_path_length_m": float(np.sum(np.hypot(tail_dx, tail_dz))),
        "final_abs_x_error_m": float(abs(x_error[-1])),
        "position_overshoot_m": position_overshoot,
        "settling_time_s": float(settling_time_s),
        "peak_abs_theta_deg": float(np.max(np.abs(theta_deg))),
        "tail_rms_theta_deg": _rms(theta_deg[tail]),
        "vane_command_rms_deg": _rms(vane_command_deg),
        "vane_command_max_deg": float(np.max(np.abs(vane_command_deg))),
        "moving_mass_max_offset_m": float(np.max(np.abs(moving_mass))),
        "moving_mass_tracking_rms_m": _rms(moving_mass - moving_mass_target),
        "finite": finite,
        "crash_reason": "; ".join(sorted(crash_reasons)),
        **transient,
        "moving_mass_rail_saturation": rail_saturation,
        "vane_saturation_percent": float(100.0 * np.mean(vane_saturation > 0.5)),
        "target_capture_discontinuity": capture_discontinuity,
        "ground_contact": bool(ground_contact),
        "settled": bool(settled),
        "tail_window_s": TAIL_WINDOW_S,
        "duration_s": float(scenario.config.duration_s),
        "physics_dt_s": float(rows[0]["physics_dt"]),
        "controller_dt_s": float(rows[0]["controller_dt"]),
        "simulation_sample_count": len(rows),
    }
    rejection_reasons = hard_gate_reasons(metric)
    metric["rejected"] = bool(rejection_reasons)
    metric["rejection_reasons"] = "; ".join(rejection_reasons)
    for key, value in metric.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ValueError(f"{scenario.key}/{variant.key}: non-finite metric {key}={value!r}")
    return metric


def _add_pairwise_percentages(results: list[SeminarRunResult]) -> None:
    by_key = {result.key: result.metrics for result in results}
    for scenario_key in {result.scenario.key for result in results}:
        locked = by_key[(scenario_key, "locked")]
        assist = by_key[(scenario_key, "assist")]
        pair_values: dict[str, float | None] = {}
        for output_key, metric_key in PAIRWISE_PERCENT_METRICS.items():
            baseline = float(locked[metric_key])
            pair_values[output_key] = (
                100.0 * (float(assist[metric_key]) - baseline) / abs(baseline)
                if abs(baseline) > 1e-12
                else None
            )
        pair_values["moving_mass_max_offset_delta_mm"] = 1000.0 * (
            float(assist["moving_mass_max_offset_m"])
            - float(locked["moving_mass_max_offset_m"])
        )
        locked.update(pair_values)
        assist.update(pair_values)


def effective_parameter_mismatches(result: SeminarRunResult) -> list[str]:
    rb = result.rb_config
    mm = rb.moving_mass
    checks = {
        "total mass": math.isclose(rb.m, 2.0),
        "moving mass": math.isclose(mm.mass_kg, 0.5),
        "rail": math.isclose(mm.max_offset_m, 0.05),
        "moving-mass rate": math.isclose(mm.max_rate_m_s, 0.2),
        "moving-mass acceleration": math.isclose(mm.max_accel_m_s2, 1.0),
        "body-up offset": math.isclose(mm.moving_mass_body_up_offset_m, 0.12),
        "moving mass enabled": mm.enabled,
        "total-COM geometry": mm.use_total_com_geometry,
        "legacy moment disabled": not mm.use_legacy_gravity_offset_moment,
    }
    failed = [name for name, passed in checks.items() if not passed]
    for key, expected in SELECTED_CONTROLLER_VALUES.items():
        actual = getattr(result.controller_config, key)
        matches = actual is expected if isinstance(expected, bool) else math.isclose(
            float(actual), float(expected), abs_tol=1e-12
        )
        if not matches:
            failed.append(f"controller:{key}")
    return failed


def validate_result_set(results: Iterable[SeminarRunResult]) -> dict[str, Any]:
    results = list(results)
    keys = [result.key for result in results]
    if len(results) != 4 or len(set(keys)) != 4:
        raise ValueError(f"expected four unique scenario/variant results, got {keys!r}")

    expected_variants = {"locked", "assist"}
    sample_counts = {len(result.run.rows) for result in results}
    if len(sample_counts) != 1:
        raise ValueError(f"simulation logs are not synchronized: sample counts={sorted(sample_counts)}")

    for scenario in {result.scenario.key for result in results}:
        pair = [result for result in results if result.scenario.key == scenario]
        if {result.variant.key for result in pair} != expected_variants:
            raise ValueError(f"{scenario}: missing locked/assist pair")
        if pair[0].scenario.config != pair[1].scenario.config:
            raise ValueError(f"{scenario}: paired scenario definitions differ")
        if pair[0].rb_config != pair[1].rb_config:
            raise ValueError(f"{scenario}: paired physical vehicle definitions differ")
        locked_times = _array(pair[0].run.rows, "sim_time")
        assist_times = _array(pair[1].run.rows, "sim_time")
        if not np.array_equal(locked_times, assist_times):
            raise ValueError(f"{scenario}: paired timestamps differ")

    for result in results:
        failed = effective_parameter_mismatches(result)
        if failed:
            raise ValueError(f"{result.key}: effective parameter mismatch: {', '.join(failed)}")
        if result.variant.key == "locked":
            offsets = _array(result.run.rows, "moving_mass_offset_m")
            targets = _array(result.run.rows, "moving_mass_target_m")
            if not (np.all(offsets == 0.0) and np.all(targets == 0.0)):
                raise ValueError(f"{result.key}: locked moving mass did not remain centered")
        elif not math.isclose(
            result.variant.assist_gain_m_per_Nm, ASSIST_GAIN_M_PER_NM, abs_tol=1e-12
        ):
            raise ValueError(f"{result.key}: assist gain mismatch")

    return {
        "result_count": len(results),
        "unique_key_count": len(set(keys)),
        "simulation_sample_count": sample_counts.pop(),
        "effective_parameter_mismatches": 0,
        "missing_metrics": 0,
        "non_finite_metrics": 0,
    }


def write_metrics_csv(results: Iterable[SeminarRunResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.metrics for result in results]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_markdown(results: Iterable[SeminarRunResult], path: str | Path) -> Path:
    results = list(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {result.key: result.metrics for result in results}
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.2f}%"

    def reduction(value: float | None) -> str:
        return "n/a" if value is None else f"{-value:.1f}% reduction"

    lines = [
        "# Seminar scenario comparison",
        "",
        "These deterministic 2D simulations compare the same 2.0 kg vehicle with its 0.5 kg moving mass physically present in both variants. The Vane-only case commands and maintains a 0 mm offset; it does not remove the mass.",
        "",
        f"Both variants use the PR #19 LOITER controller from `params/loiter_transient_provisional.json`. The active moving-mass gain is `{ASSIST_GAIN_M_PER_NM:.4f} m/Nm`, selected by the four-direction staged sweep documented in `moving_mass_gain_selection.md`.",
        "",
        "The final 2.0 seconds form the tail window. Settling requires the remainder of the 8-second run to stay within 0.05 m position error and 0.05 m/s horizontal speed. Unsettled results are reported honestly rather than extending or cropping the video.",
        "",
        "## Raw metrics from the exact rendered runs",
        "",
        "| Scenario | Variant | Tail RMS x (m) | Tail RMS vx (m/s) | Tail p-p x (m) | Tail path (m) | Abs final x error (m) | Excursion/overshoot (m) | Peak pitch (deg) | Tail RMS pitch (deg) | Vane RMS (deg) | Vane max (deg) | Mass max (mm) | Tracking RMS (mm) | Settled | Pause | Second lobe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for scenario in seminar_scenarios(results[0].scenario.config.duration_s):
        for variant in seminar_variants():
            row = by_key[(scenario.key, variant.key)]
            lines.append(
                f"| {scenario.display_name} | {variant.display_name} | "
                f"{row['tail_rms_x_m']:.5f} | {row['tail_rms_vx_m_s']:.5f} | "
                f"{row['tail_peak_to_peak_x_m']:.5f} | {row['tail_path_length_m']:.5f} | "
                f"{row['final_abs_x_error_m']:.5f} | {row['position_overshoot_m']:.5f} | "
                f"{row['peak_abs_theta_deg']:.3f} | {row['tail_rms_theta_deg']:.3f} | "
                f"{row['vane_command_rms_deg']:.3f} | {row['vane_command_max_deg']:.3f} | "
                f"{1000.0 * row['moving_mass_max_offset_m']:.2f} | "
                f"{1000.0 * row['moving_mass_tracking_rms_m']:.3f} | "
                f"{'yes' if row['settled'] else 'no'} | "
                f"{'yes' if row['premature_pause'] else 'no'} | "
                f"{'yes' if row['second_acceleration_lobe_after_full_pause'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Vane-only to assist percentage changes",
            "",
            "Negative values mean the assist result is lower than Vane-only.",
            "",
            "| Scenario | Tail RMS x | Tail RMS vx | Tail p-p x | Tail path | Final error | Excursion/overshoot | Peak pitch | Tail RMS pitch | Vane RMS | Mass travel delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in seminar_scenarios(results[0].scenario.config.duration_s):
        locked = by_key[(scenario.key, "locked")]
        assist = by_key[(scenario.key, "assist")]
        lines.append(
            f"| {scenario.display_name} | {percent(assist['tail_rms_x_change_pct'])} | "
            f"{percent(assist['tail_rms_vx_change_pct'])} | "
            f"{percent(assist['tail_peak_to_peak_x_change_pct'])} | "
            f"{percent(assist['tail_path_length_change_pct'])} | "
            f"{percent(assist['final_abs_x_error_change_pct'])} | "
            f"{percent(assist['position_overshoot_change_pct'])} | "
            f"{percent(assist['peak_abs_theta_change_pct'])} | "
            f"{percent(assist['tail_rms_theta_change_pct'])} | "
            f"{percent(assist['vane_command_rms_change_pct'])} | "
            f"{assist['moving_mass_max_offset_delta_mm']:+.2f} mm |"
        )
    lines.extend(
        [
            "",
            "## PPT Page 3 replacement values",
            "",
        ]
    )
    loiter_locked = by_key[("loiter", "locked")]
    loiter_assist = by_key[("loiter", "assist")]
    forward_locked = by_key[("forward_1m", "locked")]
    forward_assist = by_key[("forward_1m", "assist")]
    lines.extend(
        [
            "### LOITER",
            "",
            f"- Tail RMS position error: {loiter_locked['tail_rms_x_m']:.5f} m -> {loiter_assist['tail_rms_x_m']:.5f} m ({reduction(loiter_assist['tail_rms_x_change_pct'])})",
            f"- Tail path length: {loiter_locked['tail_path_length_m']:.5f} m -> {loiter_assist['tail_path_length_m']:.5f} m ({reduction(loiter_assist['tail_path_length_change_pct'])})",
            f"- Vane command RMS: {loiter_locked['vane_command_rms_deg']:.3f} deg -> {loiter_assist['vane_command_rms_deg']:.3f} deg ({reduction(loiter_assist['vane_command_rms_change_pct'])})",
            f"- Final position error: {loiter_locked['final_abs_x_error_m']:.5f} m -> {loiter_assist['final_abs_x_error_m']:.5f} m ({reduction(loiter_assist['final_abs_x_error_change_pct'])})",
            "",
            "### +1 m",
            "",
            f"- Tail RMS position error: {forward_locked['tail_rms_x_m']:.5f} m -> {forward_assist['tail_rms_x_m']:.5f} m ({reduction(forward_assist['tail_rms_x_change_pct'])})",
            f"- Peak pitch: {forward_locked['peak_abs_theta_deg']:.3f} deg -> {forward_assist['peak_abs_theta_deg']:.3f} deg ({reduction(forward_assist['peak_abs_theta_change_pct'])})",
            f"- Final position error: {forward_locked['final_abs_x_error_m']:.5f} m -> {forward_assist['final_abs_x_error_m']:.5f} m ({reduction(forward_assist['final_abs_x_error_change_pct'])}); assist overshoot {forward_assist['position_overshoot_m']:.5f} m",
            f"- Vane command RMS: {forward_locked['vane_command_rms_deg']:.3f} deg -> {forward_assist['vane_command_rms_deg']:.3f} deg ({reduction(forward_assist['vane_command_rms_change_pct'])})",
            "",
            "## Interpretation limit",
            "",
            "These results compare implementations inside the same deterministic 2D model. They are not evidence of real-flight equivalence, hardware safety, 3D stability, or calibrated aerodynamic performance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> tuple[list[dict[str, str]], str]:
    sources = [
        {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
        for path in SOURCE_FILES
    ]
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":"))
    return sources, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(
    results: Iterable[SeminarRunResult],
    output_dir: str | Path,
    render_report: dict[str, Any],
    gain_selection: dict[str, Any],
) -> Path:
    results = list(results)
    output_dir = Path(output_dir)
    manifest_render = json.loads(json.dumps(render_report))
    encoder_executable = manifest_render.get("encoder", {}).get("executable", "")
    if encoder_executable:
        manifest_render["encoder"]["executable"] = Path(encoder_executable).name
    artifacts = []
    for name in sorted(render_report.get("artifacts", [])):
        artifact = output_dir / name
        if artifact.is_file():
            artifacts.append(
                {"name": name, "size_bytes": artifact.stat().st_size, "sha256": _sha256(artifact)}
            )

    source_files, source_fingerprint = _source_provenance()
    manifest = {
        "schema_version": 2,
        "generator": "generate_seminar_videos.py",
        "parameter_files": [
            {
                "path": str(SEMINAR_PARAMETER_FILE.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(SEMINAR_PARAMETER_FILE),
            }
        ],
        "selected_controller_values": SELECTED_CONTROLLER_VALUES,
        "selected_assist_gain_m_per_Nm": ASSIST_GAIN_M_PER_NM,
        "gain_selection": gain_selection,
        "source_files": source_files,
        "source_fingerprint_sha256": source_fingerprint,
        "shared_vehicle_configuration": {
            "total_mass_kg": 2.0,
            "moving_mass_kg": 0.5,
            "moving_mass_body_up_offset_m": 0.12,
            "physical_rail_limit_m": 0.05,
            "total_com_geometry_active": True,
            "legacy_gravity_offset_moment": False,
        },
        "scenarios": [asdict(scenario) for scenario in seminar_scenarios(results[0].scenario.config.duration_s)],
        "variants": [asdict(variant) for variant in seminar_variants()],
        "tail_window_s": TAIL_WINDOW_S,
        "percentage_change_definition": "100 * (assist - vane_only) / abs(vane_only)",
        "render": manifest_render,
        "metrics": [result.metrics for result in results],
        "artifacts": artifacts,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
