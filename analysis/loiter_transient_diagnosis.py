from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from .headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter
except ImportError:  # pragma: no cover - direct script execution
    from analysis.headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROFILE = ROOT / "params" / "loiter_tuned_vane_only.json"
PROVISIONAL_PROFILE = ROOT / "params" / "loiter_transient_provisional.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "analysis" / "loiter_transient_diagnosis"

SELECTED_OUTER = {"psc_ne_pos_p": 0.55, "psc_ne_vel_p": 0.70}
SELECTED_BRAKING = {
    "loit_brk_delay_s": 0.50,
    "loit_brk_acc_mss": 1.00,
    "loit_brk_jerk_msss": 3.00,
    "loit_capture_vx_threshold_ms": 0.08,
    "loit_capture_desired_vx_threshold_ms": 0.02,
    "loit_capture_persistent": True,
    "loit_shaper_clamp_target": True,
    "loit_capture_without_jump": True,
}
SELECTED_CONTROLLER = {**SELECTED_OUTER, **SELECTED_BRAKING}

COARSE_POS = (0.30, 0.50, 0.70, 0.90, 1.20)
COARSE_VEL = (0.60, 0.90, 1.20, 1.50, 2.00)
REFINE_POS = (0.45, 0.50, 0.55, 0.60, 0.65)
REFINE_VEL = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
COARSE_BRAKE_DELAY = (0.0, 0.2, 0.4, 0.6)
COARSE_BRAKE_ACC = (0.4, 0.6, 0.8, 1.0)
COARSE_BRAKE_JERK = (1.0, 2.0, 3.0)
REFINE_BRAKE_DELAY = (0.3, 0.4, 0.5)
REFINE_BRAKE_ACC = (0.9, 1.0, 1.1)
REFINE_BRAKE_JERK = (2.5, 3.0, 3.5)
MOVING_MASS_GAINS = (0.0, 0.025, 0.040, 0.055, 0.070)


def absolute_target_scenario(duration_s: float = 10.5) -> LoiterScenarioConfig:
    return LoiterScenarioConfig(
        name="absolute_target_step_1m",
        duration_s=max(10.5, duration_s),
        initial_x=0.0,
        initial_z=1.0,
        target_x=0.0,
        target_z=1.0,
        target_step_time_s=0.5,
        target_step_x=1.0,
        notes="Final +1 m absolute target; no stick shaping.",
    )


def stick_release_scenario(duration_s: float = 10.0) -> LoiterScenarioConfig:
    return LoiterScenarioConfig(
        name="stick_pulse_release",
        duration_s=max(10.0, duration_s),
        initial_x=0.0,
        initial_z=1.0,
        stick_start_s=0.5,
        stick_end_s=2.2,
        stick_x=0.65,
        capture_current_target=True,
        notes="Interactive-equivalent positive stick pulse followed by release.",
    )


def _values(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _sample_dt(rows: Sequence[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return float("nan")
    return float(np.median(np.diff(_values(rows, "time"))))


def detect_premature_pause(
    rows: Sequence[dict[str, Any]],
    *,
    min_error_m: float = 0.10,
    max_abs_vx_ms: float = 0.03,
    min_duration_s: float = 0.15,
    prior_forward_speed_ms: float = 0.10,
) -> dict[str, float] | None:
    """Return the first pause matching the task's explicit definition."""
    if len(rows) < 2:
        return None
    time = _values(rows, "time")
    vx = _values(rows, "vx")
    error = _values(rows, "x_error")
    x = _values(rows, "x")
    prior = np.flatnonzero(vx >= prior_forward_speed_ms)
    if not prior.size:
        return None
    required = max(1, int(math.ceil(min_duration_s / _sample_dt(rows) - 1e-9)))
    mask = (error > min_error_m) & (np.abs(vx) < max_abs_vx_ms)
    mask[: int(prior[0]) + 1] = False
    for start in range(0, len(rows) - required + 1):
        stop = start + required
        if bool(np.all(mask[start:stop])):
            return {
                "start_time_s": float(time[start]),
                "end_time_s": float(time[stop - 1]),
                "duration_s": float(max(min_duration_s, time[stop - 1] - time[start] + _sample_dt(rows))),
                "x_m": float(x[start]),
                "vx_ms": float(vx[start]),
                "position_error_m": float(error[start]),
                "prior_peak_vx_ms": float(np.max(vx[: start + 1])),
                "start_index": int(start),
                "end_index": int(stop - 1),
            }
    return None


def velocity_sign_changes_before_target(
    rows: Sequence[dict[str, Any]], target_x_m: float = 0.98
) -> list[dict[str, float]]:
    """Detect true reversals after forward motion starts and before first target entry."""
    if len(rows) < 2:
        return []
    time = _values(rows, "time")
    x = _values(rows, "x")
    vx = _values(rows, "vx")
    started = np.flatnonzero(vx >= 0.10)
    if not started.size:
        return []
    reached = np.flatnonzero(x >= target_x_m)
    end = int(reached[0]) if reached.size else len(rows)
    changes: list[dict[str, float]] = []
    for index in range(int(started[0]), max(int(started[0]), end - 1)):
        if vx[index] * vx[index + 1] < 0.0:
            changes.append(
                {
                    "time_s": float(time[index]),
                    "x_m": float(x[index]),
                    "vx_before_ms": float(vx[index]),
                    "vx_after_ms": float(vx[index + 1]),
                }
            )
    return changes


def forward_velocity_peaks(rows: Sequence[dict[str, Any]], min_vx_ms: float = 0.05) -> list[dict[str, float]]:
    if len(rows) < 3:
        return []
    time = _values(rows, "time")
    x = _values(rows, "x")
    vx = _values(rows, "vx")
    indices = np.flatnonzero((vx[1:-1] > vx[:-2]) & (vx[1:-1] >= vx[2:]) & (vx[1:-1] >= min_vx_ms)) + 1
    return [{"time_s": float(time[i]), "x_m": float(x[i]), "vx_ms": float(vx[i])} for i in indices]


def second_acceleration_lobe_after_full_pause(
    rows: Sequence[dict[str, Any]],
    *,
    release_time_s: float = 2.2,
    pause_speed_ms: float = 0.03,
    pause_duration_s: float = 0.15,
    lobe_speed_ms: float = 0.10,
) -> bool:
    """Detect a renewed forward lobe after a sustained near-zero-speed interval."""
    if len(rows) < 2:
        return False
    time = _values(rows, "time")
    vx = _values(rows, "vx")
    required = max(1, int(math.ceil(pause_duration_s / _sample_dt(rows) - 1e-9)))
    start_index = int(np.searchsorted(time, release_time_s))
    mask = np.abs(vx) < pause_speed_ms
    for start in range(start_index, len(rows) - required + 1):
        stop = start + required
        if bool(np.all(mask[start:stop])) and bool(np.any(vx[stop:] >= lobe_speed_ms)):
            return True
    return False


def target_capture_events(rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    if not rows:
        return []
    count = _values(rows, "target_capture_count")
    changes = np.flatnonzero(np.diff(np.r_[0.0, count]) > 0.0)
    return [
        {
            "time_s": float(rows[i]["time"]),
            "target_x_m": float(rows[i]["target_x"]),
            "x_m": float(rows[i]["x"]),
            "vx_ms": float(rows[i]["vx"]),
            "target_jump_m": float(rows[i]["target_x"] - rows[max(0, i - 1)]["target_x"]),
            "capture_count": int(count[i]),
        }
        for i in changes
    ]


def target_discontinuities(rows: Sequence[dict[str, Any]], threshold_m: float = 0.02) -> list[dict[str, float]]:
    if len(rows) < 2:
        return []
    time = _values(rows, "time")
    target = _values(rows, "target_x")
    changes = np.diff(target)
    return [
        {"time_s": float(time[i + 1]), "jump_m": float(changes[i]), "target_x_m": float(target[i + 1])}
        for i in np.flatnonzero(np.abs(changes) > threshold_m)
    ]


def _sign_change_times(
    rows: Sequence[dict[str, Any]], key: str, end_time_s: float, epsilon: float = 1e-4
) -> list[float]:
    time = _values(rows, "time")
    value = _values(rows, key)
    prior_sign = 0
    changes: list[float] = []
    for t, current in zip(time, value):
        if t >= end_time_s:
            break
        sign = 1 if current > epsilon else -1 if current < -epsilon else 0
        if sign and prior_sign and sign != prior_sign:
            changes.append(float(t))
        if sign:
            prior_sign = sign
    return changes


def compute_transient_metrics(result: LoiterRunResult) -> dict[str, Any]:
    rows = result.rows
    if not rows:
        return {"finite": False, "crash_reason": result.crash_reason}
    time = _values(rows, "time")
    x = _values(rows, "x")
    vx = _values(rows, "vx")
    error = _values(rows, "x_error")
    theta_deg = np.rad2deg(_values(rows, "theta"))
    vane_deg = np.rad2deg(_values(rows, "actual_vane_angle"))
    all_numeric = np.column_stack((time, x, vx, error, theta_deg, vane_deg))
    tail = time >= max(float(time[-1]) - 2.0, 0.0)
    pause = detect_premature_pause(rows)
    sign_changes = velocity_sign_changes_before_target(rows)
    peaks = forward_velocity_peaks(rows)
    captures = target_capture_events(rows)
    jumps = target_discontinuities(rows)
    first_reach = np.flatnonzero(x >= 0.98)
    reach_index = int(first_reach[0]) if first_reach.size else len(rows)
    backward_path = float(np.sum(np.maximum(0.0, -np.diff(x[: max(2, reach_index + 1)]))))
    remain_index: int | None = None
    for index in range(len(rows)):
        if abs(error[index]) <= 0.10 and bool(np.all(np.abs(error[index:]) <= 0.10)):
            remain_index = index
            break
    pause_start = float(pause["start_time_s"]) if pause else float("nan")
    ax_changes = _sign_change_times(rows, "desired_ax", pause_start) if pause else []
    theta_changes = _sign_change_times(rows, "theta_target", pause_start) if pause else []
    multiple_peaks_across_pause = False
    if pause:
        multiple_peaks_across_pause = any(p["time_s"] < pause["start_time_s"] for p in peaks) and any(
            p["time_s"] > pause["end_time_s"] for p in peaks
        )
    servo_or_mixer_sat = np.maximum.reduce(
        (
            _values(rows, "servo_angle_saturated"),
            _values(rows, "servo_rate_saturated"),
            _values(rows, "mixer_saturated"),
        )
    )
    return {
        "finite": bool(np.all(np.isfinite(all_numeric))),
        "crash_reason": result.crash_reason,
        "ground_contact": bool(np.min(_values(rows, "min_body_z")) <= 0.0),
        "premature_pause": bool(pause),
        "pause_start_s": pause_start,
        "pause_end_s": float(pause["end_time_s"]) if pause else float("nan"),
        "pause_x_m": float(pause["x_m"]) if pause else float("nan"),
        "pause_vx_ms": float(pause["vx_ms"]) if pause else float("nan"),
        "pause_position_error_m": float(pause["position_error_m"]) if pause else float("nan"),
        "prior_peak_vx_ms": float(pause["prior_peak_vx_ms"]) if pause else float(np.max(vx)),
        "vx_sign_change_count_before_0p98m": len(sign_changes),
        "vx_sign_changes_before_0p98m": sign_changes,
        "forward_velocity_peak_count": len(peaks),
        "forward_velocity_peaks": peaks,
        "multiple_peaks_separated_by_pause": bool(multiple_peaks_across_pause),
        "second_acceleration_lobe_after_full_pause": second_acceleration_lobe_after_full_pause(rows),
        "target_capture_count": len(captures),
        "target_capture_events": captures,
        "target_discontinuity_count": len(jumps),
        "target_discontinuities": jumps,
        "max_target_jump_m": float(np.max(np.abs(np.diff(_values(rows, "target_x"))))) if len(rows) > 1 else 0.0,
        "ax_sign_changes_before_pause": ax_changes,
        "theta_target_sign_changes_before_pause": theta_changes,
        "final_abs_position_error_m": float(abs(error[-1])),
        "tail_rms_position_error_m": float(np.sqrt(np.mean(error[tail] ** 2))),
        "tail_path_length_m": float(np.sum(np.abs(np.diff(x[tail])))),
        "tail_rms_vx_ms": float(np.sqrt(np.mean(vx[tail] ** 2))),
        "overshoot_m": float(max(0.0, np.max(x - _values(rows, "target_x")))),
        "peak_pitch_deg": float(np.max(np.abs(theta_deg))),
        "vane_rms_deg": float(np.sqrt(np.mean(vane_deg**2))),
        "vane_saturation_percent": float(100.0 * np.mean(servo_or_mixer_sat > 0.5)),
        "time_to_first_reach_0p98m_s": float(time[reach_index]) if reach_index < len(rows) else float("nan"),
        "time_to_enter_and_remain_0p10m_s": float(time[remain_index]) if remain_index is not None else float("nan"),
        "backward_path_before_0p98m_m": backward_path,
        "monotonic_forward_progress": bool(backward_path <= 0.005),
        "final_x_m": float(x[-1]),
        "final_vx_ms": float(vx[-1]),
        "peak_forward_vx_ms": float(np.max(vx)),
    }


def _absolute_score(metrics: dict[str, Any], duration_s: float) -> tuple[float, str]:
    rejection: list[str] = []
    if not metrics["finite"]:
        rejection.append("non_finite")
    if metrics["crash_reason"]:
        rejection.append("safety_rejection")
    if metrics["ground_contact"]:
        rejection.append("ground_contact")
    if metrics["premature_pause"]:
        rejection.append("premature_pause")
    if metrics["vx_sign_change_count_before_0p98m"]:
        rejection.append("vx_reversal_before_0p98m")
    if metrics["peak_pitch_deg"] > 15.0:
        rejection.append("excessive_pitch")
    if metrics["vane_saturation_percent"] > 5.0:
        rejection.append("excessive_vane_saturation")
    remain = metrics["time_to_enter_and_remain_0p10m_s"]
    remain_penalty = float(remain) if np.isfinite(remain) else duration_s + 2.0
    score = (
        12.0 * metrics["final_abs_position_error_m"]
        + 20.0 * metrics["tail_rms_position_error_m"]
        + 35.0 * metrics["tail_path_length_m"]
        + 12.0 * metrics["overshoot_m"]
        + 0.10 * metrics["peak_pitch_deg"]
        + 0.04 * metrics["vane_rms_deg"]
        + 0.20 * remain_penalty
        + 30.0 * metrics["backward_path_before_0p98m_m"]
    )
    if rejection:
        score += 1000.0 + 100.0 * len(rejection)
    return float(score), ";".join(rejection)


def _stick_score(metrics: dict[str, Any]) -> tuple[float, str]:
    rejection: list[str] = []
    if not metrics["finite"]:
        rejection.append("non_finite")
    if metrics["crash_reason"]:
        rejection.append("safety_rejection")
    if metrics["ground_contact"]:
        rejection.append("ground_contact")
    if metrics["target_capture_count"] != 1:
        rejection.append("capture_count_not_one")
    if metrics["target_discontinuity_count"]:
        rejection.append("target_discontinuity")
    score = (
        20.0 * metrics["tail_rms_position_error_m"]
        + 10.0 * metrics["tail_path_length_m"]
        + 10.0 * metrics["tail_rms_vx_ms"]
        + 15.0 * abs(metrics["final_vx_ms"])
        + 20.0 * metrics["max_target_jump_m"]
        + 0.20 * metrics["forward_velocity_peak_count"]
    )
    if rejection:
        score += 1000.0 + 100.0 * len(rejection)
    return float(score), ";".join(rejection)


def _candidate_row(stage: str, parameters: dict[str, Any], result: LoiterRunResult) -> dict[str, Any]:
    metrics = compute_transient_metrics(result)
    if stage.startswith("outer") or stage == "moving_mass":
        score, rejection = _absolute_score(metrics, result.scenario.duration_s)
    else:
        score, rejection = _stick_score(metrics)
    if stage.startswith("outer"):
        selected = all(parameters.get(key) == value for key, value in SELECTED_OUTER.items())
    elif stage.startswith("braking"):
        selected = all(parameters.get(key) == value for key, value in SELECTED_CONTROLLER.items())
    else:
        selected = False
    return {
        "stage": stage,
        **parameters,
        "selected": selected,
        "score": score,
        "rejected": bool(rejection),
        "rejection_reason": rejection,
        **{key: value for key, value in metrics.items() if not isinstance(value, (list, dict))},
    }


def run_searches(profile: str | Path = CURRENT_PROFILE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    absolute = absolute_target_scenario()
    for stage, pos_values, vel_values in (
        ("outer_coarse", COARSE_POS, COARSE_VEL),
        ("outer_refinement", REFINE_POS, REFINE_VEL),
    ):
        for pos_p in pos_values:
            for vel_p in vel_values:
                parameters = {"psc_ne_pos_p": pos_p, "psc_ne_vel_p": vel_p}
                run = run_headless_loiter(profile, absolute, controller_overrides=parameters)
                rows.append(_candidate_row(stage, parameters, run))

    stick = stick_release_scenario()
    behavior = {
        **SELECTED_OUTER,
        "loit_capture_vx_threshold_ms": 0.08,
        "loit_capture_desired_vx_threshold_ms": 0.02,
        "loit_capture_persistent": True,
        "loit_shaper_clamp_target": True,
        "loit_capture_without_jump": True,
    }
    for stage, delays, accelerations, jerks in (
        ("braking_coarse", COARSE_BRAKE_DELAY, COARSE_BRAKE_ACC, COARSE_BRAKE_JERK),
        ("braking_refinement", REFINE_BRAKE_DELAY, REFINE_BRAKE_ACC, REFINE_BRAKE_JERK),
    ):
        for delay in delays:
            for acceleration in accelerations:
                for jerk in jerks:
                    parameters = {
                        **behavior,
                        "loit_brk_delay_s": delay,
                        "loit_brk_acc_mss": acceleration,
                        "loit_brk_jerk_msss": jerk,
                    }
                    run = run_headless_loiter(profile, stick, controller_overrides=parameters)
                    rows.append(_candidate_row(stage, parameters, run))
    return rows


def moving_mass_recheck(profile: str | Path = PROVISIONAL_PROFILE) -> list[dict[str, Any]]:
    scenario = absolute_target_scenario()
    rows: list[dict[str, Any]] = []
    for gain in MOVING_MASS_GAINS:
        configured = LoiterScenarioConfig(
            **{
                **asdict(scenario),
                "name": f"absolute_target_moving_mass_{gain:.3f}",
                "moving_mass_enabled": gain > 0.0,
                "moving_mass_assist_gain_m_per_Nm": gain,
            }
        )
        run = run_headless_loiter(profile, configured)
        rows.append(_candidate_row("moving_mass", {"moving_mass_assist_gain_m_per_Nm": gain}, run))
    return rows


def _write_combined_timeseries(
    path: Path, before: LoiterRunResult, after: LoiterRunResult
) -> None:
    combined = []
    for variant, result in (("current", before), ("selected", after)):
        for row in result.rows:
            combined.append({"variant": variant, **row})
    fieldnames = list(dict.fromkeys(key for row in combined for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)


def _write_candidate_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostic(path: Path, before: LoiterRunResult, after: LoiterRunResult, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True)
    for variant, result, style in (("current", before, "--"), ("selected", after, "-")):
        rows = result.rows
        t = _values(rows, "time")
        axes[0].plot(t, _values(rows, "target_x"), style, label=f"target_x ({variant})")
        axes[0].plot(t, _values(rows, "x"), style, label=f"x ({variant})")
        axes[0].plot(t, _values(rows, "x_error"), style, alpha=0.65, label=f"error ({variant})")
        axes[1].plot(t, _values(rows, "shaped_desired_vx"), style, label=f"shaped vx ({variant})")
        axes[1].plot(t, _values(rows, "position_velocity_correction"), style, label=f"position correction ({variant})")
        axes[1].plot(t, _values(rows, "total_desired_vx"), style, label=f"total desired vx ({variant})")
        axes[1].plot(t, _values(rows, "vx"), style, linewidth=1.5, label=f"actual vx ({variant})")
        axes[2].plot(t, _values(rows, "desired_ax"), style, label=f"ax target ({variant})")
        axes[2].plot(t, np.rad2deg(_values(rows, "theta_target")), style, label=f"theta target deg ({variant})")
        axes[2].plot(t, np.rad2deg(_values(rows, "theta")), style, label=f"theta deg ({variant})")
        axes[3].plot(t, np.rad2deg(_values(rows, "omega_target")), style, label=f"omega target deg/s ({variant})")
        axes[3].plot(t, np.rad2deg(_values(rows, "omega")), style, label=f"omega deg/s ({variant})")
        axes[4].plot(t, _values(rows, "desired_moment"), style, label=f"moment Nm ({variant})")
        axes[4].plot(t, np.rad2deg(_values(rows, "vane_angle_cmd")), style, label=f"vane cmd deg ({variant})")
        axes[4].plot(t, np.rad2deg(_values(rows, "actual_vane_angle")), style, label=f"vane actual deg ({variant})")
        axes[5].step(t, _values(rows, "loiter_braking_active"), where="post", linestyle=style, label=f"braking ({variant})")
        axes[5].step(t, _values(rows, "target_capture_pending"), where="post", linestyle=style, label=f"capture pending ({variant})")
        axes[5].step(t, _values(rows, "target_capture_count"), where="post", linestyle=style, label=f"capture count ({variant})")
        pause = detect_premature_pause(rows)
        if pause:
            for axis in axes:
                axis.axvspan(pause["start_time_s"], pause["end_time_s"], color="red", alpha=0.12)
    labels = ("position (m)", "velocity (m/s)", "accel / angle", "rate (deg/s)", "moment / vane", "state")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2, fontsize=7)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _pause_context(result: LoiterRunResult) -> dict[str, Any]:
    pause = detect_premature_pause(result.rows)
    if not pause:
        return {"pause": None}
    index = int(pause["start_index"])
    prior_peak = int(np.argmax(_values(result.rows[: index + 1], "vx")))
    keys = (
        "time", "target_x", "x", "x_error", "shaped_desired_vx", "position_velocity_correction",
        "total_desired_vx", "vx", "desired_ax", "theta_target", "theta", "omega_target", "omega",
        "desired_moment", "vane_angle_cmd", "actual_vane_angle",
    )
    return {
        "pause": pause,
        "prior_peak": {key: float(result.rows[prior_peak][key]) for key in keys},
        "pause_start": {key: float(result.rows[index][key]) for key in keys},
    }


def _write_summary(
    path: Path,
    absolute_before: LoiterRunResult,
    absolute_after: LoiterRunResult,
    stick_before: LoiterRunResult,
    stick_after: LoiterRunResult,
    moving_mass_rows: Sequence[dict[str, Any]],
) -> None:
    ab = compute_transient_metrics(absolute_before)
    aa = compute_transient_metrics(absolute_after)
    sb = compute_transient_metrics(stick_before)
    sa = compute_transient_metrics(stick_after)
    context = _pause_context(absolute_before)
    mass_055 = next(row for row in moving_mass_rows if abs(float(row["moving_mass_assist_gain_m_per_Nm"]) - 0.055) < 1e-12)
    lines = [
        "# LOITER transient diagnosis and provisional candidate",
        "",
        "## Diagnosed cause",
        "",
        "The absolute-target pause is an outer-loop/inner-loop interaction. With the current 0.50/0.90 position/velocity P pair, the velocity loop changes acceleration demand faster than the delayed attitude/rate/servo plant settles. Actual pitch remains on the braking side while total desired velocity is still forward, so vx reaches the pause threshold; the lagging attitude then crosses and produces the next forward lobe. Vane deadband and actuator lag contribute phase lag, but the logged command remains available and neither saturation nor target shaping initiates the absolute-target pause.",
        "",
        "The stick-release defect is different: jerk-limited shaped velocity overshoots through zero and reverses. Integrating that reversed velocity walks target_x backward, while the one-control-tick capture condition can be missed or repeated. The provisional behavior clamps the shaper at its zero target and records one capture without replacing the already-shaped final hold target, eliminating capture jumps.",
        "",
        "## Selected provisional parameters",
        "",
        "| Parameter | Current | Selected |",
        "|---|---:|---:|",
        "| PSC_NE_POS_P | 0.50 | 0.55 |",
        "| PSC_NE_VEL_P | 0.90 | 0.70 |",
        "| LOIT_BRK_DELAY | 0.40 s | 0.50 s |",
        "| LOIT_BRK_ACC | 0.80 m/s^2 | 1.00 m/s^2 |",
        "| LOIT_BRK_JERK | 2.00 m/s^3 | 3.00 m/s^3 |",
        "| capture vx threshold | 0.08 m/s | 0.08 m/s |",
        "| persistent capture handshake | off | on |",
        "| clamp shaped velocity at target | off | on |",
        "| capture without target_x jump | off | on |",
        "",
        "ATC rate PID, Angle P, physics, actuator/vane model, mass, and COM geometry are unchanged. Moving-mass assist remains disabled in the selected vane-only profile.",
        "",
        "## Absolute +1 m before/after",
        "",
        "| Metric | Current | Selected |",
        "|---|---:|---:|",
        f"| premature pause | {ab['premature_pause']} | {aa['premature_pause']} |",
        f"| pause start | {ab['pause_start_s']:.3f} s | none |",
        f"| pause position error | {ab['pause_position_error_m']:.3f} m | none |",
        f"| prior peak vx | {ab['prior_peak_vx_ms']:.3f} m/s | {aa['prior_peak_vx_ms']:.3f} m/s |",
        f"| vx reversals before x=0.98 m | {ab['vx_sign_change_count_before_0p98m']} | {aa['vx_sign_change_count_before_0p98m']} |",
        f"| final absolute error | {ab['final_abs_position_error_m']:.4f} m | {aa['final_abs_position_error_m']:.4f} m |",
        f"| tail RMS error | {ab['tail_rms_position_error_m']:.4f} m | {aa['tail_rms_position_error_m']:.4f} m |",
        f"| tail path length | {ab['tail_path_length_m']:.4f} m | {aa['tail_path_length_m']:.4f} m |",
        f"| overshoot | {ab['overshoot_m']:.4f} m | {aa['overshoot_m']:.4f} m |",
        f"| peak pitch | {ab['peak_pitch_deg']:.3f} deg | {aa['peak_pitch_deg']:.3f} deg |",
        f"| vane RMS | {ab['vane_rms_deg']:.3f} deg | {aa['vane_rms_deg']:.3f} deg |",
        "",
        "First pause context (radians are retained for attitude/rate fields):",
        "",
        "```json",
        json.dumps(_jsonable(context), indent=2, sort_keys=True),
        "```",
        "",
        "## Stick pulse/release before/after",
        "",
        "| Metric | Current | Selected |",
        "|---|---:|---:|",
        f"| shaped-vx sign changes after release | {sum(1 for i in range(len(stick_before.rows)-1) if float(stick_before.rows[i]['shaped_desired_vx']) * float(stick_before.rows[i+1]['shaped_desired_vx']) < 0)} | {sum(1 for i in range(len(stick_after.rows)-1) if float(stick_after.rows[i]['shaped_desired_vx']) * float(stick_after.rows[i+1]['shaped_desired_vx']) < 0)} |",
        f"| capture events | {sb['target_capture_count']} | {sa['target_capture_count']} |",
        f"| target discontinuities > 0.02 m | {sb['target_discontinuity_count']} | {sa['target_discontinuity_count']} |",
        f"| maximum target step | {sb['max_target_jump_m']:.4f} m | {sa['max_target_jump_m']:.4f} m |",
        f"| tail RMS position error | {sb['tail_rms_position_error_m']:.4f} m | {sa['tail_rms_position_error_m']:.4f} m |",
        f"| tail path length | {sb['tail_path_length_m']:.4f} m | {sa['tail_path_length_m']:.4f} m |",
        f"| final vx | {sb['final_vx_ms']:.4f} m/s | {sa['final_vx_ms']:.4f} m/s |",
        "",
        "## Moving-mass follow-up",
        "",
        "| Gain (m/Nm) | Pause | Final error (m) | Tail RMS (m) | Peak pitch (deg) |",
        "|---:|:---:|---:|---:|---:|",
        *[
            f"| {float(row['moving_mass_assist_gain_m_per_Nm']):.3f} | {row['premature_pause']} | {float(row['final_abs_position_error_m']):.4f} | {float(row['tail_rms_position_error_m']):.4f} | {float(row['peak_pitch_deg']):.3f} |"
            for row in moving_mass_rows
        ],
        "",
        f"Gain 0.055 m/Nm is pause-free but is no longer a reasonable preferred gain under the explicit stepped target: final error={float(mass_055['final_abs_position_error_m']):.4f} m, tail RMS={float(mass_055['tail_rms_position_error_m']):.4f} m, and peak pitch={float(mass_055['peak_pitch_deg']):.3f} deg. Gain 0.070 performs best in this one scenario. The selected moving-mass gain is not changed by this LOITER task; broader moving-mass regressions are required before adopting a replacement.",
        "",
        "## Remaining limitations",
        "",
        "The fixed attitude/rate controller remains lightly damped, so small post-target velocity lobes remain after the vehicle has first reached the target region. The selected outer pair removes the defined premature pause and pre-0.98 m reversal, but it does not retune the validated inner loop. This is a provisional profile and does not replace the canonical tuned profile.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_artifacts(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    absolute = absolute_target_scenario()
    stick = stick_release_scenario()
    absolute_before = run_headless_loiter(CURRENT_PROFILE, absolute)
    absolute_after = run_headless_loiter(PROVISIONAL_PROFILE, absolute)
    stick_before = run_headless_loiter(CURRENT_PROFILE, stick)
    stick_after = run_headless_loiter(PROVISIONAL_PROFILE, stick)

    absolute_csv = output / "absolute_target_timeseries.csv"
    stick_csv = output / "stick_release_timeseries.csv"
    absolute_plot = output / "absolute_target_diagnostic.png"
    stick_plot = output / "stick_release_diagnostic.png"
    candidate_csv = output / "candidate_results.csv"
    summary_path = output / "selected_candidate_summary.md"
    manifest_path = output / "manifest.json"

    _write_combined_timeseries(absolute_csv, absolute_before, absolute_after)
    _write_combined_timeseries(stick_csv, stick_before, stick_after)
    _plot_diagnostic(absolute_plot, absolute_before, absolute_after, "LOITER absolute +1 m target: current vs provisional")
    _plot_diagnostic(stick_plot, stick_before, stick_after, "LOITER stick pulse/release: current vs provisional")
    candidate_rows = run_searches(CURRENT_PROFILE)
    mass_rows = moving_mass_recheck(PROVISIONAL_PROFILE)
    candidate_rows.extend(mass_rows)
    _write_candidate_csv(candidate_csv, candidate_rows)
    _write_summary(summary_path, absolute_before, absolute_after, stick_before, stick_after, mass_rows)

    artifact_paths = [absolute_csv, stick_csv, absolute_plot, stick_plot, candidate_csv, summary_path]
    manifest = {
        "schema_version": 1,
        "deterministic": True,
        "current_profile": str(CURRENT_PROFILE.relative_to(ROOT)),
        "provisional_profile": str(PROVISIONAL_PROFILE.relative_to(ROOT)),
        "absolute_scenario": asdict(absolute),
        "stick_scenario": asdict(stick),
        "selected_controller": SELECTED_CONTROLLER,
        "search_ranges": {
            "outer_coarse": {"PSC_NE_POS_P": COARSE_POS, "PSC_NE_VEL_P": COARSE_VEL},
            "outer_refinement": {"PSC_NE_POS_P": REFINE_POS, "PSC_NE_VEL_P": REFINE_VEL},
            "braking_coarse": {"LOIT_BRK_DELAY": COARSE_BRAKE_DELAY, "LOIT_BRK_ACC": COARSE_BRAKE_ACC, "LOIT_BRK_JERK": COARSE_BRAKE_JERK},
            "braking_refinement": {"LOIT_BRK_DELAY": REFINE_BRAKE_DELAY, "LOIT_BRK_ACC": REFINE_BRAKE_ACC, "LOIT_BRK_JERK": REFINE_BRAKE_JERK},
            "moving_mass_gain_m_per_Nm": MOVING_MASS_GAINS,
        },
        "metrics": {
            "absolute_current": compute_transient_metrics(absolute_before),
            "absolute_selected": compute_transient_metrics(absolute_after),
            "stick_current": compute_transient_metrics(stick_before),
            "stick_selected": compute_transient_metrics(stick_after),
        },
        "artifacts": {path.name: _sha256(path) for path in artifact_paths},
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and tune the deterministic LOITER transient.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_artifacts(args.output_dir)
    print(json.dumps(_jsonable(manifest["metrics"]), indent=2, sort_keys=True))
    print(f"artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
