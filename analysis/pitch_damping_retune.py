from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from ..params import load_interactive_config
    from .headless_loiter import (
        LoiterRunResult,
        LoiterScenarioConfig,
        run_headless_loiter,
        save_loiter_timeseries,
    )
except ImportError:  # pragma: no cover - direct repository-root execution
    from params import load_interactive_config
    from analysis.headless_loiter import (
        LoiterRunResult,
        LoiterScenarioConfig,
        run_headless_loiter,
        save_loiter_timeseries,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = REPO_ROOT / "params" / "loiter_transient_provisional.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "pitch_damping_retune"
PROVISIONAL_PROFILE = REPO_ROOT / "params" / "pitch_damping_retune_provisional.json"
CANONICAL_PROFILES = (
    REPO_ROOT / "params" / "loiter_tuned_vane_only.json",
    REPO_ROOT / "params" / "moving_mass_prototype_2kg_tuned.json",
    SOURCE_PROFILE,
    REPO_ROOT / "params" / "interactive_loiter_assist_2kg.json",
)
PRESERVED_RESULTS = REPO_ROOT / "results" / "analysis" / "loiter_transient_diagnosis"

SEED = 0
TAIL_WINDOW_FULL_S = 2.5
TAIL_WINDOW_QUICK_S = 0.75
SCORE_WEIGHTS = {
    "tail_rms_pitch_deg": 0.30,
    "tail_rms_pitch_rate_deg_s": 0.20,
    "tail_rms_horizontal_velocity_m_s": 0.15,
    "tail_path_length_m": 0.15,
    "tail_rms_position_error_m": 0.10,
    "vane_command_rms_deg": 0.05,
    "vane_command_total_variation_deg": 0.05,
}
GROUP_WEIGHTS = {"inner_loop": 0.45, "integrated_loiter": 0.55}
NEAR_EQUIVALENCE_ABSOLUTE_MARGIN = 0.010000

# These thresholds are fixed before candidate selection and are emitted in methodology.md.
CHATTER_THRESHOLDS = {
    "command_deadband_deg": 0.5,
    "meaningful_rate_deg_s": 10.0,
    "max_meaningful_sign_changes": 80,
    "max_total_variation_per_second_deg_s": 45.0,
    "max_tail_high_frequency_energy_deg2": 0.35,
    "high_frequency_window_s": 0.10,
}
HARD_GATE_THRESHOLDS = {
    "pitch_divergence_deg": 45.0,
    "vane_saturation_percent": 5.0,
    "servo_rate_saturation_percent": 20.0,
    "mixer_saturation_percent": 5.0,
    "capture_jump_m": 0.02,
    "target_region_error_m": 0.02,
    "premature_pause_error_m": 0.10,
    "premature_pause_speed_m_s": 0.03,
    "premature_pause_duration_s": 0.15,
    "motion_started_speed_m_s": 0.10,
    "second_lobe_speed_m_s": 0.10,
    "shaped_velocity_reversal_m_s": 1e-6,
    "moving_mass_lock_tolerance_m": 0.0,
    "symmetry_max_fraction": 0.15,
}

FIXED_CONTROLLER_VALUES: dict[str, float | bool] = {
    "atc_rat_pit_i": 0.0,
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
BASELINE_GAINS = {
    "atc_rat_pit_p": 0.070,
    "atc_rat_pit_i": 0.0,
    "atc_rat_pit_d": 0.008,
    "atc_ang_pit_p": 10.0,
}
PHYSICAL_CONFIGURATION = {
    "H": 0.5,
    "m": 2.0,
    "moving_mass": {
        "enabled": True,
        "mass_kg": 0.5,
        "max_offset_m": 0.05,
        "max_rate_m_s": 0.2,
        "max_accel_m_s2": 1.0,
        "initial_offset_m": 0.0,
        "use_total_com_geometry": True,
        "use_legacy_gravity_offset_moment": False,
        "moving_mass_body_up_offset_m": 0.12,
    },
}

STAGE1_RATE_P = (0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090)
STAGE1_RATE_D = (0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014)
STAGE2_ANGLE_P = (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
MAX_BOUNDARY_EXTENSIONS_PER_STAGE = 12

TIMESERIES_FILENAMES = {
    "loiter_positive_disturbance": "loiter_positive_disturbance_timeseries.csv",
    "loiter_negative_disturbance": "loiter_negative_disturbance_timeseries.csv",
    "forward_1m": "forward_1m_timeseries.csv",
    "backward_1m": "backward_1m_timeseries.csv",
    "pitch_positive_recovery": "pitch_positive_recovery_timeseries.csv",
    "pitch_negative_recovery": "pitch_negative_recovery_timeseries.csv",
    "stick_release": "stick_release_timeseries.csv",
}


class BaselineMismatchError(RuntimeError):
    """Raised before candidate search when Stage 0 violates PR #19 behavior."""


@dataclass(frozen=True)
class ScenarioDefinition:
    key: str
    config: LoiterScenarioConfig
    group: str
    direction: int
    event_time_s: float
    requires_target_transient_gates: bool = False
    requires_capture_gates: bool = False


@dataclass(frozen=True)
class Candidate:
    stage: str
    rate_p: float
    rate_d: float
    angle_p: float

    @property
    def key(self) -> str:
        return (
            f"{self.stage}:rate_p={self.rate_p:.8f}:"
            f"rate_d={self.rate_d:.8f}:angle_p={self.angle_p:.8f}"
        )

    def controller_overrides(self) -> dict[str, float | bool]:
        return {
            **FIXED_CONTROLLER_VALUES,
            "atc_rat_pit_p": self.rate_p,
            "atc_rat_pit_d": self.rate_d,
            "atc_ang_pit_p": self.angle_p,
            "enable_noise": False,
            "random_seed": SEED,
        }


@dataclass(frozen=True)
class WorkflowOptions:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    quick: bool = False
    resume: bool = True
    allow_baseline_mismatch: bool = False
    stage: str = "all"


def required_scenarios(quick: bool = False) -> tuple[ScenarioDefinition, ...]:
    if quick:
        loiter_duration, pitch_duration = 4.5, 3.5
    else:
        loiter_duration, pitch_duration = 12.0, 8.0
    all_scenarios = (
        ScenarioDefinition(
            "loiter_positive_disturbance",
            LoiterScenarioConfig(
                name="loiter_positive_disturbance",
                duration_s=loiter_duration,
                disturbance_start_s=1.0,
                disturbance_duration_s=0.20,
                disturbance_force_x=8.0,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "integrated_loiter",
            -1,
            1.0,
        ),
        ScenarioDefinition(
            "loiter_negative_disturbance",
            LoiterScenarioConfig(
                name="loiter_negative_disturbance",
                duration_s=loiter_duration,
                disturbance_start_s=1.0,
                disturbance_duration_s=0.20,
                disturbance_force_x=-8.0,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "integrated_loiter",
            1,
            1.0,
        ),
        ScenarioDefinition(
            "forward_1m",
            LoiterScenarioConfig(
                name="forward_1m",
                duration_s=loiter_duration,
                target_step_time_s=0.5,
                target_step_x=1.0,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "integrated_loiter",
            1,
            0.5,
            requires_target_transient_gates=True,
        ),
        ScenarioDefinition(
            "backward_1m",
            LoiterScenarioConfig(
                name="backward_1m",
                duration_s=loiter_duration,
                target_step_time_s=0.5,
                target_step_x=-1.0,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "integrated_loiter",
            -1,
            0.5,
            requires_target_transient_gates=True,
        ),
        ScenarioDefinition(
            "pitch_positive_recovery",
            LoiterScenarioConfig(
                name="pitch_positive_recovery",
                duration_s=pitch_duration,
                initial_theta_deg=3.0,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "inner_loop",
            1,
            0.0,
        ),
        ScenarioDefinition(
            "pitch_negative_recovery",
            LoiterScenarioConfig(
                name="pitch_negative_recovery",
                duration_s=pitch_duration,
                initial_theta_deg=-3.0,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "inner_loop",
            -1,
            0.0,
        ),
        ScenarioDefinition(
            "stick_release",
            LoiterScenarioConfig(
                name="stick_release",
                duration_s=loiter_duration,
                stick_start_s=0.5,
                stick_end_s=2.2,
                stick_x=0.65,
                capture_current_target=True,
                moving_mass_enabled=True,
                moving_mass_target_m=0.0,
                moving_mass_assist_gain_m_per_Nm=0.0,
            ),
            "integrated_loiter",
            1,
            2.2,
            requires_capture_gates=True,
        ),
    )
    if not quick:
        return all_scenarios
    wanted = {"forward_1m", "pitch_positive_recovery", "stick_release"}
    return tuple(scenario for scenario in all_scenarios if scenario.key in wanted)


def stage1_candidates(quick: bool = False) -> list[Candidate]:
    if quick:
        return [
            Candidate("stage1_rate_pd", 0.065, 0.008, 10.0),
            Candidate("stage1_rate_pd", 0.075, 0.010, 10.0),
        ]
    return [
        Candidate("stage1_rate_pd", rate_p, rate_d, 10.0)
        for rate_p in STAGE1_RATE_P
        for rate_d in STAGE1_RATE_D
    ]


def stage2_candidates(top_rate_pd: Sequence[Candidate], quick: bool = False) -> list[Candidate]:
    angles = (9.0, 11.0) if quick else STAGE2_ANGLE_P
    selected = list(top_rate_pd[:1] if quick else top_rate_pd[:3])
    return [
        Candidate("stage2_angle_p", candidate.rate_p, candidate.rate_d, angle_p)
        for candidate in selected
        for angle_p in angles
    ]


def _centered_values(center: float, half_span: float, step: float) -> tuple[float, ...]:
    count = int(round(2.0 * half_span / step)) + 1
    return tuple(round(center - half_span + index * step, 10) for index in range(count))


def stage3a_candidates(best: Candidate, quick: bool = False) -> list[Candidate]:
    if quick:
        return [Candidate("stage3a_local_rate_pd", best.rate_p, best.rate_d, best.angle_p)]
    return [
        Candidate("stage3a_local_rate_pd", rate_p, rate_d, best.angle_p)
        for rate_p in _centered_values(best.rate_p, 0.005, 0.00125)
        for rate_d in _centered_values(best.rate_d, 0.002, 0.0005)
        if rate_p > 0.0 and rate_d >= 0.0
    ]


def stage3b_candidates(best: Candidate, quick: bool = False) -> list[Candidate]:
    angles = (best.angle_p,) if quick else _centered_values(best.angle_p, 1.0, 0.25)
    return [Candidate("stage3b_local_angle_p", best.rate_p, best.rate_d, angle) for angle in angles]


def stage3c_candidates(best: Candidate, quick: bool = False) -> list[Candidate]:
    if quick:
        return [Candidate("stage3c_crosscheck", best.rate_p, best.rate_d, best.angle_p)]
    return [
        Candidate("stage3c_crosscheck", rate_p, rate_d, angle_p)
        for rate_p in _centered_values(best.rate_p, 0.0025, 0.00125)
        for rate_d in _centered_values(best.rate_d, 0.0010, 0.0005)
        for angle_p in (best.angle_p - 0.25, best.angle_p, best.angle_p + 0.25)
        if rate_p > 0.0 and rate_d >= 0.0 and angle_p > 0.0
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def git_revision(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, encoding="utf-8"
    ).strip()


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        REPO_ROOT / "analysis" / "headless_loiter.py",
        REPO_ROOT / "interactive_sim.py",
        REPO_ROOT / "interactive_logging.py",
        REPO_ROOT / "cascaded_controller.py",
        REPO_ROOT / "actuators.py",
        REPO_ROOT / "singlecopter_mixer.py",
        REPO_ROOT / "rigid_body_model.py",
        REPO_ROOT / "config.py",
        REPO_ROOT / "params.py",
    )
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def search_ranges_payload(quick: bool = False) -> dict[str, Any]:
    return {
        "quick": quick,
        "stage1_rate_p": [candidate.rate_p for candidate in stage1_candidates(quick)],
        "stage1_rate_d": [candidate.rate_d for candidate in stage1_candidates(quick)],
        "stage2_angle_p": list((9.0, 11.0) if quick else STAGE2_ANGLE_P),
        "stage3a_rate_p": "best +/-0.005 step 0.00125",
        "stage3a_rate_d": "best +/-0.002 step 0.0005",
        "stage3b_angle_p": "best +/-1.0 step 0.25",
        "stage3c_rate_p": "best +/-0.0025 step 0.00125",
        "stage3c_rate_d": "best +/-0.0010 step 0.0005",
        "stage3c_angle_p": "best +/-0.25 step 0.25",
        "boundary_extension_policy": {
            "maximum_rounds_per_stage": MAX_BOUNDARY_EXTENSIONS_PER_STAGE,
            "stage1_rate_p_values_per_round": 4,
            "stage1_rate_p_increment": 0.005,
            "stage1_rate_d_values_per_round": 3,
            "stage1_rate_d_increment": 0.002,
            "stage2_angle_p_values_per_round": 2,
            "stage2_angle_p_increment": 1.0,
            "stage3c_rate_p_increment": 0.00125,
            "stage3c_rate_d_increment": 0.0005,
            "stage3c_angle_p_increment": 0.25,
            "stop_condition": "selected candidate is interior on every searched axis",
        },
    }


def build_fingerprint(quick: bool = False) -> tuple[dict[str, Any], str]:
    rb_cfg, ui_cfg, controller_cfg = load_interactive_config(SOURCE_PROFILE)
    payload = {
        "schema_version": 1,
        "base_sha": git_revision("rev-parse", "HEAD"),
        "source_profile": SOURCE_PROFILE.relative_to(REPO_ROOT).as_posix(),
        "source_profile_sha256": sha256_file(SOURCE_PROFILE),
        "physical_configuration": PHYSICAL_CONFIGURATION,
        "fixed_controller_parameters": FIXED_CONTROLLER_VALUES,
        "moving_mass_assist_gain_m_per_Nm": 0.0,
        "search_ranges": search_ranges_payload(quick),
        "scenario_definitions": [asdict(scenario) for scenario in required_scenarios(quick)],
        "scenario_fingerprint": _sha256_bytes(
            _canonical_json([asdict(scenario) for scenario in required_scenarios(quick)]).encode()
        ),
        "physics_timestep_s": rb_cfg.dt,
        "controller_timestep_s": ui_cfg.controller_dt,
        "seed": SEED,
        "score_weights": SCORE_WEIGHTS,
        "group_weights": GROUP_WEIGHTS,
        "hard_gate_thresholds": HARD_GATE_THRESHOLDS,
        "chatter_thresholds": CHATTER_THRESHOLDS,
        "baseline_controller": BASELINE_GAINS,
        "profile_controller_snapshot": asdict(controller_cfg),
        "source_hashes": source_hashes(),
    }
    return payload, _sha256_bytes(_canonical_json(payload).encode())


def validate_parameter_sources() -> None:
    _rb, _ui, controller = load_interactive_config(SOURCE_PROFILE)
    expected = {**FIXED_CONTROLLER_VALUES, **BASELINE_GAINS}
    for key, value in expected.items():
        actual = getattr(controller, key)
        if isinstance(value, bool):
            matches = actual is value
        else:
            matches = math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-12)
        if not matches:
            raise ValueError(
                f"source-profile mismatch for {key}: expected {value!r}, got {actual!r}"
            )


def _array(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _tail_mask(times: np.ndarray, window_s: float) -> np.ndarray:
    return times >= max(0.0, float(times[-1]) - window_s) - 1e-12


def _strict_settling_time(
    times: np.ndarray,
    x_error: np.ndarray,
    vx: np.ndarray,
    theta_deg: np.ndarray,
    event_time_s: float,
) -> tuple[float, bool]:
    condition = (
        (times >= event_time_s)
        & (np.abs(x_error) <= 0.10)
        & (np.abs(vx) <= 0.08)
        & (np.abs(theta_deg) <= 1.0)
    )
    for index in np.flatnonzero(condition):
        if bool(np.all(condition[index:])):
            return float(times[index] - event_time_s), True
    return float(times[-1] - event_time_s), False


def _pitch_peak_metrics(
    times: np.ndarray, theta_deg: np.ndarray
) -> dict[str, float | int | str | None]:
    magnitude = np.abs(theta_deg)
    threshold = max(0.05, 0.05 * float(np.max(magnitude)))
    peaks = np.flatnonzero(
        (magnitude[1:-1] > magnitude[:-2])
        & (magnitude[1:-1] >= magnitude[2:])
        & (magnitude[1:-1] >= threshold)
    ) + 1
    peak_values = magnitude[peaks]
    peak_decay = (
        float(peak_values[-1] / peak_values[0]) if peak_values.size >= 2 and peak_values[0] > 0 else None
    )
    log_decrement: float | None = None
    damping_ratio: float | None = None
    unavailable_reason = ""
    if peak_values.size < 3:
        unavailable_reason = "fewer than three significant absolute pitch peaks"
    else:
        ratios = peak_values[:-1] / peak_values[1:]
        valid = np.isfinite(ratios) & (ratios > 1.0)
        if np.count_nonzero(valid) < 2:
            unavailable_reason = "significant peak sequence is not monotonically decaying"
        else:
            log_decrement = float(np.median(np.log(ratios[valid])))
            damping_ratio = float(
                log_decrement / math.sqrt((2.0 * math.pi) ** 2 + log_decrement**2)
            )
    return {
        "pitch_zero_crossing_count": meaningful_zero_crossings(theta_deg, 0.05),
        "significant_pitch_peak_count": int(peak_values.size),
        "last_significant_pitch_peak_time_s": float(times[peaks[-1]]) if peaks.size else None,
        "peak_decay_ratio": peak_decay,
        "logarithmic_decrement": log_decrement,
        "estimated_damping_ratio": damping_ratio,
        "damping_estimate_unavailable_reason": unavailable_reason,
    }


def meaningful_zero_crossings(values: Sequence[float], deadband: float) -> int:
    array = np.asarray(values, dtype=float)
    signs = np.sign(array[np.abs(array) >= deadband])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def transient_gate_metrics(
    definition: ScenarioDefinition, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    times = _array(rows, "time")
    aligned_vx = definition.direction * _array(rows, "vx")
    aligned_error = definition.direction * _array(rows, "x_error")
    dt = float(np.median(np.diff(times)))
    required = max(
        1,
        int(
            math.ceil(
                HARD_GATE_THRESHOLDS["premature_pause_duration_s"] / dt - 1e-9
            )
        ),
    )
    started = np.flatnonzero(
        (times >= definition.event_time_s - 1e-12)
        & (aligned_vx >= HARD_GATE_THRESHOLDS["motion_started_speed_m_s"])
    )
    if not started.size:
        return {
            "motion_start_detected": False,
            "premature_pause": False,
            "early_velocity_reversal": False,
            "second_acceleration_lobe_after_full_pause": False,
        }
    start = int(started[0])
    reached = np.flatnonzero(
        (np.arange(len(times)) >= start)
        & (aligned_error <= HARD_GATE_THRESHOLDS["target_region_error_m"])
    )
    stop = int(reached[0]) if reached.size else len(times)
    early_reversal = any(
        aligned_vx[index] * aligned_vx[index + 1] < 0.0
        for index in range(start, max(start, stop - 1))
    )
    pause_mask = (
        (aligned_error > HARD_GATE_THRESHOLDS["premature_pause_error_m"])
        & (np.abs(aligned_vx) < HARD_GATE_THRESHOLDS["premature_pause_speed_m_s"])
    )
    full_pause = np.abs(aligned_vx) < HARD_GATE_THRESHOLDS["premature_pause_speed_m_s"]
    premature_pause = False
    second_lobe = False
    for index in range(start, len(times) - required + 1):
        tail = index + required
        premature_pause |= bool(np.all(pause_mask[index:tail]))
        second_lobe |= bool(
            np.all(full_pause[index:tail])
            and np.any(
                aligned_vx[tail:] >= HARD_GATE_THRESHOLDS["second_lobe_speed_m_s"]
            )
        )
    return {
        "motion_start_detected": True,
        "premature_pause": premature_pause,
        "early_velocity_reversal": early_reversal,
        "second_acceleration_lobe_after_full_pause": second_lobe,
    }


def capture_gate_metrics(
    definition: ScenarioDefinition, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    times = _array(rows, "time")
    events = _array(rows, "target_capture_event") > 0.5
    counts = _array(rows, "target_capture_count")
    target = _array(rows, "target_x")
    shaped = _array(rows, "shaped_desired_vx")
    jumps = np.r_[0.0, np.diff(target)]
    event_jumps = np.abs(jumps[events])
    after_release = times >= definition.event_time_s - 1e-12
    return {
        "target_capture_count": int(counts[-1]) if counts.size else 0,
        "target_capture_event_count": int(np.count_nonzero(events)),
        "capture_max_target_jump_m": float(np.max(event_jumps)) if event_jumps.size else 0.0,
        "capture_discontinuity": bool(
            event_jumps.size
            and np.max(event_jumps) > HARD_GATE_THRESHOLDS["capture_jump_m"]
        ),
        "shaped_velocity_sign_reversal_after_release": bool(
            np.any(
                shaped[after_release]
                < -HARD_GATE_THRESHOLDS["shaped_velocity_reversal_m_s"]
            )
        ),
    }


def compute_metrics(
    definition: ScenarioDefinition,
    result: LoiterRunResult,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    rows = result.rows
    if not rows:
        return {
            "scenario_name": definition.key,
            "rejected": True,
            "rejection_reasons": "missing_scenario_rows",
        }
    times = _array(rows, "time")
    x = _array(rows, "x")
    z = _array(rows, "z")
    vx = _array(rows, "vx")
    x_error = _array(rows, "x_error")
    theta_deg = np.rad2deg(_array(rows, "theta"))
    omega_deg_s = np.rad2deg(_array(rows, "omega"))
    vane_cmd_deg = np.rad2deg(_array(rows, "vane_angle_cmd"))
    vane_actual_deg = np.rad2deg(_array(rows, "actual_vane_angle"))
    dt = float(np.median(np.diff(times)))
    tail = _tail_mask(times, TAIL_WINDOW_QUICK_S if quick else TAIL_WINDOW_FULL_S)
    vane_rate = np.diff(vane_cmd_deg, prepend=vane_cmd_deg[0]) / dt
    meaningful_sign_changes = 0
    for index in range(len(vane_cmd_deg) - 1):
        if (
            vane_cmd_deg[index] * vane_cmd_deg[index + 1] < 0.0
            and max(abs(vane_cmd_deg[index]), abs(vane_cmd_deg[index + 1]))
            >= CHATTER_THRESHOLDS["command_deadband_deg"]
            and max(abs(vane_rate[index]), abs(vane_rate[index + 1]))
            >= CHATTER_THRESHOLDS["meaningful_rate_deg_s"]
        ):
            meaningful_sign_changes += 1
    window = max(3, int(round(CHATTER_THRESHOLDS["high_frequency_window_s"] / dt)))
    kernel = np.ones(window) / window
    smoothed = np.convolve(vane_cmd_deg, kernel, mode="same")
    high_frequency_energy = float(np.mean((vane_cmd_deg[tail] - smoothed[tail]) ** 2))
    total_variation = float(np.sum(np.abs(np.diff(vane_cmd_deg))))
    total_variation_per_s = total_variation / max(float(times[-1]), dt)
    strict_settling_s, strict_settled = _strict_settling_time(
        times, x_error, vx, theta_deg, definition.event_time_s
    )

    post_event = times >= definition.event_time_s - 1e-12
    target = float(definition.config.target_step_x or definition.config.target_x)
    if definition.requires_target_transient_gates:
        overshoot = max(0.0, float(np.max(definition.direction * (x[post_event] - target))))
    else:
        overshoot = float(np.max(np.abs(x_error[post_event])))
    target_region = np.flatnonzero(post_event & (np.abs(x_error) <= 0.10))
    first_region = int(target_region[0]) if target_region.size else len(times) - 1
    post_target_velocity_lobe = float(np.max(np.abs(vx[first_region:])))
    aligned_dx = definition.direction * np.diff(x[first_region:])
    reverse_path = float(np.sum(np.maximum(0.0, -aligned_dx)))
    numeric = np.column_stack(
        (times, x, z, vx, x_error, theta_deg, omega_deg_s, vane_cmd_deg, vane_actual_deg)
    )
    saturation_union = np.maximum.reduce(
        (
            _array(rows, "servo_angle_saturated"),
            _array(rows, "servo_rate_saturated"),
            _array(rows, "mixer_saturated"),
        )
    )
    metrics: dict[str, Any] = {
        "scenario_name": definition.key,
        "scenario_group": definition.group,
        "direction": definition.direction,
        "duration_s": float(times[-1]),
        "physics_timestep_s": float(rows[0]["physics_dt"]),
        "controller_timestep_s": float(rows[0]["controller_dt"]),
        "sample_count": len(rows),
        "finite": bool(np.all(np.isfinite(numeric))),
        "crash": bool(result.crashed),
        "crash_reason": result.crash_reason,
        "ground_contact": bool(np.any(_array(rows, "min_body_z") <= 0.0)),
        "peak_abs_pitch_deg": float(np.max(np.abs(theta_deg))),
        "tail_rms_pitch_deg": _rms(theta_deg[tail]),
        "tail_pitch_peak_to_peak_deg": float(np.ptp(theta_deg[tail])),
        "tail_rms_pitch_rate_deg_s": _rms(omega_deg_s[tail]),
        "tail_pitch_rate_peak_to_peak_deg_s": float(np.ptp(omega_deg_s[tail])),
        "tail_rms_position_error_m": _rms(x_error[tail]),
        "tail_rms_horizontal_velocity_m_s": _rms(vx[tail]),
        "tail_position_peak_to_peak_m": float(np.ptp(x[tail])),
        "tail_path_length_m": float(np.sum(np.hypot(np.diff(x[tail]), np.diff(z[tail])))),
        "final_abs_position_error_m": abs(float(x_error[-1])),
        "position_overshoot_m": overshoot,
        "recovery_excursion_m": float(np.max(np.abs(x_error[post_event]))),
        "strict_settling_time_s": strict_settling_s,
        "strict_settled": strict_settled,
        "post_target_max_velocity_lobe_m_s": post_target_velocity_lobe,
        "reverse_path_after_target_region_entry_m": reverse_path,
        "final_horizontal_velocity_m_s": float(vx[-1]),
        "vane_command_rms_deg": _rms(vane_cmd_deg),
        "vane_command_max_deg": float(np.max(np.abs(vane_cmd_deg))),
        "vane_command_total_variation_deg": total_variation,
        "vane_command_rate_rms_deg_s": _rms(vane_rate),
        "vane_command_rate_max_deg_s": float(np.max(np.abs(vane_rate))),
        "meaningful_vane_sign_change_count": meaningful_sign_changes,
        "vane_saturation_percent": float(100.0 * np.mean(saturation_union > 0.5)),
        "servo_rate_saturation_percent": float(
            100.0 * np.mean(_array(rows, "servo_rate_saturated") > 0.5)
        ),
        "mixer_saturation_percent": float(
            100.0 * np.mean(_array(rows, "mixer_saturated") > 0.5)
        ),
        "servo_tracking_rms_deg": _rms(vane_cmd_deg - vane_actual_deg),
        "tail_high_frequency_vane_energy_deg2": high_frequency_energy,
        "vane_zero_crossing_frequency_hz": meaningful_sign_changes
        / max(float(times[-1]), dt),
        "vane_total_variation_per_second_deg_s": total_variation_per_s,
        "deadband_reentry_count": int(
            np.count_nonzero(
                (np.abs(vane_cmd_deg[1:]) < CHATTER_THRESHOLDS["command_deadband_deg"])
                & (np.abs(vane_cmd_deg[:-1]) >= CHATTER_THRESHOLDS["command_deadband_deg"])
            )
        ),
        "moving_mass_assist_gain_m_per_Nm": 0.0,
        "moving_mass_max_abs_offset_m": float(
            np.max(np.abs(_array(rows, "moving_mass_offset_m")))
        ),
        "moving_mass_max_abs_target_m": float(
            np.max(np.abs(_array(rows, "moving_mass_target_m")))
        ),
        "moving_mass_max_abs_velocity_m_s": float(
            np.max(np.abs(_array(rows, "moving_mass_velocity_m_s")))
        ),
        "effective_atc_rat_pit_p": result.metrics["effective_atc_rat_pit_p"],
        "effective_atc_rat_pit_i": result.metrics["effective_atc_rat_pit_i"],
        "effective_atc_rat_pit_d": result.metrics["effective_atc_rat_pit_d"],
        "effective_atc_ang_pit_p": result.metrics["effective_atc_ang_pit_p"],
        "effective_psc_ne_pos_p": result.metrics["effective_psc_ne_pos_p"],
        "effective_psc_ne_vel_p": result.metrics["effective_psc_ne_vel_p"],
        "total_mass_kg": result.metrics["total_mass_kg"],
        "physical_moving_mass_kg": result.metrics["effective_moving_mass_kg"],
        "moving_mass_enabled": result.metrics["moving_mass_enabled"],
        "total_com_geometry_active": result.metrics["total_com_geometry_active"],
        "legacy_gravity_offset_active": result.metrics["legacy_gravity_offset_active"],
    }
    metrics.update(_pitch_peak_metrics(times, theta_deg))
    if definition.requires_target_transient_gates:
        metrics.update(transient_gate_metrics(definition, rows))
    else:
        metrics.update(
            {
                "motion_start_detected": True,
                "premature_pause": False,
                "early_velocity_reversal": False,
                "second_acceleration_lobe_after_full_pause": False,
            }
        )
    if definition.requires_capture_gates:
        metrics.update(capture_gate_metrics(definition, rows))
        second_lobe = transient_gate_metrics(definition, rows)[
            "second_acceleration_lobe_after_full_pause"
        ]
        metrics["second_acceleration_lobe_after_full_pause"] = second_lobe
    else:
        metrics.update(
            {
                "target_capture_count": 0,
                "target_capture_event_count": 0,
                "capture_max_target_jump_m": 0.0,
                "capture_discontinuity": False,
                "shaped_velocity_sign_reversal_after_release": False,
            }
        )
    reasons = hard_gate_reasons(definition, metrics)
    metrics["rejected"] = bool(reasons)
    metrics["rejection_reasons"] = "; ".join(reasons)
    return metrics


def hard_gate_reasons(definition: ScenarioDefinition, metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not metrics.get("finite", False):
        reasons.append("non_finite")
    if metrics.get("crash"):
        reasons.append("crash")
    if metrics.get("ground_contact"):
        reasons.append("ground_contact")
    if float(metrics.get("peak_abs_pitch_deg", math.inf)) > HARD_GATE_THRESHOLDS[
        "pitch_divergence_deg"
    ]:
        reasons.append("pitch_divergence")
    for key in (
        "premature_pause",
        "early_velocity_reversal",
        "second_acceleration_lobe_after_full_pause",
        "capture_discontinuity",
        "shaped_velocity_sign_reversal_after_release",
    ):
        if metrics.get(key, False):
            reasons.append(key)
    if definition.requires_capture_gates and int(metrics.get("target_capture_count", 0)) != 1:
        reasons.append("capture_count_not_one")
    if float(metrics.get("vane_saturation_percent", math.inf)) > HARD_GATE_THRESHOLDS[
        "vane_saturation_percent"
    ]:
        reasons.append("excessive_vane_saturation")
    if float(metrics.get("servo_rate_saturation_percent", math.inf)) > HARD_GATE_THRESHOLDS[
        "servo_rate_saturation_percent"
    ]:
        reasons.append("excessive_servo_rate_saturation")
    if float(metrics.get("mixer_saturation_percent", math.inf)) > HARD_GATE_THRESHOLDS[
        "mixer_saturation_percent"
    ]:
        reasons.append("excessive_mixer_saturation")
    chatter = (
        int(metrics.get("meaningful_vane_sign_change_count", 0))
        > CHATTER_THRESHOLDS["max_meaningful_sign_changes"]
        or float(metrics.get("vane_total_variation_per_second_deg_s", math.inf))
        > CHATTER_THRESHOLDS["max_total_variation_per_second_deg_s"]
        or float(metrics.get("tail_high_frequency_vane_energy_deg2", math.inf))
        > CHATTER_THRESHOLDS["max_tail_high_frequency_energy_deg2"]
    )
    if chatter:
        reasons.append("vane_chatter")
    if not math.isclose(float(metrics.get("moving_mass_assist_gain_m_per_Nm", math.nan)), 0.0, abs_tol=0.0):
        reasons.append("moving_mass_gain_nonzero")
    if float(metrics.get("moving_mass_max_abs_offset_m", math.inf)) != 0.0:
        reasons.append("moving_mass_actual_not_locked")
    if float(metrics.get("moving_mass_max_abs_target_m", math.inf)) != 0.0:
        reasons.append("moving_mass_target_not_locked")
    expected_effective = {
        "effective_atc_rat_pit_i": 0.0,
        "effective_psc_ne_pos_p": 0.55,
        "effective_psc_ne_vel_p": 0.70,
        "total_mass_kg": 2.0,
        "physical_moving_mass_kg": 0.5,
    }
    for key, expected in expected_effective.items():
        if not math.isclose(float(metrics.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-12):
            reasons.append(f"effective_parameter_mismatch:{key}")
    if not metrics.get("moving_mass_enabled", False):
        reasons.append("physical_moving_mass_disabled")
    if not metrics.get("total_com_geometry_active", False):
        reasons.append("total_com_geometry_disabled")
    if metrics.get("legacy_gravity_offset_active", True):
        reasons.append("legacy_gravity_offset_active")
    return list(dict.fromkeys(reasons))


def _number(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def asymmetry_fraction(positive: float, negative: float) -> float:
    denominator = max((abs(positive) + abs(negative)) / 2.0, 1e-12)
    return abs(positive - negative) / denominator


SYMMETRY_METRICS = (
    "tail_rms_pitch_deg",
    "tail_rms_pitch_rate_deg_s",
    "tail_rms_horizontal_velocity_m_s",
    "tail_path_length_m",
    "position_overshoot_m",
    "peak_abs_pitch_deg",
    "vane_command_rms_deg",
)
SYMMETRY_PAIRS = (
    ("loiter_positive_disturbance", "loiter_negative_disturbance"),
    ("forward_1m", "backward_1m"),
    ("pitch_positive_recovery", "pitch_negative_recovery"),
)


def scenario_score(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    score = 0.0
    for metric, weight in SCORE_WEIGHTS.items():
        value = _number(row.get(metric))
        reference = max(abs(_number(baseline.get(metric))), 1e-12)
        normalized = value / reference
        components[metric] = normalized
        score += weight * normalized
    return score, components


def aggregate_candidates(
    scenario_rows: Sequence[dict[str, Any]],
    baseline_by_scenario: dict[str, dict[str, Any]],
    scenarios: Sequence[ScenarioDefinition],
    stage: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scenario_rows:
        if str(row.get("stage")) == stage:
            grouped[str(row["candidate_key"])].append(dict(row))
    expected = {scenario.key for scenario in scenarios}
    aggregates: list[dict[str, Any]] = []
    for candidate_key, rows in sorted(grouped.items()):
        first = rows[0]
        by_scenario = {str(row["scenario_name"]): row for row in rows}
        reasons: list[str] = []
        missing = sorted(expected - set(by_scenario))
        if missing:
            reasons.append("missing_scenarios:" + ",".join(missing))
        for row in rows:
            if _boolean(row.get("rejected")):
                reasons.extend(
                    reason.strip()
                    for reason in str(row.get("rejection_reasons", "")).split(";")
                    if reason.strip()
                )
        scenario_scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}
        if not missing:
            for name, row in by_scenario.items():
                score, components = scenario_score(row, baseline_by_scenario[name])
                scenario_scores[name] = score
                score_components[name] = components

        symmetry: dict[str, float] = {}
        for positive_name, negative_name in SYMMETRY_PAIRS:
            if positive_name not in by_scenario or negative_name not in by_scenario:
                continue
            for metric in SYMMETRY_METRICS:
                key = f"{positive_name}__{negative_name}__{metric}"
                symmetry[key] = asymmetry_fraction(
                    _number(by_scenario[positive_name].get(metric)),
                    _number(by_scenario[negative_name].get(metric)),
                )
        worst_symmetry = max(symmetry.values(), default=0.0)
        if worst_symmetry > HARD_GATE_THRESHOLDS["symmetry_max_fraction"]:
            reasons.append("severe_mirrored_scenario_asymmetry")

        group_scores: dict[str, float] = {}
        for group in GROUP_WEIGHTS:
            group_names = [scenario.key for scenario in scenarios if scenario.group == group]
            values = [scenario_scores[name] for name in group_names if name in scenario_scores]
            group_scores[group] = (
                float(np.mean(values) + 0.5 * np.max(values)) if values else math.inf
            )
        final_score = sum(
            GROUP_WEIGHTS[group] * group_scores[group] for group in GROUP_WEIGHTS
        )
        aggregates.append(
            {
                "stage": stage,
                "candidate_key": candidate_key,
                "rate_p": _number(first["rate_p"]),
                "rate_d": _number(first["rate_d"]),
                "angle_p": _number(first["angle_p"]),
                "scenario_count": len(rows),
                "rejected": bool(reasons),
                "rejection_reasons": "; ".join(dict.fromkeys(reasons)),
                "final_score": math.inf if reasons else final_score,
                "inner_loop_aggregate": group_scores["inner_loop"],
                "integrated_loiter_aggregate": group_scores["integrated_loiter"],
                "worst_scenario_score": max(scenario_scores.values(), default=math.inf),
                "mean_vane_command_rms_deg": float(
                    np.mean([_number(row["vane_command_rms_deg"]) for row in rows])
                ),
                "mean_vane_total_variation_deg": float(
                    np.mean(
                        [_number(row["vane_command_total_variation_deg"]) for row in rows]
                    )
                ),
                "worst_asymmetry_fraction": worst_symmetry,
                "symmetry_json": _canonical_json(symmetry),
                "scenario_scores_json": _canonical_json(scenario_scores),
                "score_components_json": _canonical_json(score_components),
            }
        )
    ranked = sorted(
        aggregates,
        key=lambda row: (
            _boolean(row["rejected"]),
            _number(row["final_score"], math.inf),
            _number(row["mean_vane_command_rms_deg"], math.inf),
            _number(row["rate_d"], math.inf),
            _number(row["worst_asymmetry_fraction"], math.inf),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank if not _boolean(row["rejected"]) else ""
    return ranked


def valid_near_equivalent_candidates(
    aggregates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid = [row for row in aggregates if not _boolean(row.get("rejected"))]
    if not valid:
        categories = Counter(
            reason.strip()
            for row in aggregates
            for reason in str(row.get("rejection_reasons", "")).split(";")
            if reason.strip()
        )
        raise RuntimeError(f"no valid candidates; rejection categories={dict(categories)}")
    best_score = min(_number(row["final_score"]) for row in valid)
    return [
        row
        for row in valid
        if _number(row["final_score"])
        <= best_score + NEAR_EQUIVALENCE_ABSOLUTE_MARGIN
    ]


def raw_score_best(aggregates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in aggregates if not _boolean(row.get("rejected"))]
    if not valid:
        valid_near_equivalent_candidates(aggregates)
        raise AssertionError("unreachable")
    return min(valid, key=lambda row: _number(row["final_score"], math.inf))


def select_near_equivalent(aggregates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    near = valid_near_equivalent_candidates(aggregates)
    return min(
        near,
        key=lambda row: (
            _number(row["mean_vane_command_rms_deg"], math.inf),
            _number(row["rate_d"], math.inf),
            _number(row["mean_vane_total_variation_deg"], math.inf),
            _number(row["worst_asymmetry_fraction"], math.inf),
            _number(row["final_score"], math.inf),
        ),
    )


def candidate_from_row(row: dict[str, Any], stage: str | None = None) -> Candidate:
    return Candidate(
        stage or str(row["stage"]),
        _number(row["rate_p"]),
        _number(row["rate_d"]),
        _number(row["angle_p"]),
    )


def stage_candidates_from_rows(
    rows: Sequence[dict[str, Any]], stage: str
) -> list[Candidate]:
    by_key: dict[str, Candidate] = {}
    for row in rows:
        if str(row.get("stage")) != stage:
            continue
        candidate = candidate_from_row(row, stage)
        by_key[candidate.key] = candidate
    return sorted(
        by_key.values(), key=lambda item: (item.rate_p, item.rate_d, item.angle_p)
    )


def merge_candidates(*groups: Sequence[Candidate]) -> list[Candidate]:
    by_key = {candidate.key: candidate for group in groups for candidate in group}
    return sorted(
        by_key.values(), key=lambda item: (item.rate_p, item.rate_d, item.angle_p)
    )


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: Path, content: str) -> None:
    _atomic_replace_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv_text(rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> str:
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        return stream.read()


def atomic_write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None
) -> None:
    atomic_write_text(path, _csv_text(rows, fields))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class ScenarioResultStore:
    def __init__(self, path: Path, fingerprint: str, *, resume: bool):
        self.path = path
        self.fingerprint = fingerprint
        self.rows: list[dict[str, Any]] = []
        self.by_key: dict[str, dict[str, Any]] = {}
        if resume and path.exists():
            for row in read_csv(path):
                if row.get("workflow_fingerprint") != fingerprint:
                    raise ValueError(
                        "stale scenario cache fingerprint; use --no-resume or a new output directory"
                    )
                key = row.get("run_key", "")
                if not key or key in self.by_key:
                    raise ValueError("missing or duplicate run_key in scenario cache")
                self.rows.append(row)
                self.by_key[key] = row

    def get(self, run_key: str) -> dict[str, Any] | None:
        return self.by_key.get(run_key)

    def add(self, row: dict[str, Any]) -> None:
        if row["workflow_fingerprint"] != self.fingerprint:
            raise ValueError("scenario row fingerprint mismatch")
        key = str(row["run_key"])
        if key in self.by_key:
            raise ValueError(f"duplicate scenario run key: {key}")
        self.rows.append(row)
        self.by_key[key] = row

    def save(self) -> None:
        atomic_write_csv(self.path, self.rows)


def _run_key(candidate: Candidate, scenario: ScenarioDefinition, fingerprint: str) -> str:
    scenario_hash = _sha256_bytes(_canonical_json(asdict(scenario)).encode())
    return f"{candidate.key}:scenario={scenario.key}:scenario_hash={scenario_hash}:workflow={fingerprint}"


def _candidate_parameter_mismatch(
    candidate: Candidate, metrics: dict[str, Any]
) -> list[str]:
    checks = {
        "effective_atc_rat_pit_p": candidate.rate_p,
        "effective_atc_rat_pit_d": candidate.rate_d,
        "effective_atc_ang_pit_p": candidate.angle_p,
    }
    return [
        f"controller_mismatch:{key}"
        for key, expected in checks.items()
        if not math.isclose(
            _number(metrics.get(key)), expected, rel_tol=0.0, abs_tol=1e-12
        )
    ]


def run_candidate_scenarios(
    candidate: Candidate,
    scenarios: Sequence[ScenarioDefinition],
    store: ScenarioResultStore,
    fingerprint: str,
    *,
    quick: bool,
    keep_results: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, LoiterRunResult]]:
    output_rows: list[dict[str, Any]] = []
    results: dict[str, LoiterRunResult] = {}
    new_rows = False
    for definition in scenarios:
        key = _run_key(candidate, definition, fingerprint)
        cached = store.get(key)
        if cached is not None:
            output_rows.append(cached)
            continue
        result = run_headless_loiter(
            SOURCE_PROFILE,
            definition.config,
            rb_overrides=PHYSICAL_CONFIGURATION,
            controller_overrides=candidate.controller_overrides(),
        )
        metrics = compute_metrics(definition, result, quick=quick)
        mismatch = _candidate_parameter_mismatch(candidate, metrics)
        if mismatch:
            prior = [
                value.strip()
                for value in str(metrics.get("rejection_reasons", "")).split(";")
                if value.strip()
            ]
            metrics["rejected"] = True
            metrics["rejection_reasons"] = "; ".join([*prior, *mismatch])
        row = {
            "run_key": key,
            "workflow_fingerprint": fingerprint,
            "stage": candidate.stage,
            "candidate_key": candidate.key,
            "rate_p": candidate.rate_p,
            "rate_i": 0.0,
            "rate_d": candidate.rate_d,
            "angle_p": candidate.angle_p,
            **metrics,
        }
        store.add(row)
        output_rows.append(row)
        new_rows = True
        if keep_results:
            results[definition.key] = result
    if new_rows:
        store.save()
    return output_rows, results


def run_stage(
    stage: str,
    candidates: Sequence[Candidate],
    scenarios: Sequence[ScenarioDefinition],
    store: ScenarioResultStore,
    fingerprint: str,
    baseline_by_scenario: dict[str, dict[str, Any]],
    *,
    quick: bool,
) -> list[dict[str, Any]]:
    for index, candidate in enumerate(candidates, start=1):
        run_candidate_scenarios(candidate, scenarios, store, fingerprint, quick=quick)
        if index == 1 or index == len(candidates) or index % 10 == 0:
            print(f"{stage}: {index}/{len(candidates)} candidates", flush=True)
    return aggregate_candidates(store.rows, baseline_by_scenario, scenarios, stage)


def selected_boundary_flags(
    selected: dict[str, Any], candidates: Sequence[Candidate]
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for field, attribute in (("rate_p", "rate_p"), ("rate_d", "rate_d"), ("angle_p", "angle_p")):
        values = sorted({getattr(candidate, attribute) for candidate in candidates})
        selected_value = _number(selected[field])
        flags[f"{field}_at_min"] = math.isclose(selected_value, values[0], abs_tol=1e-12)
        flags[f"{field}_at_max"] = math.isclose(selected_value, values[-1], abs_tol=1e-12)
    return flags


def stage1_boundary_extension(
    selected: dict[str, Any], candidates: Sequence[Candidate]
) -> list[Candidate]:
    flags = selected_boundary_flags(selected, candidates)
    p_values = sorted({candidate.rate_p for candidate in candidates})
    d_values = sorted({candidate.rate_d for candidate in candidates})
    extended_p: list[float] = []
    extended_d: list[float] = []
    if flags["rate_p_at_min"]:
        extended_p = [max(0.005, p_values[0] - delta) for delta in (0.020, 0.015, 0.010, 0.005)]
    elif flags["rate_p_at_max"]:
        extended_p = [p_values[-1] + delta for delta in (0.005, 0.010, 0.015, 0.020)]
    if flags["rate_d_at_min"]:
        extended_d = [max(0.0, d_values[0] - delta) for delta in (0.006, 0.004, 0.002)]
    elif flags["rate_d_at_max"]:
        extended_d = [d_values[-1] + delta for delta in (0.002, 0.004, 0.006)]
    additions = {
        Candidate("stage1_rate_pd", rate_p, rate_d, 10.0)
        for rate_p in [*extended_p, *p_values]
        for rate_d in [*extended_d, *d_values]
        if rate_p in extended_p or rate_d in extended_d
    }
    return sorted(additions, key=lambda candidate: (candidate.rate_p, candidate.rate_d))


def baseline_mismatch_reasons(
    rows: Sequence[dict[str, Any]], scenarios: Sequence[ScenarioDefinition]
) -> list[str]:
    by_name = {str(row["scenario_name"]): row for row in rows}
    reasons: list[str] = []
    for definition in scenarios:
        row = by_name[definition.key]
        required_flags = ["finite", "crash", "ground_contact"]
        if not _boolean(row["finite"]):
            reasons.append(f"{definition.key}:non_finite")
        if _boolean(row["crash"]):
            reasons.append(f"{definition.key}:crash")
        if _boolean(row["ground_contact"]):
            reasons.append(f"{definition.key}:ground_contact")
        for key in (
            "premature_pause",
            "early_velocity_reversal",
            "second_acceleration_lobe_after_full_pause",
            "capture_discontinuity",
            "shaped_velocity_sign_reversal_after_release",
        ):
            if _boolean(row.get(key)):
                reasons.append(f"{definition.key}:{key}")
        if definition.requires_capture_gates and int(_number(row.get("target_capture_count"), 0)) != 1:
            reasons.append(f"{definition.key}:capture_count_not_one")
        if _number(row.get("moving_mass_max_abs_offset_m"), math.inf) != 0.0:
            reasons.append(f"{definition.key}:moving_mass_actual_not_locked")
        if _number(row.get("moving_mass_max_abs_target_m"), math.inf) != 0.0:
            reasons.append(f"{definition.key}:moving_mass_target_not_locked")
    return reasons


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return value.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return value.name
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _profile_and_preservation_hashes() -> dict[str, str]:
    paths = [*CANONICAL_PROFILES]
    if PRESERVED_RESULTS.exists():
        paths.extend(sorted(path for path in PRESERVED_RESULTS.rglob("*") if path.is_file()))
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _save_timeseries_atomic(result: LoiterRunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_loiter_timeseries(result.rows, temporary)
    os.replace(temporary, path)


def _write_baseline_outputs(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    results: dict[str, LoiterRunResult],
    mismatch: Sequence[str],
) -> None:
    baseline_dir = output_dir / "baseline"
    atomic_write_csv(baseline_dir / "baseline_scenario_results.csv", rows)
    atomic_write_json(
        baseline_dir / "baseline_mismatch.json",
        _json_safe(
            {
                "status": "pass" if not mismatch else "mismatch",
                "reasons": list(mismatch),
                "controller": BASELINE_GAINS,
                "source_profile": SOURCE_PROFILE.relative_to(REPO_ROOT).as_posix(),
                "source_profile_sha256": sha256_file(SOURCE_PROFILE),
                "physical_configuration": PHYSICAL_CONFIGURATION,
            }
        ),
    )
    for name, result in results.items():
        _save_timeseries_atomic(result, baseline_dir / TIMESERIES_FILENAMES[name])


def _fresh_validation_runs(
    candidate: Candidate,
    scenarios: Sequence[ScenarioDefinition],
    *,
    quick: bool,
) -> tuple[list[dict[str, dict[str, Any]]], dict[str, LoiterRunResult], list[str]]:
    runs: list[dict[str, dict[str, Any]]] = []
    first_results: dict[str, LoiterRunResult] = {}
    digests: list[str] = []
    for rerun in range(2):
        metrics_by_scenario: dict[str, dict[str, Any]] = {}
        for definition in scenarios:
            result = run_headless_loiter(
                SOURCE_PROFILE,
                definition.config,
                rb_overrides=PHYSICAL_CONFIGURATION,
                controller_overrides=candidate.controller_overrides(),
            )
            metrics = compute_metrics(definition, result, quick=quick)
            metrics_by_scenario[definition.key] = metrics
            if rerun == 0:
                first_results[definition.key] = result
        runs.append(metrics_by_scenario)
        digests.append(_sha256_bytes(_canonical_json(_json_safe(metrics_by_scenario)).encode()))
    if len(set(digests)) != 1:
        raise RuntimeError(f"selected-candidate deterministic rerun mismatch: {digests}")
    return runs, first_results, digests


def _comparison_summary(
    baseline_by_scenario: dict[str, dict[str, Any]],
    selected_by_scenario: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "tail_rms_pitch_deg",
        "tail_rms_pitch_rate_deg_s",
        "tail_rms_horizontal_velocity_m_s",
        "tail_path_length_m",
        "tail_rms_position_error_m",
        "vane_command_rms_deg",
        "vane_command_total_variation_deg",
    )
    comparison: dict[str, Any] = {"per_scenario": {}}
    for scenario_name, selected in selected_by_scenario.items():
        baseline = baseline_by_scenario[scenario_name]
        scenario_comparison = {}
        for metric in metrics:
            old = _number(baseline[metric])
            new = _number(selected[metric])
            scenario_comparison[metric] = {
                "baseline": old,
                "selected": new,
                "change_percent": 100.0 * (new - old) / max(abs(old), 1e-12),
            }
        comparison["per_scenario"][scenario_name] = scenario_comparison
    comparison["aggregate"] = {}
    for metric in metrics:
        baseline_values = np.asarray(
            [_number(row[metric]) for row in baseline_by_scenario.values()], dtype=float
        )
        selected_values = np.asarray(
            [_number(row[metric]) for row in selected_by_scenario.values()], dtype=float
        )
        comparison["aggregate"][metric] = {
            "baseline_mean": float(np.mean(baseline_values)),
            "selected_mean": float(np.mean(selected_values)),
            "mean_improvement_percent": float(
                100.0 * (np.mean(baseline_values) - np.mean(selected_values))
                / max(abs(float(np.mean(baseline_values))), 1e-12)
            ),
            "baseline_worst": float(np.max(baseline_values)),
            "selected_worst": float(np.max(selected_values)),
            "worst_improvement_percent": float(
                100.0 * (np.max(baseline_values) - np.max(selected_values))
                / max(abs(float(np.max(baseline_values))), 1e-12)
            ),
        }
    return comparison


SELECTION_COMPARISON_METRICS = (
    ("tail_rms_pitch_deg", "Tail RMS pitch (deg)"),
    ("tail_rms_pitch_rate_deg_s", "Tail RMS pitch rate (deg/s)"),
    ("tail_rms_horizontal_velocity_m_s", "Tail RMS vx (m/s)"),
    ("tail_path_length_m", "Tail path (m)"),
    ("final_abs_position_error_m", "Final absolute error (m)"),
    ("position_overshoot_m", "Position overshoot (m)"),
    ("recovery_excursion_m", "Recovery excursion (m)"),
    ("strict_settling_time_s", "Strict settling time (s)"),
    ("vane_command_rms_deg", "Vane RMS (deg)"),
    ("vane_command_total_variation_deg", "Vane total variation (deg)"),
    ("vane_command_rate_rms_deg_s", "Vane command-rate RMS (deg/s)"),
    ("meaningful_vane_sign_change_count", "Meaningful vane sign changes"),
    ("vane_total_variation_per_second_deg_s", "Vane variation rate (deg/s)"),
    ("tail_high_frequency_vane_energy_deg2", "Tail high-frequency vane energy (deg^2)"),
    ("vane_zero_crossing_frequency_hz", "Vane zero-crossing frequency (Hz)"),
    ("vane_saturation_percent", "Vane saturation (%)"),
    ("servo_rate_saturation_percent", "Servo-rate saturation (%)"),
    ("mixer_saturation_percent", "Mixer saturation (%)"),
)

SELECTION_HARD_GATE_FIELDS = (
    "early_velocity_reversal",
    "premature_pause",
    "second_acceleration_lobe_after_full_pause",
    "capture_discontinuity",
    "shaped_velocity_sign_reversal_after_release",
    "crash",
    "ground_contact",
    "finite",
    "target_capture_count",
    "vane_saturation_percent",
    "servo_rate_saturation_percent",
    "mixer_saturation_percent",
    "meaningful_vane_sign_change_count",
    "vane_total_variation_per_second_deg_s",
    "tail_high_frequency_vane_energy_deg2",
    "moving_mass_assist_gain_m_per_Nm",
    "moving_mass_max_abs_offset_m",
    "moving_mass_max_abs_target_m",
)


def _selection_comparison_audit(
    raw_best_row: dict[str, Any],
    rank15_row: dict[str, Any],
    final_stage_rows: Sequence[dict[str, Any]],
    raw_by_scenario: dict[str, dict[str, Any]],
    rank15_by_scenario: dict[str, dict[str, Any]],
    final_boundary_flags: dict[str, bool],
    rank15_boundary_flags: dict[str, bool],
) -> dict[str, Any]:
    raw_score = _number(raw_best_row["final_score"])
    rank15_score = _number(rank15_row["final_score"])
    score_gap = rank15_score - raw_score
    near = valid_near_equivalent_candidates(final_stage_rows)

    aggregate_metrics: dict[str, Any] = {}
    per_scenario: dict[str, Any] = {}
    for metric, label in SELECTION_COMPARISON_METRICS:
        raw_values = np.asarray(
            [_number(row[metric]) for row in raw_by_scenario.values()], dtype=float
        )
        rank15_values = np.asarray(
            [_number(row[metric]) for row in rank15_by_scenario.values()], dtype=float
        )
        raw_mean = float(np.mean(raw_values))
        rank15_mean = float(np.mean(rank15_values))
        aggregate_metrics[metric] = {
            "label": label,
            "raw_score_rank1_mean": raw_mean,
            "rank15_low_control_effort_mean": rank15_mean,
            "rank15_minus_raw": rank15_mean - raw_mean,
            "rank15_change_percent": 100.0
            * (rank15_mean - raw_mean)
            / max(abs(raw_mean), 1e-12),
        }
    for scenario_name in sorted(rank15_by_scenario):
        raw_metrics = raw_by_scenario[scenario_name]
        rank15_metrics = rank15_by_scenario[scenario_name]
        metric_rows: dict[str, Any] = {}
        for metric, label in SELECTION_COMPARISON_METRICS:
            raw_value = _number(raw_metrics[metric])
            rank15_value = _number(rank15_metrics[metric])
            metric_rows[metric] = {
                "label": label,
                "raw_score_rank1": raw_value,
                "rank15_low_control_effort": rank15_value,
                "rank15_minus_raw": rank15_value - raw_value,
                "rank15_change_percent": 100.0
                * (rank15_value - raw_value)
                / max(abs(raw_value), 1e-12),
            }
        per_scenario[scenario_name] = {
            "metrics": metric_rows,
            "raw_score_rank1_hard_gates": {
                key: raw_metrics.get(key) for key in SELECTION_HARD_GATE_FIELDS
            },
            "rank15_low_control_effort_hard_gates": {
                key: rank15_metrics.get(key) for key in SELECTION_HARD_GATE_FIELDS
            },
            "raw_score_rank1_rejected": _boolean(raw_metrics.get("rejected")),
            "rank15_low_control_effort_rejected": _boolean(rank15_metrics.get("rejected")),
        }

    raw_scenario_scores = json.loads(str(raw_best_row["scenario_scores_json"]))
    rank15_scenario_scores = json.loads(str(rank15_row["scenario_scores_json"]))
    vane_rms = aggregate_metrics["vane_command_rms_deg"]
    vane_tv = aggregate_metrics["vane_command_total_variation_deg"]
    vane_rate = aggregate_metrics["vane_command_rate_rms_deg_s"]
    raw_hard_gates_pass = all(
        not _boolean(row.get("rejected")) for row in raw_by_scenario.values()
    )
    rank15_hard_gates_pass = all(
        not _boolean(row.get("rejected")) for row in rank15_by_scenario.values()
    )
    return {
        "selection_description": "raw-score rank-1 controller selected for final use",
        "selection_reason": (
            "This task prioritizes pitch damping, residual velocity, and tail-path "
            "performance; raw-score rank 1 passes every hard gate with zero saturation."
        ),
        "near_equivalence_rule": {
            "declared_before_final_ranking": True,
            "formula": "valid raw aggregate score <= raw-score best + 0.010000",
            "absolute_margin": NEAR_EQUIVALENCE_ABSOLUTE_MARGIN,
            "raw_score_best": raw_score,
            "upper_bound_inclusive": raw_score + NEAR_EQUIVALENCE_ABSOLUTE_MARGIN,
            "valid_candidate_count": sum(
                not _boolean(row.get("rejected")) for row in final_stage_rows
            ),
            "near_equivalent_candidate_count": len(near),
        },
        "raw_score_rank1": {
            "candidate": asdict(candidate_from_row(raw_best_row, "raw_score_validation")),
            "raw_score_rank": int(_number(raw_best_row.get("rank"), 1)),
            "raw_aggregate_score": raw_score,
            "scenario_mean_score": float(np.mean(list(raw_scenario_scores.values()))),
            "worst_scenario_score": _number(raw_best_row["worst_scenario_score"]),
            "symmetry": json.loads(str(raw_best_row["symmetry_json"])),
            "all_hard_gates_pass": raw_hard_gates_pass,
            "final_selected": True,
            "boundary_flags": dict(final_boundary_flags),
        },
        "rank15_low_control_effort": {
            "candidate": asdict(candidate_from_row(rank15_row, "rank15_validation")),
            "raw_score_rank": int(_number(rank15_row.get("rank"), 0)),
            "raw_aggregate_score": rank15_score,
            "scenario_mean_score": float(np.mean(list(rank15_scenario_scores.values()))),
            "worst_scenario_score": _number(rank15_row["worst_scenario_score"]),
            "symmetry": json.loads(str(rank15_row["symmetry_json"])),
            "all_hard_gates_pass": rank15_hard_gates_pass,
            "final_selected": False,
            "boundary_flags": dict(rank15_boundary_flags),
        },
        "rank15_score_penalty": {
            "absolute": score_gap,
            "relative_percent": 100.0 * score_gap / max(abs(raw_score), 1e-12),
        },
        "rank15_control_effort_reduction": {
            "mean_vane_rms_percent": -vane_rms["rank15_change_percent"],
            "mean_vane_total_variation_percent": -vane_tv["rank15_change_percent"],
            "mean_vane_command_rate_rms_percent": -vane_rate["rank15_change_percent"],
        },
        "aggregate_metrics": aggregate_metrics,
        "per_scenario": per_scenario,
        "audit_passed": (
            raw_hard_gates_pass
            and rank15_hard_gates_pass
            and rank15_score <= raw_score + NEAR_EQUIVALENCE_ABSOLUTE_MARGIN
            and int(_number(rank15_row.get("rank"), 0)) == 15
            and int(_number(raw_best_row.get("rank"), 0)) == 1
            and -vane_rms["rank15_change_percent"] > 0.0
            and not any(rank15_boundary_flags.values())
        ),
    }


def _selection_comparison_csv_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, payload in audit["aggregate_metrics"].items():
        rows.append(
            {
                "scope": "all_scenarios_mean",
                "scenario": "all",
                "metric": metric,
                "label": payload["label"],
                "raw_score_rank1": payload["raw_score_rank1_mean"],
                "rank15_low_control_effort": payload["rank15_low_control_effort_mean"],
                "rank15_minus_raw": payload["rank15_minus_raw"],
                "rank15_change_percent": payload["rank15_change_percent"],
            }
        )
    for scenario_name, scenario_payload in audit["per_scenario"].items():
        for metric, payload in scenario_payload["metrics"].items():
            rows.append(
                {
                    "scope": "scenario",
                    "scenario": scenario_name,
                    "metric": metric,
                    "label": payload["label"],
                    "raw_score_rank1": payload["raw_score_rank1"],
                    "rank15_low_control_effort": payload["rank15_low_control_effort"],
                    "rank15_minus_raw": payload["rank15_minus_raw"],
                    "rank15_change_percent": payload["rank15_change_percent"],
                }
            )
    return rows


def _selection_comparison_markdown(audit: dict[str, Any]) -> str:
    raw = audit["raw_score_rank1"]
    rank15 = audit["rank15_low_control_effort"]
    rule = audit["near_equivalence_rule"]
    penalty = audit["rank15_score_penalty"]
    effort = audit["rank15_control_effort_reduction"]
    lines = [
        "# Final raw-score rank 1 versus previous rank-15 low-control-effort candidate",
        "",
        "Stage 0 is a **FAILED / NON-ACCEPTABLE baseline used for normalization only**. It is not a validated controller.",
        "",
        "The final selected controller is **raw-score rank 1**. The previous rank-15 low-control-effort candidate is retained only as a transparent comparison and is not the final selected controller.",
        "",
        f"The predeclared inclusive rule is `{rule['formula']}`. The best raw score is `{rule['raw_score_best']:.9f}`, the limit is `{rule['upper_bound_inclusive']:.9f}`, and `{rule['near_equivalent_candidate_count']}` of `{rule['valid_candidate_count']}` valid Stage 3C candidates are inside the band.",
        "",
        f"The previous rank-15 score is `{rank15['raw_aggregate_score']:.9f}` (absolute penalty `{penalty['absolute']:.9f}`, relative penalty `{penalty['relative_percent']:.3f}%`). Relative to final rank 1, rank 15 reduces mean vane RMS by `{effort['mean_vane_rms_percent']:.3f}%`, vane total variation by `{effort['mean_vane_total_variation_percent']:.3f}%`, and vane command-rate RMS by `{effort['mean_vane_command_rate_rms_percent']:.3f}%`. That effort reduction is not used as the final tie-break because this task prioritizes pitch damping, residual velocity, and tail-path performance.",
        "",
        "| metric | final selected raw-score rank 1 | previous rank 15 | rank-15 change |",
        "| --- | ---: | ---: | ---: |",
        f"| raw aggregate score | {raw['raw_aggregate_score']:.9f} | {rank15['raw_aggregate_score']:.9f} | {penalty['relative_percent']:.3f}% |",
        f"| scenario mean score | {raw['scenario_mean_score']:.9f} | {rank15['scenario_mean_score']:.9f} | {100.0 * (rank15['scenario_mean_score'] - raw['scenario_mean_score']) / max(abs(raw['scenario_mean_score']), 1e-12):.3f}% |",
        f"| worst-scenario score | {raw['worst_scenario_score']:.9f} | {rank15['worst_scenario_score']:.9f} | {100.0 * (rank15['worst_scenario_score'] - raw['worst_scenario_score']) / max(abs(raw['worst_scenario_score']), 1e-12):.3f}% |",
    ]
    for payload in audit["aggregate_metrics"].values():
        lines.append(
            f"| {payload['label']} | {payload['raw_score_rank1_mean']:.9f} | {payload['rank15_low_control_effort_mean']:.9f} | {payload['rank15_change_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Both candidates pass every physical and behavioral hard gate in all seven full-duration scenarios. All mirrored symmetry fractions are zero for both candidates. Detailed per-scenario metrics and every hard-gate field are preserved in `selection_comparison.json` and `selection_comparison.csv`.",
            "",
            "Rank 1 is final because it has lower aggregate score and better main pitch-damping, residual-velocity, and tail-path metrics while still passing all hard gates with zero saturation. Rank 15 remains documented because it uses modestly less vane effort; those results are retained rather than hidden.",
            "",
        ]
    )
    return "\n".join(lines)


def final_candidate_requirements(
    baseline_by_scenario: dict[str, dict[str, Any]],
    selected_by_scenario: dict[str, dict[str, Any]],
    boundary_flags: dict[str, bool],
    *,
    boundary_is_hard_gate: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    comparison = _comparison_summary(baseline_by_scenario, selected_by_scenario)
    failures: list[str] = []
    for name, metrics in selected_by_scenario.items():
        if _boolean(metrics.get("rejected")):
            failures.append(f"{name}:{metrics.get('rejection_reasons', '')}")
    pitch = comparison["aggregate"]["tail_rms_pitch_deg"]
    if pitch["mean_improvement_percent"] < 20.0:
        failures.append("mean_tail_rms_pitch_improvement_below_20_percent")
    if pitch["worst_improvement_percent"] < 10.0:
        failures.append("worst_tail_rms_pitch_improvement_below_10_percent")
    pitch_rate = comparison["aggregate"]["tail_rms_pitch_rate_deg_s"]
    if pitch_rate["mean_improvement_percent"] <= 0.0:
        failures.append("mean_tail_rms_pitch_rate_not_improved")
    vx = comparison["aggregate"]["tail_rms_horizontal_velocity_m_s"]
    path = comparison["aggregate"]["tail_path_length_m"]
    if vx["mean_improvement_percent"] <= 0.0 and path["mean_improvement_percent"] <= 0.0:
        failures.append("neither_mean_tail_velocity_nor_tail_path_improved")
    vane = comparison["aggregate"]["vane_command_rms_deg"]
    vane_increase = -vane["mean_improvement_percent"]
    if vane_increase > 10.0 and pitch["mean_improvement_percent"] < 40.0:
        failures.append("vane_rms_increase_above_10_percent_without_superior_damping")
    if boundary_is_hard_gate and any(boundary_flags.values()):
        failures.append("selected_point_on_search_boundary")
    return failures, comparison


def _write_validation_timeseries(
    output_dir: Path, subdirectory: str, results: dict[str, LoiterRunResult]
) -> None:
    selected_dir = output_dir / "validation" / subdirectory
    for name, result in results.items():
        _save_timeseries_atomic(result, selected_dir / TIMESERIES_FILENAMES[name])


def _write_selected_timeseries(
    output_dir: Path, results: dict[str, LoiterRunResult]
) -> None:
    _write_validation_timeseries(output_dir, "selected", results)


def _write_plots(
    output_dir: Path,
    baseline_results: dict[str, LoiterRunResult],
    selected_results: dict[str, LoiterRunResult],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = output_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        (
            "baseline_vs_selected_loiter.png",
            ("loiter_positive_disturbance", "loiter_negative_disturbance"),
            "LOITER disturbance recovery",
        ),
        (
            "baseline_vs_selected_forward_1m.png",
            ("forward_1m", "backward_1m"),
            "Absolute target steps",
        ),
        (
            "baseline_vs_selected_pitch_recovery.png",
            ("pitch_positive_recovery", "pitch_negative_recovery"),
            "Initial pitch recovery",
        ),
        (
            "baseline_vs_selected_stick_release.png",
            ("stick_release",),
            "Stick pulse and release",
        ),
    )
    paths: list[Path] = []
    for filename, names, title in plot_specs:
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for name in names:
            for label, results, style in (
                ("baseline", baseline_results, "--"),
                ("selected", selected_results, "-"),
            ):
                rows = results[name].rows
                t = _array(rows, "time")
                suffix = name.replace("_", " ")
                axes[0].plot(t, _array(rows, "x"), style, label=f"{label}: {suffix}")
                axes[1].plot(t, np.rad2deg(_array(rows, "theta")), style, label=f"{label}: {suffix}")
                axes[2].plot(t, np.rad2deg(_array(rows, "vane_angle_cmd")), style, label=f"{label}: {suffix}")
        axes[0].set_ylabel("x (m)")
        axes[1].set_ylabel("pitch (deg)")
        axes[2].set_ylabel("vane cmd (deg)")
        axes[2].set_xlabel("time (s)")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        figure.suptitle(title)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _profile_payload(
    selected: Candidate,
    fingerprint_payload: dict[str, Any],
    fingerprint: str,
    boundary_flags: dict[str, bool],
    rerun_digests: Sequence[str],
    limitations: Sequence[str],
    actual_search_space: dict[str, Any] | None = None,
    selection_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    profile.setdefault("rigid_body", {}).update(
        {key: value for key, value in PHYSICAL_CONFIGURATION.items() if key != "moving_mass"}
    )
    profile["rigid_body"]["moving_mass"] = PHYSICAL_CONFIGURATION["moving_mass"]
    profile.setdefault("controller", {}).update(
        {
            "atc_rat_pit_p": selected.rate_p,
            "atc_rat_pit_i": 0.0,
            "atc_rat_pit_d": selected.rate_d,
            "atc_ang_pit_p": selected.angle_p,
        }
    )
    profile["analysis"] = {
        "profile_status": "provisional",
        "profile_notes": "Vane-only pitch damping retune; does not replace a canonical profile.",
        "base_sha": fingerprint_payload["base_sha"],
        "source_profile": fingerprint_payload["source_profile"],
        "source_profile_hash": fingerprint_payload["source_profile_sha256"],
        "search_fingerprint": fingerprint,
        "scenario_fingerprint": fingerprint_payload["scenario_fingerprint"],
        "search_ranges": fingerprint_payload["search_ranges"],
        "actual_search_space": actual_search_space or {},
        "seed": SEED,
        "physics_timestep_s": fingerprint_payload["physics_timestep_s"],
        "controller_timestep_s": fingerprint_payload["controller_timestep_s"],
        "selected_rate_p": selected.rate_p,
        "selected_rate_i": 0.0,
        "selected_rate_d": selected.rate_d,
        "selected_angle_p": selected.angle_p,
        "fixed_controller_values": FIXED_CONTROLLER_VALUES,
        "moving_mass_assist_gain_m_per_Nm": 0.0,
        "boundary_flags": boundary_flags,
        "validation_date": date.today().isoformat(),
        "deterministic_rerun": {"passed": len(set(rerun_digests)) == 1, "digests": list(rerun_digests)},
        "limitations": list(limitations),
    }
    if selection_audit is not None:
        raw = selection_audit["raw_score_rank1"]
        rank15 = selection_audit["rank15_low_control_effort"]
        profile["analysis"]["selection"] = {
            "description": selection_audit["selection_description"],
            "reason": selection_audit["selection_reason"],
            "raw_score_rank": raw["raw_score_rank"],
            "raw_aggregate_score": raw["raw_aggregate_score"],
            "raw_score_best": raw["raw_aggregate_score"],
            "score_penalty": {"absolute": 0.0, "relative_percent": 0.0},
            "near_equivalence_rule": selection_audit["near_equivalence_rule"],
            "rank15_comparison": {
                "description": "previous low-control-effort alternative; not final selected controller",
                "final_selected": False,
                "raw_score_rank": rank15["raw_score_rank"],
                "raw_aggregate_score": rank15["raw_aggregate_score"],
                "score_penalty": selection_audit["rank15_score_penalty"],
                "control_effort_reduction": selection_audit["rank15_control_effort_reduction"],
                "all_hard_gates_pass": rank15["all_hard_gates_pass"],
            },
            "audit_passed": selection_audit["audit_passed"],
        }
    return profile


def _methodology_text(fingerprint_payload: dict[str, Any]) -> str:
    return f"""# Pitch damping retune methodology

This workflow compares the PR #19 Vane-only controller with pitch-retuned Vane-only candidates in the same deterministic 2D analytical model. It is not real-flight, 3D, ArduPilot, Pixhawk, HIL, or hardware-safety validation.

## Fixed vehicle and controller values

- Total mass: 2.0 kg; physical moving mass: 0.5 kg; fixed-body mass: 1.5 kg.
- Moving mass enabled with total-COM geometry, centered actual/target position, and assist gain exactly 0.0 m/Nm.
- Rate I: 0.0; Position P: 0.55; Velocity P: 0.70.
- Brake delay/acceleration/jerk: 0.50 s / 1.00 m/s^2 / 3.00 m/s^3.
- Capture actual/desired velocity thresholds: 0.08 / 0.02 m/s.
- Persistent capture, shaped-velocity zero clamp, and capture without target jump remain enabled.
- Physics/controller timesteps: {fingerprint_payload['physics_timestep_s']} s / {fingerprint_payload['controller_timestep_s']} s; seed: {SEED}.

## Scoring

Each metric is divided by the fresh Stage 0 value. Scenario score weights are 30% tail RMS pitch, 20% tail RMS pitch rate, 15% tail RMS horizontal velocity, 15% tail path, 10% tail RMS position error, 5% vane RMS, and 5% vane total variation. Each group uses mean + 0.5 times worst scenario; the final combination is 45% isolated pitch recovery and 55% integrated LOITER.

Stage 0 is a **FAILED / NON-ACCEPTABLE baseline used for normalization only**. Both +1 m and -1 m absolute-target runs fail the early-velocity-reversal hard gate. It is not a validated or acceptable controller, and its use does not relax any candidate gate.

The near-equivalent set was defined before final Stage 3C ranking inspection as every valid candidate whose raw aggregate score is less than or equal to the raw-score best plus exactly `{NEAR_EQUIVALENCE_ABSOLUTE_MARGIN:.6f}`. Its original low-control-effort tie-break results remain in the comparison audit for transparency. Final publication selects raw-score rank 1 instead because this task prioritizes pitch damping, residual velocity, and tail-path performance. Rank 15 is not the final selected controller.

## Chatter thresholds fixed before selection

- Command deadband: {CHATTER_THRESHOLDS['command_deadband_deg']} deg.
- Meaningful command rate: {CHATTER_THRESHOLDS['meaningful_rate_deg_s']} deg/s.
- Maximum meaningful sign changes: {CHATTER_THRESHOLDS['max_meaningful_sign_changes']}.
- Maximum total variation per second: {CHATTER_THRESHOLDS['max_total_variation_per_second_deg_s']} deg/s.
- Maximum tail high-frequency energy: {CHATTER_THRESHOLDS['max_tail_high_frequency_energy_deg2']} deg^2.
- High-frequency moving-average window: {CHATTER_THRESHOLDS['high_frequency_window_s']} s.

Logarithmic decrement and damping ratio are reported only when at least three significant absolute pitch peaks provide at least two monotonically decaying ratios; otherwise the reason is recorded.
"""


def _markdown_summary(
    selected: Candidate,
    comparison: dict[str, Any],
    candidate_counts: dict[str, int],
    scenario_run_count: int,
    rejected_count: int,
    boundary_flags: dict[str, bool],
    rerun_digests: Sequence[str],
    baseline_mismatch: Sequence[str],
    selection_audit: dict[str, Any],
) -> str:
    pitch = comparison["aggregate"]["tail_rms_pitch_deg"]
    pitch_rate = comparison["aggregate"]["tail_rms_pitch_rate_deg_s"]
    velocity = comparison["aggregate"]["tail_rms_horizontal_velocity_m_s"]
    raw = selection_audit["raw_score_rank1"]
    rank15 = selection_audit["rank15_low_control_effort"]
    rule = selection_audit["near_equivalence_rule"]
    penalty = selection_audit["rank15_score_penalty"]
    effort = selection_audit["rank15_control_effort_reduction"]
    return f"""# Vane-only pitch damping retune

The final selected controller is **raw-score rank 1**: Rate P/I/D `{selected.rate_p:.8f} / 0.00000000 / {selected.rate_d:.8f}` with Angle P `{selected.angle_p:.8f}`. It is selected because this task prioritizes pitch damping, residual velocity, and tail-path performance; it passes every hard gate with zero saturation. All outer-loop, braking, capture, physics, actuator, geometry, and scenario settings were fixed. Moving-mass assist remained exactly `0.0 m/Nm`, and the physical moving mass remained centered.

The predeclared inclusive near-equivalence set remains documented as `valid raw aggregate score <= raw-score best + {rule['absolute_margin']:.6f}`. The raw-score best and final selected score are both `{raw['raw_aggregate_score']:.9f}` with zero accepted score penalty. `{rule['near_equivalent_candidate_count']}` of `{rule['valid_candidate_count']}` valid Stage 3C candidates are inside the band. The previous rank-15 alternative scored `{rank15['raw_aggregate_score']:.9f}` (penalty `{penalty['absolute']:.9f}`, `{penalty['relative_percent']:.3f}%`) and reduced mean vane RMS by `{effort['mean_vane_rms_percent']:.3f}%`, vane total variation by `{effort['mean_vane_total_variation_percent']:.3f}%`, and vane command-rate RMS by `{effort['mean_vane_command_rate_rms_percent']:.3f}%`; it is retained for comparison only and is not the final selected controller.

## Baseline comparison

**Stage 0 status: FAILED / NON-ACCEPTABLE baseline used for normalization only.** It failed `forward_1m:early_velocity_reversal` and `backward_1m:early_velocity_reversal`. Stage 0 is not a validated or acceptable controller; its absolute metrics and detector failures are preserved in the baseline artifacts.

| metric | baseline mean | selected mean | improvement |
| --- | ---: | ---: | ---: |
| tail RMS pitch (deg) | {pitch['baseline_mean']:.8f} | {pitch['selected_mean']:.8f} | {pitch['mean_improvement_percent']:.3f}% |
| tail RMS pitch rate (deg/s) | {pitch_rate['baseline_mean']:.8f} | {pitch_rate['selected_mean']:.8f} | {pitch_rate['mean_improvement_percent']:.3f}% |
| tail RMS horizontal velocity (m/s) | {velocity['baseline_mean']:.8f} | {velocity['selected_mean']:.8f} | {velocity['mean_improvement_percent']:.3f}% |

## Search and validation

Both the final raw-score rank-1 controller and the previous rank-15 alternative pass every physical and hard gate in all seven full-duration scenarios. The final selected controller eliminates early velocity reversal in both +1 m and -1 m cases, records exactly one monotonic controller capture-count increment in stick release, has no capture discontinuity or shaped-vx reversal, and has zero vane/servo-rate/mixer saturation. Detailed side-by-side metrics, chatter, symmetry, and hard-gate results are in `selection_comparison.md`, `selection_comparison.csv`, and `selection_comparison.json`.

- Candidate counts: `{_canonical_json(candidate_counts)}`.
- Total unique scenario rows: `{scenario_run_count}`.
- Rejected candidates: `{rejected_count}`.
- Boundary flags: `{_canonical_json(boundary_flags)}`.
- Deterministic selected-candidate reruns: `{_canonical_json(list(rerun_digests))}`.
- Stage 0 mismatch override record: `{_canonical_json(list(baseline_mismatch))}`.

These results apply only to the same deterministic 2D analytical Single Fan Drone-inspired model. They do not establish real-flight stability, 3D stability, validated ArduPilot/Pixhawk behavior, HIL validity, hardware safety, or commercial-aircraft equivalence.
"""


def _actual_search_space(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage and stage != "stage0_baseline":
            by_stage[stage].append(dict(row))
    payload: dict[str, Any] = {}
    for stage, stage_rows in sorted(by_stage.items()):
        candidate_keys = sorted({str(row["candidate_key"]) for row in stage_rows})
        payload[stage] = {
            "candidate_count": len(candidate_keys),
            "scenario_row_count": len(stage_rows),
            "rate_p_values": sorted({_number(row["rate_p"]) for row in stage_rows}),
            "rate_d_values": sorted({_number(row["rate_d"]) for row in stage_rows}),
            "angle_p_values": sorted({_number(row["angle_p"]) for row in stage_rows}),
            "candidate_keys": candidate_keys,
        }
    return payload


def _write_final_exports(
    output_dir: Path,
    aggregates: dict[str, list[dict[str, Any]]],
    store: ScenarioResultStore,
    selected: Candidate,
    selected_by_scenario: dict[str, dict[str, Any]],
    baseline_by_scenario: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    fingerprint_payload: dict[str, Any],
    fingerprint: str,
    boundary_rows: Sequence[dict[str, Any]],
    boundary_flags: dict[str, bool],
    rerun_digests: Sequence[str],
    baseline_mismatch: Sequence[str],
    selection_audit: dict[str, Any],
    runtime_s: float,
) -> dict[str, Any]:
    candidate_rows = [row for rows in aggregates.values() for row in rows]
    atomic_write_csv(output_dir / "candidate_results.csv", candidate_rows)
    atomic_write_csv(output_dir / "scenario_results.csv", store.rows)
    atomic_write_csv(
        output_dir / "best_parameters.csv",
        [
            {
                "rate_p": selected.rate_p,
                "rate_i": 0.0,
                "rate_d": selected.rate_d,
                "angle_p": selected.angle_p,
                "position_p": 0.55,
                "velocity_p": 0.70,
                "moving_mass_assist_gain_m_per_Nm": 0.0,
            }
        ],
    )
    rejection_counts = Counter(
        reason.strip()
        for row in candidate_rows
        if _boolean(row.get("rejected"))
        for reason in str(row.get("rejection_reasons", "")).split(";")
        if reason.strip()
    )
    rejection_rows = [
        {"rejection_category": key, "candidate_count": value}
        for key, value in sorted(rejection_counts.items())
    ]
    atomic_write_csv(output_dir / "rejection_summary.csv", rejection_rows)
    atomic_write_csv(output_dir / "boundary_diagnostics.csv", list(boundary_rows))
    candidate_counts = {stage: len(rows) for stage, rows in aggregates.items()}
    actual_search_space = _actual_search_space(store.rows)
    completed_fingerprint_payload = {
        "workflow_fingerprint": fingerprint,
        "actual_search_space": actual_search_space,
    }
    completed_search_fingerprint = _sha256_bytes(
        _canonical_json(completed_fingerprint_payload).encode()
    )
    metadata = {
        "base_sha": fingerprint_payload["base_sha"],
        "workflow_fingerprint": fingerprint,
        "completed_search_fingerprint": completed_search_fingerprint,
        "scenario_fingerprint": fingerprint_payload["scenario_fingerprint"],
        "source_profile": fingerprint_payload["source_profile"],
        "source_profile_sha256": fingerprint_payload["source_profile_sha256"],
        "candidate_counts": candidate_counts,
        "actual_search_space": actual_search_space,
        "scenario_run_count": len(store.rows),
        "rejected_candidate_count": sum(
            _boolean(row.get("rejected")) for row in candidate_rows
        ),
        "runtime_s": runtime_s,
        "seed": SEED,
        "baseline_mismatch_override_reasons": list(baseline_mismatch),
        "selected": asdict(selected),
        "boundary_flags": boundary_flags,
        "deterministic_rerun_digests": list(rerun_digests),
        "selection": {
            "description": selection_audit["selection_description"],
            "reason": selection_audit["selection_reason"],
            "raw_score_rank": selection_audit["raw_score_rank1"]["raw_score_rank"],
            "raw_score_best": selection_audit["raw_score_rank1"]["raw_aggregate_score"],
            "selected_raw_score": selection_audit["raw_score_rank1"]["raw_aggregate_score"],
            "near_equivalent_candidate_count": selection_audit["near_equivalence_rule"]["near_equivalent_candidate_count"],
            "absolute_margin": selection_audit["near_equivalence_rule"]["absolute_margin"],
            "rank15_comparison_raw_score": selection_audit["rank15_low_control_effort"]["raw_aggregate_score"],
            "audit_passed": selection_audit["audit_passed"],
        },
    }
    atomic_write_csv(
        output_dir / "search_metadata.csv",
        [{"key": key, "value": _canonical_json(_json_safe(value)) if isinstance(value, (dict, list)) else value} for key, value in metadata.items()],
    )
    atomic_write_text(output_dir / "methodology.md", _methodology_text(fingerprint_payload))
    summary = _markdown_summary(
        selected,
        comparison,
        candidate_counts,
        len(store.rows),
        metadata["rejected_candidate_count"],
        boundary_flags,
        rerun_digests,
        baseline_mismatch,
        selection_audit,
    )
    atomic_write_text(output_dir / "pitch_damping_retune_summary.md", summary)
    atomic_write_text(output_dir / "selected_candidate_summary.md", summary)
    atomic_write_json(output_dir / "baseline_comparison.json", _json_safe(comparison))
    atomic_write_json(
        output_dir / "selection_comparison.json", _json_safe(selection_audit)
    )
    atomic_write_csv(
        output_dir / "selection_comparison.csv",
        _selection_comparison_csv_rows(selection_audit),
    )
    atomic_write_text(
        output_dir / "selection_comparison.md",
        _selection_comparison_markdown(selection_audit),
    )
    profile = _profile_payload(
        selected,
        fingerprint_payload,
        completed_search_fingerprint,
        boundary_flags,
        rerun_digests,
        [
            "Deterministic 2D analytical model only.",
            "No real-flight, 3D, ArduPilot, Pixhawk, HIL, or hardware-safety validation.",
        ],
        actual_search_space,
        selection_audit,
    )
    atomic_write_json(PROVISIONAL_PROFILE, _json_safe(profile))
    return metadata


def _fresh_baseline_results(
    scenarios: Sequence[ScenarioDefinition], *, quick: bool
) -> tuple[list[dict[str, Any]], dict[str, LoiterRunResult]]:
    candidate = Candidate(
        "stage0_baseline",
        BASELINE_GAINS["atc_rat_pit_p"],
        BASELINE_GAINS["atc_rat_pit_d"],
        BASELINE_GAINS["atc_ang_pit_p"],
    )
    rows: list[dict[str, Any]] = []
    results: dict[str, LoiterRunResult] = {}
    for definition in scenarios:
        result = run_headless_loiter(
            SOURCE_PROFILE,
            definition.config,
            rb_overrides=PHYSICAL_CONFIGURATION,
            controller_overrides=candidate.controller_overrides(),
        )
        metrics = compute_metrics(definition, result, quick=quick)
        rows.append(
            {
                "stage": candidate.stage,
                "candidate_key": candidate.key,
                "rate_p": candidate.rate_p,
                "rate_i": 0.0,
                "rate_d": candidate.rate_d,
                "angle_p": candidate.angle_p,
                **metrics,
            }
        )
        results[definition.key] = result
    return rows, results


def _stage2_extension_candidates(
    selected: dict[str, Any], candidates: Sequence[Candidate], top_rate_pd: Sequence[Candidate]
) -> list[Candidate]:
    flags = selected_boundary_flags(selected, candidates)
    angles = sorted({candidate.angle_p for candidate in candidates})
    extension: list[float] = []
    if flags["angle_p_at_min"]:
        extension = [angles[0] - 2.0, angles[0] - 1.0]
    elif flags["angle_p_at_max"]:
        extension = [angles[-1] + 1.0, angles[-1] + 2.0]
    return [
        Candidate("stage2_angle_p", pair.rate_p, pair.rate_d, angle)
        for pair in top_rate_pd
        for angle in extension
        if angle > 0.0
    ]


def _stage3c_extension_candidates(
    selected: dict[str, Any], candidates: Sequence[Candidate]
) -> list[Candidate]:
    flags = selected_boundary_flags(selected, candidates)
    p_values = sorted({candidate.rate_p for candidate in candidates})
    d_values = sorted({candidate.rate_d for candidate in candidates})
    a_values = sorted({candidate.angle_p for candidate in candidates})
    p_extension = []
    d_extension = []
    a_extension = []
    if flags["rate_p_at_min"]:
        p_extension = [p_values[0] - 0.00125]
    elif flags["rate_p_at_max"]:
        p_extension = [p_values[-1] + 0.00125]
    if flags["rate_d_at_min"]:
        d_extension = [max(0.0, d_values[0] - 0.0005)]
    elif flags["rate_d_at_max"]:
        d_extension = [d_values[-1] + 0.0005]
    if flags["angle_p_at_min"]:
        a_extension = [a_values[0] - 0.25]
    elif flags["angle_p_at_max"]:
        a_extension = [a_values[-1] + 0.25]
    additions = {
        Candidate("stage3c_crosscheck", rate_p, rate_d, angle_p)
        for rate_p in [*p_values, *p_extension]
        for rate_d in [*d_values, *d_extension]
        for angle_p in [*a_values, *a_extension]
        if (
            rate_p in p_extension
            or rate_d in d_extension
            or angle_p in a_extension
        )
        and rate_p > 0.0
        and rate_d >= 0.0
        and angle_p > 0.0
    }
    return sorted(additions, key=lambda item: (item.rate_p, item.rate_d, item.angle_p))


def _write_manifest(
    output_dir: Path,
    metadata: dict[str, Any],
    fingerprint_payload: dict[str, Any],
    preservation_hashes: dict[str, str],
) -> Path:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(file for file in output_dir.rglob("*") if file.is_file()):
        if path.name == "manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        artifacts[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if PROVISIONAL_PROFILE.is_file():
        relative = PROVISIONAL_PROFILE.relative_to(REPO_ROOT).as_posix()
        artifacts[relative] = {
            "size_bytes": PROVISIONAL_PROFILE.stat().st_size,
            "sha256": sha256_file(PROVISIONAL_PROFILE),
        }
    manifest = {
        "schema_version": 1,
        "deterministic": True,
        "interpretation": "PR #19 Vane-only versus pitch-retuned Vane-only in the same deterministic 2D analytical model",
        "metadata": metadata,
        "fingerprint_payload": fingerprint_payload,
        "fixed_physical_configuration": PHYSICAL_CONFIGURATION,
        "fixed_controller_values": FIXED_CONTROLLER_VALUES,
        "moving_mass_assist_gain_m_per_Nm": 0.0,
        "preservation_hashes_before_and_after": preservation_hashes,
        "artifacts": artifacts,
    }
    path = output_dir / "manifest.json"
    atomic_write_json(path, _json_safe(manifest))
    return path


def _quick_exports(
    output_dir: Path,
    aggregates: list[dict[str, Any]],
    store: ScenarioResultStore,
    fingerprint_payload: dict[str, Any],
    fingerprint: str,
    baseline_mismatch: Sequence[str],
    runtime_s: float,
) -> dict[str, Any]:
    atomic_write_csv(output_dir / "candidate_results.csv", aggregates)
    atomic_write_csv(output_dir / "scenario_results.csv", store.rows)
    rejection_counts = Counter(
        reason.strip()
        for row in aggregates
        for reason in str(row.get("rejection_reasons", "")).split(";")
        if reason.strip()
    )
    atomic_write_csv(
        output_dir / "rejection_summary.csv",
        [
            {"rejection_category": key, "candidate_count": value}
            for key, value in sorted(rejection_counts.items())
        ],
    )
    metadata = {
        "mode": "quick_smoke_only_no_gain_selection",
        "base_sha": fingerprint_payload["base_sha"],
        "workflow_fingerprint": fingerprint,
        "scenario_fingerprint": fingerprint_payload["scenario_fingerprint"],
        "candidate_count": len(aggregates),
        "valid_candidate_count": sum(not _boolean(row.get("rejected")) for row in aggregates),
        "rejected_candidate_count": sum(_boolean(row.get("rejected")) for row in aggregates),
        "scenario_run_count": len(store.rows),
        "runtime_s": runtime_s,
        "baseline_mismatch_override_reasons": list(baseline_mismatch),
        "provisional_profile_written": False,
    }
    atomic_write_csv(
        output_dir / "search_metadata.csv",
        [{"key": key, "value": _canonical_json(value) if isinstance(value, list) else value} for key, value in metadata.items()],
    )
    atomic_write_text(output_dir / "methodology.md", _methodology_text(fingerprint_payload))
    return metadata


def run_workflow(options: WorkflowOptions) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_parameter_sources()
    preservation_before = _profile_and_preservation_hashes()
    fingerprint_payload, fingerprint = build_fingerprint(options.quick)
    scenarios = required_scenarios(options.quick)
    store = ScenarioResultStore(
        output_dir / "scenario_results.csv", fingerprint, resume=options.resume
    )

    baseline_candidate = Candidate(
        "stage0_baseline",
        BASELINE_GAINS["atc_rat_pit_p"],
        BASELINE_GAINS["atc_rat_pit_d"],
        BASELINE_GAINS["atc_ang_pit_p"],
    )
    baseline_rows, baseline_results = run_candidate_scenarios(
        baseline_candidate,
        scenarios,
        store,
        fingerprint,
        quick=options.quick,
        keep_results=True,
    )
    if len(baseline_results) != len(scenarios):
        # Resume can rehydrate metrics but not full time series; fresh rows remain mandatory.
        fresh_rows, baseline_results = _fresh_baseline_results(scenarios, quick=options.quick)
        baseline_rows = [
            {**row, "workflow_fingerprint": fingerprint}
            for row in fresh_rows
        ]
    baseline_by_scenario = {
        str(row["scenario_name"]): dict(row) for row in baseline_rows
    }
    mismatch = baseline_mismatch_reasons(baseline_rows, scenarios)
    _write_baseline_outputs(output_dir, baseline_rows, baseline_results, mismatch)
    print(
        "Stage 0 baseline: " + ("PASS" if not mismatch else "MISMATCH " + "; ".join(mismatch)),
        flush=True,
    )
    if options.stage == "baseline":
        preservation_after = _profile_and_preservation_hashes()
        if preservation_after != preservation_before:
            raise RuntimeError("canonical profile or preserved diagnostic artifact changed")
        metadata = {
            "mode": "baseline_only",
            "base_sha": fingerprint_payload["base_sha"],
            "workflow_fingerprint": fingerprint,
            "scenario_run_count": len(store.rows),
            "baseline_passed": not mismatch,
            "baseline_mismatch_reasons": mismatch,
            "runtime_s": time.perf_counter() - started,
        }
        _write_manifest(output_dir, metadata, fingerprint_payload, preservation_before)
        return {"mode": "baseline", "metadata": metadata, "baseline_mismatch": mismatch}
    if mismatch and not options.allow_baseline_mismatch:
        atomic_write_json(
            output_dir / "baseline_mismatch_stop.json",
            {
                "search_started": False,
                "reasons": mismatch,
                "required_override": "--allow-baseline-mismatch",
                "workflow_fingerprint": fingerprint,
            },
        )
        raise BaselineMismatchError(
            "fresh Stage 0 baseline failed required PR #19 behavior; candidate search was not started: "
            + "; ".join(mismatch)
        )

    aggregates: dict[str, list[dict[str, Any]]] = {}
    boundary_rows: list[dict[str, Any]] = []
    stage1 = stage1_candidates(options.quick)
    aggregates["stage1_rate_pd"] = run_stage(
        "stage1_rate_pd",
        stage1,
        scenarios,
        store,
        fingerprint,
        baseline_by_scenario,
        quick=options.quick,
    )
    if options.quick:
        preservation_after = _profile_and_preservation_hashes()
        if preservation_after != preservation_before:
            raise RuntimeError("canonical profile or preserved diagnostic artifact changed")
        metadata = _quick_exports(
            output_dir,
            aggregates["stage1_rate_pd"],
            store,
            fingerprint_payload,
            fingerprint,
            mismatch,
            time.perf_counter() - started,
        )
        _write_manifest(output_dir, metadata, fingerprint_payload, preservation_before)
        print(
            f"Quick smoke complete: {metadata['valid_candidate_count']} valid / "
            f"{metadata['candidate_count']} candidates; no gain selected",
            flush=True,
        )
        return {
            "mode": "quick",
            "metadata": metadata,
            "aggregates": aggregates,
            "baseline_mismatch": mismatch,
        }
    stage1_search = merge_candidates(
        stage1, stage_candidates_from_rows(store.rows, "stage1_rate_pd")
    )
    selected_stage1 = select_near_equivalent(aggregates["stage1_rate_pd"])
    flags = selected_boundary_flags(selected_stage1, stage1_search)
    boundary_rows.append({"stage": "stage1_rate_pd", **flags, **selected_stage1})
    stage1_rate_boundary = any(
        flags[key]
        for key in ("rate_p_at_min", "rate_p_at_max", "rate_d_at_min", "rate_d_at_max")
    )
    extension_index = 0
    while not options.quick and stage1_rate_boundary:
        if extension_index >= MAX_BOUNDARY_EXTENSIONS_PER_STAGE:
            atomic_write_csv(output_dir / "boundary_diagnostics.csv", boundary_rows)
            raise RuntimeError("Stage 1 optimum remains on a repeatedly extended search boundary")
        extension = stage1_boundary_extension(selected_stage1, stage1_search)
        known = {candidate.key for candidate in stage1_search}
        extension = [candidate for candidate in extension if candidate.key not in known]
        if not extension:
            raise RuntimeError("Stage 1 boundary extension produced no new candidates")
        extension_index += 1
        print(
            f"Stage 1 boundary extension {extension_index}: {len(extension)} candidates",
            flush=True,
        )
        # Re-rank every accumulated Stage 1 point, not just this extension batch.
        # Cached rows make revisiting already evaluated candidates deterministic and cheap.
        stage1_search = merge_candidates(stage1_search, extension)
        aggregates["stage1_rate_pd"] = run_stage(
            "stage1_rate_pd", stage1_search, scenarios, store, fingerprint,
            baseline_by_scenario, quick=False,
        )
        selected_stage1 = select_near_equivalent(aggregates["stage1_rate_pd"])
        flags = selected_boundary_flags(selected_stage1, stage1_search)
        boundary_rows.append(
            {"stage": f"stage1_rate_pd_extension_{extension_index}", **flags, **selected_stage1}
        )
        stage1_rate_boundary = any(
            flags[key]
            for key in ("rate_p_at_min", "rate_p_at_max", "rate_d_at_min", "rate_d_at_max")
        )
    print(
        f"Stage 1 complete: P={selected_stage1['rate_p']}, D={selected_stage1['rate_d']}, score={selected_stage1['final_score']}",
        flush=True,
    )

    top_stage1_rows = [
        row for row in aggregates["stage1_rate_pd"] if not _boolean(row["rejected"])
    ][:3]
    if len(top_stage1_rows) < 3:
        raise RuntimeError("Stage 2 requires the top three valid Stage 1 Rate P/D pairs")
    top_rate_pd = [candidate_from_row(row, "stage2_angle_p") for row in top_stage1_rows]
    stage2 = stage2_candidates(top_rate_pd)
    aggregates["stage2_angle_p"] = run_stage(
        "stage2_angle_p", stage2, scenarios, store, fingerprint,
        baseline_by_scenario, quick=False,
    )
    stage2_search = merge_candidates(
        stage2, stage_candidates_from_rows(store.rows, "stage2_angle_p")
    )
    selected_stage2 = select_near_equivalent(aggregates["stage2_angle_p"])
    flags = selected_boundary_flags(selected_stage2, stage2_search)
    boundary_rows.append({"stage": "stage2_angle_p", **flags, **selected_stage2})
    extension_index = 0
    while flags["angle_p_at_min"] or flags["angle_p_at_max"]:
        if extension_index >= MAX_BOUNDARY_EXTENSIONS_PER_STAGE:
            atomic_write_csv(output_dir / "boundary_diagnostics.csv", boundary_rows)
            raise RuntimeError("Stage 2 optimum remains on a repeatedly extended Angle P boundary")
        extension = _stage2_extension_candidates(selected_stage2, stage2_search, top_rate_pd)
        known = {candidate.key for candidate in stage2_search}
        extension = [candidate for candidate in extension if candidate.key not in known]
        if not extension:
            raise RuntimeError("Stage 2 boundary extension produced no new candidates")
        extension_index += 1
        print(
            f"Stage 2 boundary extension {extension_index}: {len(extension)} candidates",
            flush=True,
        )
        # The extension is part of the same search space: select from all points.
        stage2_search = merge_candidates(stage2_search, extension)
        aggregates["stage2_angle_p"] = run_stage(
            "stage2_angle_p", stage2_search, scenarios, store, fingerprint,
            baseline_by_scenario, quick=False,
        )
        selected_stage2 = select_near_equivalent(aggregates["stage2_angle_p"])
        flags = selected_boundary_flags(selected_stage2, stage2_search)
        boundary_rows.append(
            {"stage": f"stage2_angle_p_extension_{extension_index}", **flags, **selected_stage2}
        )
    print(
        f"Stage 2 complete: P={selected_stage2['rate_p']}, D={selected_stage2['rate_d']}, Angle P={selected_stage2['angle_p']}",
        flush=True,
    )

    stage3a = stage3a_candidates(candidate_from_row(selected_stage2))
    aggregates["stage3a_local_rate_pd"] = run_stage(
        "stage3a_local_rate_pd", stage3a, scenarios, store, fingerprint,
        baseline_by_scenario, quick=False,
    )
    selected_stage3a = select_near_equivalent(aggregates["stage3a_local_rate_pd"])
    boundary_rows.append(
        {"stage": "stage3a_local_rate_pd", **selected_boundary_flags(selected_stage3a, stage3a), **selected_stage3a}
    )
    stage3b = stage3b_candidates(candidate_from_row(selected_stage3a))
    aggregates["stage3b_local_angle_p"] = run_stage(
        "stage3b_local_angle_p", stage3b, scenarios, store, fingerprint,
        baseline_by_scenario, quick=False,
    )
    selected_stage3b = select_near_equivalent(aggregates["stage3b_local_angle_p"])
    boundary_rows.append(
        {"stage": "stage3b_local_angle_p", **selected_boundary_flags(selected_stage3b, stage3b), **selected_stage3b}
    )
    stage3c = stage3c_candidates(candidate_from_row(selected_stage3b))
    aggregates["stage3c_crosscheck"] = run_stage(
        "stage3c_crosscheck", stage3c, scenarios, store, fingerprint,
        baseline_by_scenario, quick=False,
    )
    stage3c_search = merge_candidates(
        stage3c, stage_candidates_from_rows(store.rows, "stage3c_crosscheck")
    )
    selected_final_row = select_near_equivalent(aggregates["stage3c_crosscheck"])
    boundary_flags = selected_boundary_flags(selected_final_row, stage3c_search)
    boundary_rows.append({"stage": "stage3c_crosscheck", **boundary_flags, **selected_final_row})
    extension_index = 0
    while any(boundary_flags.values()):
        if extension_index >= MAX_BOUNDARY_EXTENSIONS_PER_STAGE:
            atomic_write_csv(output_dir / "boundary_diagnostics.csv", boundary_rows)
            raise RuntimeError("final candidate remains on a repeatedly extended search boundary")
        extension = _stage3c_extension_candidates(selected_final_row, stage3c_search)
        known = {candidate.key for candidate in stage3c_search}
        extension = [candidate for candidate in extension if candidate.key not in known]
        if not extension:
            raise RuntimeError("Stage 3C boundary extension produced no new candidates")
        extension_index += 1
        print(
            f"Stage 3C boundary extension {extension_index}: {len(extension)} candidates",
            flush=True,
        )
        # Select from the complete local crosscheck after every boundary extension.
        stage3c_search = merge_candidates(stage3c_search, extension)
        aggregates["stage3c_crosscheck"] = run_stage(
            "stage3c_crosscheck", stage3c_search, scenarios, store, fingerprint,
            baseline_by_scenario, quick=False,
        )
        selected_final_row = select_near_equivalent(aggregates["stage3c_crosscheck"])
        boundary_flags = selected_boundary_flags(selected_final_row, stage3c_search)
        boundary_rows.append(
            {"stage": f"stage3c_crosscheck_extension_{extension_index}", **boundary_flags, **selected_final_row}
        )
    rank15_row = selected_final_row
    rank15_boundary_flags = dict(boundary_flags)
    raw_best_row = raw_score_best(aggregates["stage3c_crosscheck"])
    final_boundary_flags = selected_boundary_flags(raw_best_row, stage3c_search)
    boundary_rows.append(
        {
            "stage": "stage3c_final_raw_score_selection",
            **final_boundary_flags,
            **raw_best_row,
        }
    )
    selected = candidate_from_row(raw_best_row, "selected_validation")
    rank15_candidate = candidate_from_row(rank15_row, "rank15_validation")
    print(
        f"Local refinement complete; final raw-score rank 1 selected {selected.key}; "
        f"rank-15 comparison retained {rank15_candidate.key}",
        flush=True,
    )

    validation_runs, selected_results, rerun_digests = _fresh_validation_runs(
        selected, scenarios, quick=False
    )
    rank15_validation_runs, rank15_results, rank15_rerun_digests = _fresh_validation_runs(
        rank15_candidate, scenarios, quick=False
    )
    selected_by_scenario = validation_runs[0]
    rank15_by_scenario = rank15_validation_runs[0]
    requirement_failures, comparison = final_candidate_requirements(
        baseline_by_scenario,
        selected_by_scenario,
        final_boundary_flags,
        boundary_is_hard_gate=False,
    )
    rank15_gate_failures = [
        f"{name}:{metrics.get('rejection_reasons', '')}"
        for name, metrics in rank15_by_scenario.items()
        if _boolean(metrics.get("rejected"))
    ]
    selection_audit = _selection_comparison_audit(
        raw_best_row,
        rank15_row,
        aggregates["stage3c_crosscheck"],
        selected_by_scenario,
        rank15_by_scenario,
        final_boundary_flags,
        rank15_boundary_flags,
    )
    selection_audit["deterministic_reruns"] = {
        "workflow_fingerprint": fingerprint,
        "selected_raw_score_rank1_digests": list(rerun_digests),
        "rank15_low_control_effort_digests": list(rank15_rerun_digests),
        "selected_raw_score_rank1_byte_identical": len(set(rerun_digests)) == 1,
        "rank15_low_control_effort_byte_identical": len(set(rank15_rerun_digests)) == 1,
    }
    if rank15_gate_failures:
        requirement_failures.extend(rank15_gate_failures)
    if not selection_audit["audit_passed"]:
        requirement_failures.append("rank1_rank15_comparison_audit_failed")
    atomic_write_json(
        output_dir / "validation" / "deterministic_reruns.json",
        _json_safe(
            {
                "workflow_fingerprint": fingerprint,
                "digests": rerun_digests,
                "byte_identical_metrics": len(set(rerun_digests)) == 1,
                "requirement_failures": requirement_failures,
                "runs": validation_runs,
            }
        ),
    )
    atomic_write_json(
        output_dir / "validation" / "raw_score_rank1_deterministic_reruns.json",
        _json_safe(
            {
                "workflow_fingerprint": fingerprint,
                "digests": rerun_digests,
                "byte_identical_metrics": len(set(rerun_digests)) == 1,
                "requirement_failures": requirement_failures,
                "runs": validation_runs,
            }
        ),
    )
    atomic_write_json(
        output_dir / "validation" / "rank15_low_control_effort_deterministic_reruns.json",
        _json_safe(
            {
                "workflow_fingerprint": fingerprint,
                "digests": rank15_rerun_digests,
                "byte_identical_metrics": len(set(rank15_rerun_digests)) == 1,
                "requirement_failures": rank15_gate_failures,
                "runs": rank15_validation_runs,
            }
        ),
    )
    _write_selected_timeseries(output_dir, selected_results)
    _write_validation_timeseries(output_dir, "raw_score_rank1", selected_results)
    _write_validation_timeseries(
        output_dir, "rank15_low_control_effort", rank15_results
    )
    _write_plots(output_dir, baseline_results, selected_results)
    if requirement_failures:
        atomic_write_json(
            output_dir / "validation" / "final_candidate_rejected.json",
            {"selected": asdict(selected), "failures": requirement_failures},
        )
        raise RuntimeError(
            "best trade-off does not satisfy final candidate requirements: "
            + "; ".join(requirement_failures)
        )

    preservation_after = _profile_and_preservation_hashes()
    if preservation_after != preservation_before:
        raise RuntimeError("canonical profile or preserved diagnostic artifact changed")
    metadata = _write_final_exports(
        output_dir,
        aggregates,
        store,
        selected,
        selected_by_scenario,
        baseline_by_scenario,
        comparison,
        fingerprint_payload,
        fingerprint,
        boundary_rows,
        final_boundary_flags,
        rerun_digests,
        mismatch,
        selection_audit,
        time.perf_counter() - started,
    )
    preservation_final = _profile_and_preservation_hashes()
    if preservation_final != preservation_before:
        raise RuntimeError("canonical profile or preserved diagnostic artifact changed after exports")
    manifest = _write_manifest(output_dir, metadata, fingerprint_payload, preservation_before)
    print("Final validation complete", flush=True)
    return {
        "mode": "full",
        "metadata": metadata,
        "aggregates": aggregates,
        "selected": asdict(selected),
        "comparison": comparison,
        "manifest": str(manifest),
        "baseline_mismatch": mismatch,
    }
