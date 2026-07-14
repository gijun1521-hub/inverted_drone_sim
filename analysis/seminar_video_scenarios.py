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
LOITER_PARAMETER_FILE = REPO_ROOT / "params" / "loiter_tuned_vane_only.json"
MOVING_MASS_PARAMETER_FILE = REPO_ROOT / "params" / "moving_mass_prototype_2kg_tuned.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "seminar_videos"
TAIL_WINDOW_S = 2.0
ASSIST_GAIN_M_PER_NM = 0.055

SELECTED_CONTROLLER_VALUES = {
    "atc_rat_pit_p": 0.07,
    "atc_rat_pit_i": 0.0,
    "atc_rat_pit_d": 0.008,
    "atc_ang_pit_p": 10.0,
    "psc_ne_pos_p": 0.5,
    "psc_ne_vel_p": 0.9,
}

SHARED_RIGID_BODY_OVERRIDES = {
    "m": 2.0,
    "moving_mass": {
        "enabled": True,
        "mass_kg": 0.5,
        "max_offset_m": 0.05,
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
    "ground_contact",
    "rejected",
    "rejection_reasons",
    "settled",
    "tail_window_s",
    "duration_s",
    "physics_dt_s",
    "controller_dt_s",
    "simulation_sample_count",
]


@dataclass(frozen=True)
class SeminarScenarioDefinition:
    key: str
    display_name: str
    config: LoiterScenarioConfig
    settling_reference_time_s: float


@dataclass(frozen=True)
class SeminarVariantDefinition:
    key: str
    display_name: str
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
        ),
    )


def seminar_variants() -> tuple[SeminarVariantDefinition, ...]:
    return (
        SeminarVariantDefinition("locked", "Mass locked at center", 0.0),
        SeminarVariantDefinition("assist", "Active moving-mass assist", ASSIST_GAIN_M_PER_NM),
    )


def _effective_configs() -> tuple[RigidBodyConfig, InteractiveSimConfig, ControllerConfig]:
    rb_cfg, ui_cfg, controller_cfg = load_interactive_config(MOVING_MASS_PARAMETER_FILE)
    rb_cfg = apply_dataclass_overrides(rb_cfg, SHARED_RIGID_BODY_OVERRIDES, "seminar rigid body")
    controller_cfg = apply_dataclass_overrides(
        controller_cfg, SELECTED_CONTROLLER_VALUES, "seminar controller"
    )
    return rb_cfg, ui_cfg, controller_cfg


def validate_parameter_sources() -> None:
    """Ensure both selected immutable profiles carry the requested controller tune."""
    for path in (LOITER_PARAMETER_FILE, MOVING_MASS_PARAMETER_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
        _rb, _ui, controller = load_interactive_config(path)
        for key, expected in SELECTED_CONTROLLER_VALUES.items():
            actual = float(getattr(controller, key))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{path}: {key}={actual!r}, expected {expected!r}")


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
        MOVING_MASS_PARAMETER_FILE,
        effective_scenario,
        rb_overrides=SHARED_RIGID_BODY_OVERRIDES,
        controller_overrides=SELECTED_CONTROLLER_VALUES,
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

    if scenario.key == "forward_1m":
        post_event = times >= scenario.settling_reference_time_s - 1e-12
        position_overshoot = (
            float(max(0.0, np.max(x[post_event]) - 1.0)) if np.any(post_event) else 0.0
        )
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
    rejection_reasons: list[str] = []
    if crash_reasons:
        rejection_reasons.extend(sorted(crash_reasons))
    if not np.all(np.isfinite(numeric_series)):
        rejection_reasons.append("non-finite simulation data")

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
        "ground_contact": bool(ground_contact),
        "rejected": bool(rejection_reasons),
        "rejection_reasons": "; ".join(rejection_reasons),
        "settled": bool(settled),
        "tail_window_s": TAIL_WINDOW_S,
        "duration_s": float(scenario.config.duration_s),
        "physics_dt_s": float(rows[0]["physics_dt"]),
        "controller_dt_s": float(rows[0]["controller_dt"]),
        "simulation_sample_count": len(rows),
    }
    for key, value in metric.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ValueError(f"{scenario.key}/{variant.key}: non-finite metric {key}={value!r}")
    return metric


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

    for result in results:
        rb = result.rb_config
        mm = rb.moving_mass
        checks = {
            "total mass": math.isclose(rb.m, 2.0),
            "moving mass": math.isclose(mm.mass_kg, 0.5),
            "rail": math.isclose(mm.max_offset_m, 0.05),
            "body-up offset": math.isclose(mm.moving_mass_body_up_offset_m, 0.12),
            "moving mass enabled": mm.enabled,
            "total-COM geometry": mm.use_total_com_geometry,
            "legacy moment disabled": not mm.use_legacy_gravity_offset_moment,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{result.key}: effective parameter mismatch: {', '.join(failed)}")
        for key, expected in SELECTED_CONTROLLER_VALUES.items():
            if not math.isclose(float(getattr(result.controller_config, key)), expected, abs_tol=1e-12):
                raise ValueError(f"{result.key}: controller mismatch for {key}")
        if result.variant.key == "locked":
            offsets = _array(result.run.rows, "moving_mass_offset_m")
            targets = _array(result.run.rows, "moving_mass_target_m")
            if not (np.all(offsets == 0.0) and np.all(targets == 0.0)):
                raise ValueError(f"{result.key}: locked moving mass did not remain centered")

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
    lines = [
        "# Seminar scenario comparison",
        "",
        "These deterministic 2D simulations compare the same 2.0 kg vehicle with its 0.5 kg moving mass physically present in both variants. The locked case commands and maintains a 0 mm offset; it does not remove the mass.",
        "",
        "The final 2.0 seconds form the tail window. `tail_rms_x_m` is RMS target error. Settling requires the remaining run to stay within 0.05 m position error and 0.05 m/s horizontal speed; an unsettled run is reported at the available observation-window limit. For the disturbance scenario, `position_overshoot_m` is the peak absolute recovery excursion after the force ends. For the +1 m step it is excursion beyond +1.0 m.",
        "",
        "| Scenario | Variant | Tail RMS x (m) | Final |x error| (m) | Peak |pitch| (deg) | Vane RMS (deg) | Moving mass max (mm) | Settled |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for scenario in seminar_scenarios(results[0].scenario.config.duration_s):
        for variant in seminar_variants():
            row = by_key[(scenario.key, variant.key)]
            lines.append(
                f"| {scenario.display_name} | {variant.display_name} | "
                f"{row['tail_rms_x_m']:.5f} | {row['final_abs_x_error_m']:.5f} | "
                f"{row['peak_abs_theta_deg']:.3f} | {row['vane_command_rms_deg']:.3f} | "
                f"{1000.0 * row['moving_mass_max_offset_m']:.2f} | "
                f"{'yes' if row['settled'] else 'no'} |"
            )
    lines.extend(["", "## Pairwise comparison", ""])
    for scenario in seminar_scenarios(results[0].scenario.config.duration_s):
        locked = by_key[(scenario.key, "locked")]
        assist = by_key[(scenario.key, "assist")]
        delta = assist["tail_rms_x_m"] - locked["tail_rms_x_m"]
        lines.extend(
            [
                f"For **{scenario.display_name}**, active assist changes tail RMS x error by {delta:+.5f} m relative to the locked-mass simulation.",
                "",
            ]
        )
    lines.extend(
        [
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


def write_manifest(
    results: Iterable[SeminarRunResult],
    output_dir: str | Path,
    render_report: dict[str, Any],
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

    manifest = {
        "schema_version": 1,
        "generator": "generate_seminar_videos.py",
        "parameter_files": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in (LOITER_PARAMETER_FILE, MOVING_MASS_PARAMETER_FILE)
        ],
        "selected_controller_values": SELECTED_CONTROLLER_VALUES,
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
        "render": manifest_render,
        "metrics": [result.metrics for result in results],
        "artifacts": artifacts,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
