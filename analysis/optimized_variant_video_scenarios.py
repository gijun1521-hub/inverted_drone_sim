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
    from ..params import load_interactive_config
    from .headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter
except ImportError:  # pragma: no cover - direct execution from repository root
    from config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from params import load_interactive_config
    from analysis.headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "results" / "analysis" / "variant_controller_optimization" / "profiles"
VANE_ONLY_PROFILE = PROFILE_DIR / "vane_only.json"
MOVING_MASS_ASSIST_PROFILE = PROFILE_DIR / "moving_mass_assist.json"
PR25_RESULTS = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "variant_controller_optimization"
    / "validation"
    / "selected_scenario_results.csv"
)
PR24_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "final_seminar_videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "final_optimized_controller_videos"
TAIL_WINDOW_S = 2.0

CONTROLLER_KEYS = (
    "atc_rat_pit_p",
    "atc_rat_pit_i",
    "atc_rat_pit_d",
    "atc_ang_pit_p",
    "psc_ne_pos_p",
    "psc_ne_vel_p",
)

PR25_METRIC_TOLERANCES = {
    "peak_abs_theta_deg": 0.25,
    "position_overshoot_m": 0.02,
    "final_abs_x_error_m": 0.03,
    "moving_mass_max_target_m": 0.002,
    "moving_mass_max_offset_m": 0.002,
}

METRIC_COLUMNS = [
    "scenario",
    "variant",
    "profile_source",
    "assist_gain_m_per_Nm",
    *CONTROLLER_KEYS,
    "tail_rms_x_m",
    "tail_rms_vx_m_s",
    "final_abs_x_error_m",
    "position_overshoot_m",
    "settling_time_s",
    "settled",
    "peak_abs_theta_deg",
    "vane_command_rms_deg",
    "vane_command_max_deg",
    "moving_mass_max_target_m",
    "moving_mass_max_offset_m",
    "moving_mass_max_rate_m_s",
    "moving_mass_max_acceleration_m_s2",
    "ground_contact",
    "duration_s",
    "physics_dt_s",
    "controller_dt_s",
    "simulation_sample_count",
]


@dataclass(frozen=True)
class OptimizedScenarioDefinition:
    key: str
    display_name: str
    config: LoiterScenarioConfig
    settling_reference_time_s: float


@dataclass(frozen=True)
class OptimizedVariantDefinition:
    key: str
    display_name: str
    hud_mass_label: str
    profile_path: Path
    profile_source: str
    assist_gain_m_per_Nm: float


@dataclass(frozen=True)
class OptimizedRunResult:
    scenario: OptimizedScenarioDefinition
    variant: OptimizedVariantDefinition
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): _sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def optimized_scenarios(duration_s: float = 10.0) -> tuple[OptimizedScenarioDefinition, ...]:
    return (
        OptimizedScenarioDefinition(
            key="loiter",
            display_name="LOITER: 8 N world-frame disturbance",
            config=LoiterScenarioConfig(
                name="final_optimized_loiter_horizontal_disturbance",
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
                    "PR #24 seminar timing and initialization: 8 N world-frame horizontal "
                    "disturbance from 1.5 s through 1.7 s."
                ),
            ),
            settling_reference_time_s=1.7,
        ),
        OptimizedScenarioDefinition(
            key="forward_1m",
            display_name="LOITER: absolute +1 m command",
            config=LoiterScenarioConfig(
                name="final_optimized_forward_1m_step",
                duration_s=float(duration_s),
                initial_x=0.0,
                initial_z=1.0,
                initial_theta_deg=0.0,
                target_x=0.0,
                target_z=1.0,
                target_step_time_s=1.0,
                target_step_x=1.0,
                notes="Absolute x target changes from 0 m to +1 m at t=1.0 s and is held.",
            ),
            settling_reference_time_s=1.0,
        ),
    )


def _read_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_gain(profile: dict[str, Any]) -> float:
    optimization = profile.get("analysis", {}).get("variant_controller_optimization", {})
    candidate = optimization.get("selected_candidate", {})
    if "moving_mass_assist_gain_m_per_Nm" not in candidate:
        raise ValueError("profile lacks the PR #25 selected-candidate assist gain")
    return float(candidate["moving_mass_assist_gain_m_per_Nm"])


def expected_profile_values() -> dict[str, dict[str, float]]:
    """Load validation references from PR #25 instead of duplicating selected gains."""
    field_map = {
        "atc_rat_pit_p": "effective_atc_rat_pit_p",
        "atc_rat_pit_i": "effective_atc_rat_pit_i",
        "atc_rat_pit_d": "effective_atc_rat_pit_d",
        "atc_ang_pit_p": "effective_atc_ang_pit_p",
        "psc_ne_pos_p": "effective_psc_ne_pos_p",
        "psc_ne_vel_p": "effective_psc_ne_vel_p",
        "assist_gain_m_per_Nm": "moving_mass_assist_gain_m_per_Nm",
    }
    by_variant: dict[str, list[dict[str, float]]] = {}
    with PR25_RESULTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["scenario_name"] not in {"forward_1m", "loiter_positive_disturbance"}:
                continue
            by_variant.setdefault(row["variant"], []).append(
                {runtime: float(row[source]) for runtime, source in field_map.items()}
            )
    if set(by_variant) != {"vane_only", "moving_mass_assist"}:
        raise ValueError("PR #25 selected results lack both optimized variants")
    expected: dict[str, dict[str, float]] = {}
    for variant, rows in by_variant.items():
        if len(rows) != 2 or rows[0] != rows[1]:
            raise ValueError(f"PR #25 effective values are inconsistent for {variant}")
        expected[variant] = rows[0]
    return expected


def optimized_variants() -> tuple[OptimizedVariantDefinition, ...]:
    definitions = (
        (
            "vane_only",
            "Independently optimized Vane-only",
            "Physical moving mass locked at center",
            VANE_ONLY_PROFILE,
        ),
        (
            "moving_mass_assist",
            "Independently optimized moving-mass assist",
            "Optimized moving-mass assist",
            MOVING_MASS_ASSIST_PROFILE,
        ),
    )
    variants = []
    for key, display_name, hud_mass_label, path in definitions:
        profile = _read_profile(path)
        variants.append(
            OptimizedVariantDefinition(
                key=key,
                display_name=display_name,
                hud_mass_label=hud_mass_label,
                profile_path=path,
                profile_source=path.relative_to(REPO_ROOT).as_posix(),
                assist_gain_m_per_Nm=_selected_gain(profile),
            )
        )
    return tuple(variants)


def validate_profile_sources() -> dict[str, Any]:
    report: dict[str, Any] = {"profiles": {}}
    expected_values = expected_profile_values()
    for variant in optimized_variants():
        profile = _read_profile(variant.profile_path)
        expected = expected_values[variant.key]
        controller = profile.get("controller", {})
        for key in CONTROLLER_KEYS:
            actual = float(controller[key])
            if not math.isclose(actual, expected[key], rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{variant.profile_source}: {key}={actual!r}, expected {expected[key]!r}")
        if not math.isclose(
            variant.assist_gain_m_per_Nm,
            expected["assist_gain_m_per_Nm"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{variant.profile_source}: selected assist gain mismatch")
        report["profiles"][variant.key] = {
            "path": variant.profile_source,
            "sha256": _sha256(variant.profile_path),
            "controller": {key: float(controller[key]) for key in CONTROLLER_KEYS},
            "assist_gain_m_per_Nm": variant.assist_gain_m_per_Nm,
        }
    if report["profiles"]["vane_only"]["controller"] == report["profiles"]["moving_mass_assist"]["controller"]:
        raise ValueError("optimized variants unexpectedly use identical controllers")
    return report


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _settling_metric(
    times: np.ndarray,
    x_error: np.ndarray,
    vx: np.ndarray,
    reference_time_s: float,
    duration_s: float,
) -> tuple[float, bool]:
    mask = times >= reference_time_s - 1e-12
    event_times = times[mask]
    if event_times.size == 0:
        return max(0.0, duration_s - reference_time_s), False
    suffix_error = np.maximum.accumulate(np.abs(x_error[mask])[::-1])[::-1]
    suffix_vx = np.maximum.accumulate(np.abs(vx[mask])[::-1])[::-1]
    candidates = np.flatnonzero((suffix_error <= 0.05) & (suffix_vx <= 0.05))
    if candidates.size:
        return float(max(0.0, event_times[candidates[0]] - reference_time_s)), True
    return max(0.0, duration_s - reference_time_s), False


def compute_metrics(
    scenario: OptimizedScenarioDefinition,
    variant: OptimizedVariantDefinition,
    run: LoiterRunResult,
    controller: ControllerConfig,
) -> dict[str, Any]:
    rows = run.rows
    if not rows:
        raise ValueError(f"{scenario.key}/{variant.key}: simulation produced no rows")
    times = _array(rows, "sim_time")
    x = _array(rows, "x_cg")
    vx = _array(rows, "vx")
    theta_deg = np.rad2deg(_array(rows, "theta"))
    vane_command_deg = np.rad2deg(_array(rows, "vane_angle_cmd"))
    mass_target = _array(rows, "moving_mass_target_m")
    mass_offset = _array(rows, "moving_mass_offset_m")
    mass_rate = _array(rows, "moving_mass_velocity_m_s")
    x_error = _array(rows, "x_error")
    tail = times >= max(0.0, scenario.config.duration_s - TAIL_WINDOW_S) - 1e-12
    dt = float(rows[0]["physics_dt"])
    mass_accel = np.diff(mass_rate, prepend=0.0) / dt
    settling_time, settled = _settling_metric(
        times, x_error, vx, scenario.settling_reference_time_s, scenario.config.duration_s
    )
    post_event = times >= scenario.settling_reference_time_s - 1e-12
    if scenario.key == "forward_1m":
        overshoot = float(max(0.0, np.max(x[post_event]) - 1.0)) if np.any(post_event) else 0.0
    else:
        overshoot = float(np.max(np.abs(x_error[post_event]))) if np.any(post_event) else 0.0
    ground_contact = any(
        str(row.get("crash_reason", "")) == "ground contact"
        or float(row.get("min_body_z", 1.0)) <= 0.0
        for row in rows
    )
    metric: dict[str, Any] = {
        "scenario": scenario.key,
        "variant": variant.key,
        "profile_source": variant.profile_source,
        "assist_gain_m_per_Nm": variant.assist_gain_m_per_Nm,
        **{key: float(getattr(controller, key)) for key in CONTROLLER_KEYS},
        "tail_rms_x_m": _rms(x_error[tail]),
        "tail_rms_vx_m_s": _rms(vx[tail]),
        "final_abs_x_error_m": float(abs(x_error[-1])),
        "position_overshoot_m": overshoot,
        "settling_time_s": settling_time,
        "settled": settled,
        "peak_abs_theta_deg": float(np.max(np.abs(theta_deg))),
        "vane_command_rms_deg": _rms(vane_command_deg),
        "vane_command_max_deg": float(np.max(np.abs(vane_command_deg))),
        "moving_mass_max_target_m": float(np.max(np.abs(mass_target))),
        "moving_mass_max_offset_m": float(np.max(np.abs(mass_offset))),
        "moving_mass_max_rate_m_s": float(np.max(np.abs(mass_rate))),
        "moving_mass_max_acceleration_m_s2": float(np.max(np.abs(mass_accel))),
        "ground_contact": ground_contact,
        "duration_s": float(scenario.config.duration_s),
        "physics_dt_s": dt,
        "controller_dt_s": float(rows[0]["controller_dt"]),
        "simulation_sample_count": len(rows),
    }
    for key, value in metric.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ValueError(f"{scenario.key}/{variant.key}: non-finite metric {key}={value!r}")
    return metric


def run_optimized_variant(
    scenario: OptimizedScenarioDefinition,
    variant: OptimizedVariantDefinition,
) -> OptimizedRunResult:
    effective_scenario = replace(
        scenario.config,
        moving_mass_enabled=True,
        moving_mass_target_m=0.0,
        moving_mass_assist_gain_m_per_Nm=variant.assist_gain_m_per_Nm,
    )
    run = run_headless_loiter(variant.profile_path, effective_scenario)
    rb_config, ui_config, controller_config = load_interactive_config(variant.profile_path)
    return OptimizedRunResult(
        scenario=scenario,
        variant=variant,
        run=run,
        metrics=compute_metrics(scenario, variant, run, controller_config),
        rb_config=rb_config,
        ui_config=ui_config,
        controller_config=controller_config,
    )


def run_all_scenarios(duration_s: float = 10.0) -> list[OptimizedRunResult]:
    validate_profile_sources()
    results = [
        run_optimized_variant(scenario, variant)
        for scenario in optimized_scenarios(duration_s)
        for variant in optimized_variants()
    ]
    validate_result_set(results)
    return results


def validate_result_set(results: Iterable[OptimizedRunResult]) -> dict[str, Any]:
    results = list(results)
    keys = [result.key for result in results]
    expected_keys = {
        (scenario, variant)
        for scenario in ("loiter", "forward_1m")
        for variant in ("vane_only", "moving_mass_assist")
    }
    if set(keys) != expected_keys or len(keys) != 4:
        raise ValueError(f"expected four optimized result pairs, got {keys!r}")
    if len({len(result.run.rows) for result in results}) != 1:
        raise ValueError("optimized simulations do not share a timestep/sample count")
    by_key = {result.key: result for result in results}
    for scenario in ("loiter", "forward_1m"):
        left = by_key[(scenario, "vane_only")]
        right = by_key[(scenario, "moving_mass_assist")]
        if left.scenario.config != right.scenario.config:
            raise ValueError(f"{scenario}: base scenario conditions differ")
        for attr in ("m", "H", "W", "vane_angle_max_deg", "vane_rate_limit_deg_s"):
            if getattr(left.rb_config, attr) != getattr(right.rb_config, attr):
                raise ValueError(f"{scenario}: vehicle parameter {attr} differs")
    expected_values = expected_profile_values()
    for result in results:
        expected = expected_values[result.variant.key]
        for key in CONTROLLER_KEYS:
            if not math.isclose(float(getattr(result.controller_config, key)), expected[key], abs_tol=1e-15):
                raise ValueError(f"{result.key}: runtime controller mismatch for {key}")
        rows = result.run.rows
        if result.variant.key == "vane_only":
            exact_zero_keys = (
                "moving_mass_target_m",
                "moving_mass_offset_m",
                "moving_mass_velocity_m_s",
            )
            if result.variant.assist_gain_m_per_Nm != 0.0:
                raise ValueError("Vane-only runtime assist gain is not exactly zero")
            for key in exact_zero_keys:
                if not np.all(_array(rows, key) == 0.0):
                    raise ValueError(f"{result.key}: {key} is not exactly zero")
            rates = _array(rows, "moving_mass_velocity_m_s")
            if not np.all(np.diff(rates, prepend=0.0) == 0.0):
                raise ValueError(f"{result.key}: moving-mass acceleration is not exactly zero")
        elif not math.isclose(
            result.variant.assist_gain_m_per_Nm,
            expected_values["moving_mass_assist"]["assist_gain_m_per_Nm"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("assist runtime gain does not match the PR #25 selection")
    return {
        "result_count": len(results),
        "unique_key_count": len(set(keys)),
        "simulation_sample_count": len(results[0].run.rows),
        "controllers_are_distinct": True,
        "vane_only_mass_state_exact_zero": True,
        "assist_gain_verified": True,
    }


def compare_to_pr25(results: Iterable[OptimizedRunResult]) -> dict[str, Any]:
    results = list(results)
    committed: dict[tuple[str, str], dict[str, str]] = {}
    with PR25_RESULTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scenario = str(row["scenario_name"])
            if scenario in {"forward_1m", "loiter_positive_disturbance"}:
                committed[("loiter" if scenario.startswith("loiter") else scenario, row["variant"])] = row
    comparisons = []
    field_map = {
        "peak_abs_theta_deg": "peak_abs_pitch_deg",
        "position_overshoot_m": "position_overshoot_m",
        "final_abs_x_error_m": "final_abs_position_error_m",
        "moving_mass_max_target_m": "moving_mass_max_abs_target_m",
        "moving_mass_max_offset_m": "moving_mass_max_abs_offset_m",
    }
    full_horizon = min(result.scenario.config.duration_s for result in results) >= 9.5
    for result in results:
        reference = committed[result.key]
        for runtime_name, reference_name in (field_map.items() if full_horizon else ()):
            actual = float(result.metrics[runtime_name])
            expected = float(reference[reference_name])
            tolerance = PR25_METRIC_TOLERANCES[runtime_name]
            delta = abs(actual - expected)
            comparisons.append(
                {
                    "scenario": result.scenario.key,
                    "variant": result.variant.key,
                    "metric": runtime_name,
                    "actual": actual,
                    "pr25_committed": expected,
                    "absolute_tolerance": tolerance,
                    "absolute_delta": delta,
                    "passed": delta <= tolerance,
                }
            )
        parameter_map = {
            "atc_rat_pit_p": "effective_atc_rat_pit_p",
            "atc_rat_pit_i": "effective_atc_rat_pit_i",
            "atc_rat_pit_d": "effective_atc_rat_pit_d",
            "atc_ang_pit_p": "effective_atc_ang_pit_p",
            "psc_ne_pos_p": "effective_psc_ne_pos_p",
            "psc_ne_vel_p": "effective_psc_ne_vel_p",
            "assist_gain_m_per_Nm": "moving_mass_assist_gain_m_per_Nm",
        }
        for runtime_name, reference_name in parameter_map.items():
            actual = float(result.metrics[runtime_name])
            expected = float(reference[reference_name])
            delta = abs(actual - expected)
            comparisons.append(
                {
                    "scenario": result.scenario.key,
                    "variant": result.variant.key,
                    "metric": runtime_name,
                    "actual": actual,
                    "pr25_committed": expected,
                    "absolute_tolerance": 1e-15,
                    "absolute_delta": delta,
                    "passed": delta <= 1e-15,
                }
            )
    failed = [row for row in comparisons if not row["passed"]]
    if failed:
        raise ValueError(f"PR #25 selected-result comparison failed: {failed!r}")
    return {
        "source": PR25_RESULTS.relative_to(REPO_ROOT).as_posix(),
        "note": "Video events use PR #24 timing; tolerances cover time translation and the 10 s video horizon.",
        "dynamic_metrics_skipped_for_short_horizon": not full_horizon,
        "passed": True,
        "comparison_count": len(comparisons),
        "tolerances": PR25_METRIC_TOLERANCES,
        "comparisons": comparisons,
    }


def write_metrics_csv(results: Iterable[OptimizedRunResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result.metrics for result in results)
    return path


def write_summary_markdown(
    results: Iterable[OptimizedRunResult],
    path: str | Path,
    pr25_comparison: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    lines = [
        "# Final independently optimized controller videos",
        "",
        "These deterministic 2D simulation-only renders use the two full-precision profiles selected in PR #25. They do not claim hardware, HIL, or real-flight validation.",
        "",
        "Both variants use identical scenario definitions, vehicle/moving-mass physics, actuator limits, integration timesteps, camera bounds, frame timing, and resolution. Only the profile-selected controller and assist behavior differ.",
        "",
        "| Scenario | Variant | Profile | Gain (m/Nm) | Final |x error| (m) | Overshoot/excursion (m) | Settling (s) | Peak pitch (deg) | Mass max (mm) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        row = result.metrics
        lines.append(
            f"| {result.scenario.display_name} | {result.variant.display_name} | `{result.variant.profile_source}` | "
            f"{row['assist_gain_m_per_Nm']:.17g} | {row['final_abs_x_error_m']:.6f} | "
            f"{row['position_overshoot_m']:.6f} | {row['settling_time_s']:.3f} | "
            f"{row['peak_abs_theta_deg']:.3f} | {1000.0 * row['moving_mass_max_offset_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- PR #25 selected-scenario comparisons: **PASS** ({pr25_comparison['comparison_count']} explicit checks).",
            "- Vane-only assist gain, target, offset, velocity, and acceleration: **exactly zero**.",
            "- Moving-mass assist gain: **loaded from and matched to the PR #25 selected candidate**.",
            "- Physical moving mass remains installed in both variants; the Vane-only rail state is locked at center.",
            "",
        ]
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    return path


def write_manifest(
    results: Iterable[OptimizedRunResult],
    output_dir: str | Path,
    render_report: dict[str, Any],
    validation: dict[str, Any],
    pr25_comparison: dict[str, Any],
    pr24_hashes_before: dict[str, str],
) -> Path:
    results = list(results)
    output_dir = Path(output_dir)
    pr24_hashes_after = directory_hashes(PR24_OUTPUT_DIR)
    if pr24_hashes_before != pr24_hashes_after:
        raise RuntimeError("PR #24 final seminar artifacts changed during generation")
    render = json.loads(json.dumps(render_report))
    executable = render.get("encoder", {}).get("executable")
    if executable:
        render["encoder"]["executable"] = Path(executable).name
    artifacts = []
    for name in sorted(render_report["artifacts"]):
        artifact = output_dir / name
        if artifact.is_file():
            artifacts.append({"name": name, "size_bytes": artifact.stat().st_size, "sha256": _sha256(artifact)})
    manifest = {
        "schema_version": 1,
        "generator": "generate_optimized_variant_videos.py",
        "deterministic": True,
        "simulation_only": True,
        "real_flight_claim": False,
        "profiles": validate_profile_sources()["profiles"],
        "scenarios": [asdict(item) for item in optimized_scenarios(results[0].scenario.config.duration_s)],
        "variants": [
            {
                **asdict(item),
                "profile_path": item.profile_source,
            }
            for item in optimized_variants()
        ],
        "validation": validation,
        "pr25_selected_result_comparison": pr25_comparison,
        "pr24_artifacts_preserved": True,
        "pr24_artifact_hashes": pr24_hashes_after,
        "render": render,
        "metrics": [result.metrics for result in results],
        "artifacts": artifacts,
    }
    path = output_dir / "manifest.json"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n")
    return path
