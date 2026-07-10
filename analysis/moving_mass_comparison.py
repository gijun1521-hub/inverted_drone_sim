from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

try:
    from .headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter, scenario_by_name
except ImportError:  # pragma: no cover - supports top-level script execution
    from analysis.headless_loiter import LoiterRunResult, LoiterScenarioConfig, run_headless_loiter, scenario_by_name


DEFAULT_SCENARIOS = [
    "pitch_assist_probe",
    "stick_move_release",
    "horizontal_impulse_recovery",
    "initial_x_offset_recovery",
    "authority_stress",
]

METRIC_FIELDS = [
    "final_abs_x_error",
    "rms_x_error",
    "final_abs_z_error",
    "rms_z_error",
    "max_theta_deg",
    "rms_theta_deg",
    "final_theta_deg",
    "max_omega_deg_s",
    "max_vane_cmd_deg",
    "max_vane_actual_deg",
    "mixer_saturation_percent",
    "mixer_angle_saturation_percent",
    "authority_limited_percent",
    "servo_rate_saturation_percent",
    "motor_saturation_percent",
    "moving_mass_max_offset_m",
    "moving_mass_saturation_percent",
    "effective_moving_mass_kg",
    "effective_moving_mass_max_offset_m",
    "effective_moving_mass_max_rate_m_s",
    "effective_moving_mass_max_accel_m_s2",
    "max_abs_total_com_body_right_m",
    "max_abs_total_com_body_up_m",
    "max_abs_thrust_moment_from_com_offset",
    "rms_thrust_moment_from_com_offset",
    "max_abs_vane_moment_about_total_com",
    "max_abs_legacy_moving_mass_moment",
]

METADATA_FIELDS = [
    "moving_mass_model_mode",
    "baseline_variant",
    "mode_baseline_variant",
    "total_com_geometry_active",
    "legacy_gravity_offset_active",
    "state_dimension",
    "total_mass_kg",
    "moving_mass_mass_kg",
    "moving_mass_body_up_offset_m",
]

DERIVED_FIELDS = [
    "delta_max_theta_deg",
    "delta_rms_theta_deg",
    "delta_final_abs_x_error",
    "delta_rms_x_error",
    "attitude_improvement_score",
    "delta_vs_mode_baseline_max_theta_deg",
    "delta_vs_mode_baseline_rms_theta_deg",
    "delta_vs_mode_baseline_final_abs_x_error",
    "delta_vs_mode_baseline_rms_x_error",
]

CSV_FIELDS = [
    "param_file",
    "scenario_name",
    "variant",
    "variant_description",
    "pass",
    "crash_reason",
    "moving_mass_enabled",
    *METADATA_FIELDS,
    *METRIC_FIELDS,
    *DERIVED_FIELDS,
    "notes",
]


@dataclass(frozen=True)
class MovingMassVariant:
    name: str
    description: str
    moving_mass_enabled: bool
    moving_mass_target_m: float = 0.0
    moving_mass_assist_gain_m_per_Nm: float = 0.0
    use_total_com_geometry: bool = False
    use_legacy_gravity_offset_moment: bool = True
    mode_baseline_variant: str = "vane_only"

    @property
    def moving_mass_model_mode(self) -> str:
        if self.use_total_com_geometry:
            return "total_com_geometry"
        if self.moving_mass_enabled and self.use_legacy_gravity_offset_moment:
            return "legacy_gravity_offset"
        return "disabled"


@dataclass(frozen=True)
class MovingMassComparisonResult:
    variant: MovingMassVariant
    scenario: LoiterScenarioConfig
    run: LoiterRunResult
    row: dict[str, float | int | str | bool]


def default_variants(
    *,
    fixed_target_m: float = 0.02,
    proportional_gain_m_per_Nm: float = 0.025,
) -> list[MovingMassVariant]:
    return [
        MovingMassVariant(
            name="vane_only",
            description="Moving mass disabled; existing vane-only baseline.",
            moving_mass_enabled=False,
        ),
        MovingMassVariant(
            name="moving_mass_fixed_target",
            description="Moving mass enabled with a fixed target offset.",
            moving_mass_enabled=True,
            moving_mass_target_m=fixed_target_m,
        ),
        MovingMassVariant(
            name="moving_mass_proportional_assist",
            description="Moving mass enabled; target is proportional to desired pitch moment.",
            moving_mass_enabled=True,
            moving_mass_assist_gain_m_per_Nm=proportional_gain_m_per_Nm,
        ),
        MovingMassVariant(
            name="total_com_geometry_centered",
            description="Total-COM geometry enabled with the moving mass fixed at zero lateral offset.",
            moving_mass_enabled=False,
            use_total_com_geometry=True,
            use_legacy_gravity_offset_moment=False,
        ),
        MovingMassVariant(
            name="total_com_geometry_fixed_target",
            description="Total-COM geometry with the same fixed target command as the legacy variant.",
            moving_mass_enabled=True,
            moving_mass_target_m=fixed_target_m,
            use_total_com_geometry=True,
            use_legacy_gravity_offset_moment=False,
            mode_baseline_variant="total_com_geometry_centered",
        ),
        MovingMassVariant(
            name="total_com_geometry_proportional_assist",
            description="Total-COM geometry with the same proportional command as the legacy variant.",
            moving_mass_enabled=True,
            moving_mass_assist_gain_m_per_Nm=proportional_gain_m_per_Nm,
            use_total_com_geometry=True,
            use_legacy_gravity_offset_moment=False,
            mode_baseline_variant="total_com_geometry_centered",
        ),
    ]


def resolve_comparison_scenarios(name: str | None, duration_s: float | None = None) -> list[LoiterScenarioConfig]:
    if name is None:
        names = ["pitch_assist_probe"]
    elif name == "all":
        names = DEFAULT_SCENARIOS
    else:
        names = [name]
    return [scenario_by_name(scenario_name, duration_s) for scenario_name in names]


def scenario_for_variant(scenario: LoiterScenarioConfig, variant: MovingMassVariant) -> LoiterScenarioConfig:
    return replace(
        scenario,
        moving_mass_enabled=variant.moving_mass_enabled,
        moving_mass_target_m=variant.moving_mass_target_m,
        moving_mass_assist_gain_m_per_Nm=variant.moving_mass_assist_gain_m_per_Nm,
    )


def _row_for_result(
    param_file: str,
    scenario: LoiterScenarioConfig,
    variant: MovingMassVariant,
    result: LoiterRunResult,
) -> dict[str, float | int | str | bool]:
    metrics = result.metrics
    effective_flags = (
        bool(metrics.get("moving_mass_enabled", False)),
        bool(metrics.get("total_com_geometry_active", False)),
        bool(metrics.get("use_legacy_gravity_offset_moment", False)),
    )
    requested_flags = (
        variant.moving_mass_enabled,
        variant.use_total_com_geometry,
        variant.use_legacy_gravity_offset_moment,
    )
    if effective_flags != requested_flags:
        raise ValueError(
            f"variant {variant.name!r} requested moving-mass flags {requested_flags!r} "
            f"but the effective run used {effective_flags!r}"
        )
    effective_legacy_active = effective_flags[0] and effective_flags[2]
    if bool(metrics.get("legacy_gravity_offset_active", False)) != effective_legacy_active:
        raise ValueError(
            f"variant {variant.name!r} reported inconsistent effective legacy-moment activity"
        )
    expected_state_dimension = 11 if effective_flags[0] else 8
    if int(metrics.get("state_dimension", 0)) != expected_state_dimension:
        raise ValueError(
            f"variant {variant.name!r} expected a {expected_state_dimension}-state run "
            f"but reported {metrics.get('state_dimension')!r}"
        )
    effective_mode = (
        "total_com_geometry"
        if effective_flags[1]
        else "legacy_gravity_offset"
        if effective_flags[0] and effective_flags[2]
        else "disabled"
    )
    if effective_mode != variant.moving_mass_model_mode:
        raise ValueError(
            f"variant {variant.name!r} requested model mode "
            f"{variant.moving_mass_model_mode!r} but the effective mode is {effective_mode!r}"
        )
    row: dict[str, float | int | str | bool] = {
        "param_file": param_file,
        "scenario_name": scenario.name,
        "variant": variant.name,
        "variant_description": variant.description,
        "pass": bool(metrics.get("pass", False)),
        "crash_reason": str(metrics.get("crash_reason", "")),
        "moving_mass_enabled": bool(metrics.get("moving_mass_enabled", False)),
        "moving_mass_model_mode": effective_mode,
        "baseline_variant": "vane_only",
        "mode_baseline_variant": variant.mode_baseline_variant,
        "total_com_geometry_active": bool(metrics.get("total_com_geometry_active", False)),
        "legacy_gravity_offset_active": effective_legacy_active,
        "state_dimension": int(metrics.get("state_dimension", 0)),
        "total_mass_kg": float(metrics.get("total_mass_kg", 0.0)),
        "moving_mass_mass_kg": float(metrics.get("effective_moving_mass_kg", 0.0)),
        "moving_mass_body_up_offset_m": float(
            metrics.get("effective_moving_mass_body_up_offset_m", 0.0)
        ),
        "notes": str(metrics.get("notes", "")),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field, 0.0)
    for field in DERIVED_FIELDS:
        row[field] = 0.0
    return row


def add_baseline_deltas(rows: list[dict[str, float | int | str | bool]]) -> None:
    indexed: dict[tuple[str, str, str], dict[str, float | int | str | bool]] = {}
    for row in rows:
        key = (str(row["param_file"]), str(row["scenario_name"]), str(row["variant"]))
        if key in indexed:
            raise ValueError(
                "duplicate moving-mass comparison row for "
                f"param_file={key[0]!r}, scenario={key[1]!r}, variant={key[2]!r}"
            )
        indexed[key] = row

    delta_metrics = {
        "max_theta_deg": "delta_max_theta_deg",
        "rms_theta_deg": "delta_rms_theta_deg",
        "final_abs_x_error": "delta_final_abs_x_error",
        "rms_x_error": "delta_rms_x_error",
    }
    mode_delta_metrics = {
        "max_theta_deg": "delta_vs_mode_baseline_max_theta_deg",
        "rms_theta_deg": "delta_vs_mode_baseline_rms_theta_deg",
        "final_abs_x_error": "delta_vs_mode_baseline_final_abs_x_error",
        "rms_x_error": "delta_vs_mode_baseline_rms_x_error",
    }
    for row in rows:
        group = (str(row["param_file"]), str(row["scenario_name"]))
        historical_name = str(row["baseline_variant"])
        mode_name = str(row["mode_baseline_variant"])
        baseline = indexed.get((*group, historical_name))
        if baseline is None:
            raise ValueError(
                f"missing baseline variant {historical_name!r} for "
                f"param_file={group[0]!r}, scenario={group[1]!r}"
            )
        mode_baseline = indexed.get((*group, mode_name))
        if mode_baseline is None:
            raise ValueError(
                f"missing mode baseline variant {mode_name!r} for "
                f"param_file={group[0]!r}, scenario={group[1]!r}"
            )
        for metric, delta_field in delta_metrics.items():
            row[delta_field] = float(row[metric]) - float(baseline[metric])
        for metric, delta_field in mode_delta_metrics.items():
            row[delta_field] = float(row[metric]) - float(mode_baseline[metric])
        row["attitude_improvement_score"] = -0.5 * (
            float(row["delta_max_theta_deg"]) + float(row["delta_rms_theta_deg"])
        )


def run_moving_mass_comparison(
    *,
    param_path: str = "params/loiter_example.json",
    scenarios: Iterable[LoiterScenarioConfig],
    variants: Iterable[MovingMassVariant] | None = None,
) -> list[MovingMassComparisonResult]:
    selected_variants = list(variants) if variants is not None else default_variants()
    results: list[MovingMassComparisonResult] = []
    rows: list[dict[str, float | int | str | bool]] = []
    for scenario in scenarios:
        for variant in selected_variants:
            variant_scenario = scenario_for_variant(scenario, variant)
            run = run_headless_loiter(
                param_path,
                variant_scenario,
                rb_overrides={
                    "moving_mass": {
                        "enabled": variant.moving_mass_enabled,
                        "use_total_com_geometry": variant.use_total_com_geometry,
                        "use_legacy_gravity_offset_moment": variant.use_legacy_gravity_offset_moment,
                    }
                },
            )
            row = _row_for_result(param_path, scenario, variant, run)
            rows.append(row)
            results.append(MovingMassComparisonResult(variant, scenario, run, row))
    add_baseline_deltas(rows)
    return results


def write_csv(results: list[MovingMassComparisonResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.row)
    return path


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _best_attitude_rows(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | str | bool]]:
    non_baseline = [row for row in rows if row["variant"] != "vane_only"]
    return sorted(non_baseline, key=lambda row: float(row["attitude_improvement_score"]), reverse=True)


def _worsened_rows(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | str | bool]]:
    return [
        row
        for row in rows
        if row["variant"] != "vane_only"
        and (
            float(row["delta_max_theta_deg"]) > 0.0
            or float(row["delta_rms_theta_deg"]) > 0.0
            or float(row["delta_final_abs_x_error"]) > 0.0
            or float(row["delta_rms_x_error"]) > 0.0
        )
    ]


def write_markdown(results: list[MovingMassComparisonResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.row for result in results]
    variants = {result.variant.name: result.variant.description for result in results}
    lines = [
        "# Moving Mass Comparison Analysis",
        "",
        "## Purpose",
        "",
        "This report compares the legacy gravity-offset and total-COM geometry formulations in the same headless scenarios.",
        "It is an analytical measurement workflow, not a physics change or calibrated flight prediction.",
        "",
        "## Baseline Definitions",
        "",
        "- `baseline_variant` is `vane_only`; the existing `delta_*` fields remain differences versus the historical simulator baseline.",
        "- `mode_baseline_variant` selects the formulation-matched reference used by `delta_vs_mode_baseline_*` fields.",
        "- Geometry active variants use `total_com_geometry_centered` as their mode baseline, isolating the additional effect of lateral mass motion.",
        "- Negative error or angle deltas mean the raw metric decreased. Positive `attitude_improvement_score` means the average of max and RMS pitch angle decreased versus `vane_only`.",
        "",
        "## Variant Definitions",
        "",
    ]
    variant_groups = [
        ("Historical baseline", ["vane_only"]),
        ("Legacy gravity-offset variants", ["moving_mass_fixed_target", "moving_mass_proportional_assist"]),
        ("Total-COM centered baseline", ["total_com_geometry_centered"]),
        (
            "Total-COM active variants",
            ["total_com_geometry_fixed_target", "total_com_geometry_proportional_assist"],
        ),
    ]
    for heading, names in variant_groups:
        present = [name for name in names if name in variants]
        if not present:
            continue
        lines += [f"### {heading}", ""]
        for name in present:
            lines.append(f"- `{name}`: {variants[name]}")

    effective_by_param: dict[str, dict[str, float | int | str | bool]] = {}
    for row in rows:
        effective_by_param.setdefault(str(row["param_file"]), row)
    lines += [
        "",
        "## Effective Mass And Geometry Configuration",
        "",
        "| parameter source | total_mass_kg | moving_mass_mass_kg | configured_max_offset_m | moving_mass_body_up_offset_m |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for param_file, row in effective_by_param.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(param_file),
                    _fmt(row["total_mass_kg"]),
                    _fmt(row["moving_mass_mass_kg"]),
                    _fmt(row["effective_moving_mass_max_offset_m"]),
                    _fmt(row["moving_mass_body_up_offset_m"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Scenario Summary",
        "",
        "| scenario | variant | model mode | mode baseline | max_theta_deg | rms_theta_deg | final_abs_x_error | rms_x_error | mm_max_offset_m | notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["scenario_name"]),
                    _md_cell(row["variant"]),
                    _md_cell(row["moving_mass_model_mode"]),
                    _md_cell(row["mode_baseline_variant"]),
                    _fmt(row["max_theta_deg"]),
                    _fmt(row["rms_theta_deg"]),
                    _fmt(row["final_abs_x_error"]),
                    _fmt(row["rms_x_error"]),
                    _fmt(row["moving_mass_max_offset_m"]),
                    _md_cell(row["notes"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Deltas Versus Mode-Matched Baseline",
        "",
        "| scenario | variant | mode baseline | delta_max_theta_deg | delta_rms_theta_deg | delta_final_abs_x_error | delta_rms_x_error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["variant"] == row["mode_baseline_variant"]:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["scenario_name"]),
                    _md_cell(row["variant"]),
                    _md_cell(row["mode_baseline_variant"]),
                    _fmt(row["delta_vs_mode_baseline_max_theta_deg"]),
                    _fmt(row["delta_vs_mode_baseline_rms_theta_deg"]),
                    _fmt(row["delta_vs_mode_baseline_final_abs_x_error"]),
                    _fmt(row["delta_vs_mode_baseline_rms_x_error"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Geometry Diagnostics",
        "",
        "`vane_moment_about_total_com` is the complete vane moment about the selected COM; it is not attributed solely to moving-mass motion.",
        "COM metrics use metres; thrust, vane, and legacy moment metrics use N·m. RMS uses every logged post-step sample.",
        "",
        "| scenario | variant | max_abs_com_right_m | max_abs_com_up_m | max_abs_thrust_com_moment | rms_thrust_com_moment | max_abs_vane_com_moment | max_abs_legacy_moment |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["scenario_name"]),
                    _md_cell(row["variant"]),
                    _fmt(row["max_abs_total_com_body_right_m"]),
                    _fmt(row["max_abs_total_com_body_up_m"]),
                    _fmt(row["max_abs_thrust_moment_from_com_offset"]),
                    _fmt(row["rms_thrust_moment_from_com_offset"]),
                    _fmt(row["max_abs_vane_moment_about_total_com"]),
                    _fmt(row["max_abs_legacy_moving_mass_moment"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Best Attitude Reduction Versus Vane-Only",
        "",
        "| scenario | variant | delta_max_theta_deg | delta_rms_theta_deg | attitude_score | delta_final_abs_x_error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    best_rows = [row for row in _best_attitude_rows(rows) if float(row["attitude_improvement_score"]) > 0.0]
    if best_rows:
        for row in best_rows[:10]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(row["scenario_name"]),
                        _md_cell(row["variant"]),
                        _fmt(row["delta_max_theta_deg"]),
                        _fmt(row["delta_rms_theta_deg"]),
                        _fmt(row["attitude_improvement_score"]),
                        _fmt(row["delta_final_abs_x_error"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| no improvement under this configuration | n/a | 0.000 | 0.000 | 0.000 | 0.000 |")
    lines += [
        "",
        "## Cases That Worsen Versus Vane-Only",
        "",
        "| scenario | variant | delta_max_theta_deg | delta_rms_theta_deg | delta_final_abs_x_error | delta_rms_x_error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    worsened = _worsened_rows(rows)
    if worsened:
        for row in worsened:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(row["scenario_name"]),
                        _md_cell(row["variant"]),
                        _fmt(row["delta_max_theta_deg"]),
                        _fmt(row["delta_rms_theta_deg"]),
                        _fmt(row["delta_final_abs_x_error"]),
                        _fmt(row["delta_rms_x_error"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| no worsened cases in this run | n/a | 0.000 | 0.000 | 0.000 | 0.000 |")
    lines += [
        "",
        "## Notes And Limitations",
        "",
        "- Under this analytical scenario, deltas describe numerical differences only; they do not establish real-flight accuracy or global benefit.",
        "- Geometry variants should be judged both versus `vane_only` and relative to the geometry-centered baseline.",
        "- The geometry formulation changes thrust and vane arms about the instantaneous total COM and does not apply the legacy `m_m * g * offset` term.",
        "- A run can reduce RMS theta but increase final x error; inspect attitude and position metrics together.",
        "- This does not include reaction kick, moving-mass acceleration reaction, internal momentum coupling, or position-dependent inertia (`Iyy`).",
        "- The comparison remains 2D and does not validate a 3D aircraft.",
        "- Effective parameters are loaded assumptions, not calibrated flight values.",
        "- No flip controller, reinforcement learning, four-vane physics wiring, or real-flight calibration is included.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_plots(results: list[MovingMassComparisonResult], output_dir: str | Path, required: bool = False) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if required:
            raise RuntimeError("matplotlib is not installed; CSV and Markdown were still generated.")
        print("matplotlib is not installed; CSV and Markdown were still generated.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result.row for result in results]
    paths: list[Path] = []
    for metric in ("max_theta_deg", "rms_theta_deg", "final_abs_x_error"):
        labels = [f"{row['scenario_name']}\n{row['variant']}" for row in rows]
        values = [float(row[metric]) for row in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.5))
        ax.bar(range(len(rows)), values)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel(metric)
        fig.tight_layout()
        path = output_dir / f"{metric}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths
