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
]

DERIVED_FIELDS = [
    "delta_max_theta_deg",
    "delta_rms_theta_deg",
    "delta_final_abs_x_error",
    "delta_rms_x_error",
    "attitude_improvement_score",
]

CSV_FIELDS = [
    "param_file",
    "scenario_name",
    "variant",
    "variant_description",
    "pass",
    "crash_reason",
    "moving_mass_enabled",
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
    row: dict[str, float | int | str | bool] = {
        "param_file": param_file,
        "scenario_name": scenario.name,
        "variant": variant.name,
        "variant_description": variant.description,
        "pass": bool(metrics.get("pass", False)),
        "crash_reason": str(metrics.get("crash_reason", "")),
        "moving_mass_enabled": bool(metrics.get("moving_mass_enabled", False)),
        "notes": str(metrics.get("notes", "")),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field, 0.0)
    for field in DERIVED_FIELDS:
        row[field] = 0.0
    return row


def add_baseline_deltas(rows: list[dict[str, float | int | str | bool]]) -> None:
    baselines = {
        (str(row["param_file"]), str(row["scenario_name"])): row
        for row in rows
        if row["variant"] == "vane_only"
    }
    for row in rows:
        baseline = baselines.get((str(row["param_file"]), str(row["scenario_name"])))
        if baseline is None:
            continue
        row["delta_max_theta_deg"] = float(row["max_theta_deg"]) - float(baseline["max_theta_deg"])
        row["delta_rms_theta_deg"] = float(row["rms_theta_deg"]) - float(baseline["rms_theta_deg"])
        row["delta_final_abs_x_error"] = float(row["final_abs_x_error"]) - float(baseline["final_abs_x_error"])
        row["delta_rms_x_error"] = float(row["rms_x_error"]) - float(baseline["rms_x_error"])
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
            run = run_headless_loiter(param_path, variant_scenario)
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
        "This report compares the existing 2D quasi-static moving-mass pitch assist model against the vane-only baseline.",
        "It is a headless measurement workflow, not a physics change or calibrated flight prediction.",
        "",
        "## Variant Definitions",
        "",
    ]
    for name, description in variants.items():
        lines.append(f"- `{name}`: {description}")
    lines += [
        "",
        "## Scenario Summary",
        "",
        "| scenario | variant | max_theta_deg | rms_theta_deg | final_abs_x_error | rms_x_error | mm_max_offset_m | mm_sat_% | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario_name"]),
                    str(row["variant"]),
                    _fmt(row["max_theta_deg"]),
                    _fmt(row["rms_theta_deg"]),
                    _fmt(row["final_abs_x_error"]),
                    _fmt(row["rms_x_error"]),
                    _fmt(row["moving_mass_max_offset_m"]),
                    _fmt(row["moving_mass_saturation_percent"]),
                    str(row["notes"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Best Attitude Reduction",
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
                        str(row["scenario_name"]),
                        str(row["variant"]),
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
        "## Cases Where Moving Mass Worsens Performance",
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
                        str(row["scenario_name"]),
                        str(row["variant"]),
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
        "- Negative delta values mean the moving-mass variant reduced that metric versus `vane_only`.",
        "- Positive `attitude_improvement_score` means max and RMS pitch angle decreased on average.",
        "- A run can reduce RMS theta but increase final x error; inspect attitude and position metrics together.",
        "- These outputs compare the existing 2D quasi-static CG torque model only.",
        "- The current model does not yet include explicit total-CG geometry shift, moving-mass-induced thrust-line pitch moment, position-dependent inertia (`Iyy`) changes, inertial reaction kick from moving-mass acceleration, or full 3D coupled dynamics.",
        "- Results are provisional and must not be used as final validation for a large moving mass such as approximately 0.5 kg.",
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
