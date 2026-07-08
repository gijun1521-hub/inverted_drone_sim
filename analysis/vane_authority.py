from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .headless_loiter import LoiterScenarioConfig, run_headless_loiter
except ImportError:  # pragma: no cover - supports top-level script execution
    from analysis.headless_loiter import LoiterScenarioConfig, run_headless_loiter


DEFAULT_VANE_ANGLES = [0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25]
DEFAULT_VANE_RATES = [5, 10, 20, 40, 80, 160, 300]
DEFAULT_T_MAX_FACTORS = [1.05, 1.1, 1.2, 1.4, 1.8, 2.2, 2.5]
QUICK_VANE_ANGLES = [0.5, 5, 20]
QUICK_VANE_RATES = [5, 80]
QUICK_T_MAX_FACTORS = [1.05, 2.5]


SWEEP_FIELDS = [
    "scenario_name",
    "param_file",
    "vane_angle_max_deg",
    "vane_rate_limit_deg_s",
    "T_max_factor",
    "effective_vane_angle_max_deg",
    "effective_vane_rate_limit_deg_s",
    "effective_T_max_factor",
    "effective_T_max_N",
    "pass",
    "failure_reason",
    "crash_reason",
    "final_abs_x_error",
    "final_abs_z_error",
    "rms_x_error",
    "rms_z_error",
    "max_theta_deg",
    "max_omega_deg_s",
    "max_vane_cmd_deg",
    "max_vane_actual_deg",
    "max_thrust_cmd_N",
    "max_thrust_actual_N",
    "mixer_saturation_percent",
    "mixer_angle_saturation_percent",
    "authority_limited_percent",
    "servo_angle_saturation_percent",
    "servo_rate_saturation_percent",
    "motor_saturation_percent",
    "final_x",
    "final_z",
    "final_vx",
    "final_vz",
    "authority_margin_score",
    "recovery_score",
    "saturation_score",
    "combined_design_score",
]


SENSITIVITY_METRICS = [
    "final_abs_x_error",
    "max_vane_cmd_deg",
    "max_vane_actual_deg",
    "mixer_saturation_percent",
    "authority_limited_percent",
    "servo_rate_saturation_percent",
    "motor_saturation_percent",
    "combined_design_score",
]


@dataclass(frozen=True)
class VaneAuthorityGrid:
    vane_angle_max_deg: list[float]
    vane_rate_limit_deg_s: list[float]
    T_max_factor: list[float]


def parse_float_list(values: str | list[float]) -> list[float]:
    if isinstance(values, list):
        return [float(v) for v in values]
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def default_grid(quick: bool = False) -> VaneAuthorityGrid:
    if quick:
        return VaneAuthorityGrid(QUICK_VANE_ANGLES, QUICK_VANE_RATES, QUICK_T_MAX_FACTORS)
    return VaneAuthorityGrid(DEFAULT_VANE_ANGLES, DEFAULT_VANE_RATES, DEFAULT_T_MAX_FACTORS)


def authority_stress_scenarios(duration_s: float | None = None) -> dict[str, LoiterScenarioConfig]:
    scenarios = {
        "authority_stress": LoiterScenarioConfig(
            name="authority_stress",
            duration_s=7.0,
            initial_x=1.6,
            initial_z=0.85,
            target_x=0.0,
            target_z=1.2,
            disturbance_start_s=0.8,
            disturbance_duration_s=0.45,
            disturbance_force_x=12.0,
            max_final_x_error=2.1,
            max_final_z_error=0.35,
            max_rms_x_error=3.0,
            max_rms_z_error=1.2,
            max_theta_deg_limit=80.0,
            max_saturation_percent=100.0,
            notes="Low altitude margin, initial x offset, and horizontal impulse.",
        ),
        "impulse_light": LoiterScenarioConfig(
            name="impulse_light",
            duration_s=6.0,
            disturbance_start_s=0.8,
            disturbance_duration_s=0.20,
            disturbance_force_x=6.0,
            max_final_x_error=1.4,
            max_final_z_error=0.35,
            max_rms_x_error=1.6,
            max_rms_z_error=0.9,
            max_theta_deg_limit=60.0,
            max_saturation_percent=100.0,
            notes="Moderate horizontal impulse; many reasonable configurations should recover.",
        ),
        "impulse_heavy": LoiterScenarioConfig(
            name="impulse_heavy",
            duration_s=7.0,
            disturbance_start_s=0.8,
            disturbance_duration_s=0.55,
            disturbance_force_x=16.0,
            max_final_x_error=2.4,
            max_final_z_error=0.45,
            max_rms_x_error=3.2,
            max_rms_z_error=1.2,
            max_theta_deg_limit=85.0,
            max_saturation_percent=100.0,
            notes="Strong horizontal impulse; low-authority configurations should fail or saturate.",
        ),
        "offset_recovery_2m": LoiterScenarioConfig(
            name="offset_recovery_2m",
            duration_s=8.0,
            initial_x=2.0,
            target_x=0.0,
            max_final_x_error=1.2,
            max_final_z_error=0.35,
            max_rms_x_error=2.1,
            max_rms_z_error=0.9,
            max_theta_deg_limit=70.0,
            max_saturation_percent=100.0,
            notes="Starts 2 m away from target_x with no pilot stick input.",
        ),
        "stick_step_aggressive": LoiterScenarioConfig(
            name="stick_step_aggressive",
            duration_s=8.0,
            stick_start_s=0.4,
            stick_end_s=1.8,
            stick_x=0.95,
            capture_current_target=True,
            max_final_x_error=1.6,
            max_final_z_error=0.35,
            max_rms_x_error=2.3,
            max_rms_z_error=0.9,
            max_theta_deg_limit=80.0,
            max_saturation_percent=100.0,
            notes="Near-full horizontal stick step, release, then settle.",
        ),
        "low_thrust_margin": LoiterScenarioConfig(
            name="low_thrust_margin",
            duration_s=7.0,
            initial_x=1.2,
            initial_z=0.82,
            target_x=0.0,
            target_z=1.25,
            disturbance_start_s=0.8,
            disturbance_duration_s=0.35,
            disturbance_force_x=8.0,
            max_final_x_error=1.9,
            max_final_z_error=0.55,
            max_rms_x_error=2.5,
            max_rms_z_error=1.3,
            max_theta_deg_limit=80.0,
            max_saturation_percent=100.0,
            notes="Low vertical margin and climb demand to expose thrust margin coupling.",
        ),
    }
    if duration_s is not None:
        scenarios = {name: replace_scenario_duration(scenario, duration_s) for name, scenario in scenarios.items()}
    return scenarios


def replace_scenario_duration(scenario: LoiterScenarioConfig, duration_s: float) -> LoiterScenarioConfig:
    from dataclasses import replace

    return replace(scenario, duration_s=duration_s)


def resolve_authority_scenarios(name: str, duration_s: float | None = None) -> list[LoiterScenarioConfig]:
    scenarios = authority_stress_scenarios(duration_s)
    if name == "all":
        return list(scenarios.values())
    if name not in scenarios:
        names = ", ".join([*scenarios.keys(), "all"])
        raise ValueError(f"unknown authority scenario {name!r}; expected one of: {names}")
    return [scenarios[name]]


def failure_reason(metrics: dict[str, Any]) -> str:
    if metrics["pass"]:
        return ""
    if metrics["crash_reason"]:
        return "crash"
    limit_peak = max(
        float(metrics["mixer_saturation_percent"]),
        float(metrics["authority_limited_percent"]),
        float(metrics["servo_rate_saturation_percent"]),
        float(metrics["motor_saturation_percent"]),
    )
    if limit_peak > 0.0:
        return "saturation_or_authority_limit"
    return "large_final_error"


def _bounded_score(value: float, good_at_or_below: float) -> float:
    return 1.0 / (1.0 + max(0.0, value) / max(good_at_or_below, 1e-9))


def design_scores(metrics: dict[str, Any], scenario: LoiterScenarioConfig) -> dict[str, float]:
    recovery = 0.45 * _bounded_score(float(metrics["final_abs_x_error"]), scenario.max_final_x_error)
    recovery += 0.25 * _bounded_score(float(metrics["rms_x_error"]), scenario.max_rms_x_error)
    recovery += 0.20 * _bounded_score(float(metrics["final_abs_z_error"]), scenario.max_final_z_error)
    recovery += 0.10 * _bounded_score(float(metrics["max_theta_deg"]), scenario.max_theta_deg_limit)
    limit_activity = max(
        float(metrics["mixer_saturation_percent"]),
        float(metrics["authority_limited_percent"]),
        float(metrics["servo_angle_saturation_percent"]),
        float(metrics["servo_rate_saturation_percent"]),
        float(metrics["motor_saturation_percent"]),
    )
    saturation = max(0.0, 1.0 - limit_activity / 100.0)
    margin = max(
        0.0,
        1.0
        - 0.45 * float(metrics["authority_limited_percent"]) / 100.0
        - 0.25 * float(metrics["mixer_saturation_percent"]) / 100.0
        - 0.20 * float(metrics["servo_rate_saturation_percent"]) / 100.0
        - 0.10 * float(metrics["motor_saturation_percent"]) / 100.0,
    )
    pass_bonus = 0.15 if metrics["pass"] else 0.0
    combined = min(1.0, 0.45 * recovery + 0.25 * saturation + 0.15 * margin + pass_bonus)
    return {
        "authority_margin_score": float(margin),
        "recovery_score": float(recovery),
        "saturation_score": float(saturation),
        "combined_design_score": float(combined),
    }


def row_from_result(result, angle: float, rate: float, thrust_factor: float) -> dict[str, float | str | bool]:
    m = result.metrics
    row = {
        "scenario_name": result.scenario.name,
        "param_file": result.param_file,
        "vane_angle_max_deg": angle,
        "vane_rate_limit_deg_s": rate,
        "T_max_factor": thrust_factor,
        "effective_vane_angle_max_deg": float(m["effective_vane_angle_max_deg"]),
        "effective_vane_rate_limit_deg_s": float(m["effective_vane_rate_limit_deg_s"]),
        "effective_T_max_factor": float(m["effective_T_max_factor"]),
        "effective_T_max_N": float(m["effective_T_max_N"]),
        "pass": bool(m["pass"]),
        "failure_reason": failure_reason(m),
        "crash_reason": str(m["crash_reason"]),
        "final_abs_x_error": float(m["final_abs_x_error"]),
        "final_abs_z_error": float(m["final_abs_z_error"]),
        "rms_x_error": float(m["rms_x_error"]),
        "rms_z_error": float(m["rms_z_error"]),
        "max_theta_deg": float(m["max_theta_deg"]),
        "max_omega_deg_s": float(m["max_omega_deg_s"]),
        "max_vane_cmd_deg": float(m["max_vane_cmd_deg"]),
        "max_vane_actual_deg": float(m["max_vane_actual_deg"]),
        "max_thrust_cmd_N": float(m["max_thrust_cmd_N"]),
        "max_thrust_actual_N": float(m["max_thrust_actual_N"]),
        "mixer_saturation_percent": float(m["mixer_saturation_percent"]),
        "mixer_angle_saturation_percent": float(m["mixer_angle_saturation_percent"]),
        "authority_limited_percent": float(m["authority_limited_percent"]),
        "servo_angle_saturation_percent": float(m["servo_angle_saturation_percent"]),
        "servo_rate_saturation_percent": float(m["servo_rate_saturation_percent"]),
        "motor_saturation_percent": float(m["motor_saturation_percent"]),
        "final_x": float(m["final_x"]),
        "final_z": float(m["final_z"]),
        "final_vx": float(m["final_vx"]),
        "final_vz": float(m["final_vz"]),
    }
    row.update(design_scores(m, result.scenario))
    return row


def run_authority_sweep(
    *,
    param_path: str = "params/loiter_example.json",
    scenarios: list[LoiterScenarioConfig],
    grid: VaneAuthorityGrid,
) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for scenario in scenarios:
        for angle in grid.vane_angle_max_deg:
            for rate in grid.vane_rate_limit_deg_s:
                for thrust_factor in grid.T_max_factor:
                    result = run_headless_loiter(
                        param_path,
                        scenario,
                        rb_overrides={
                            "vane_angle_max_deg": angle,
                            "vane_rate_limit_deg_s": rate,
                            "T_max_factor": thrust_factor,
                        },
                    )
                    rows.append(row_from_result(result, angle, rate, thrust_factor))
    return rows


def sweep_sensitivity(rows: list[dict[str, float | str | bool]]) -> dict[str, float | int | bool]:
    stats: dict[str, float | int | bool] = {}
    for metric in SENSITIVITY_METRICS:
        values = [round(float(row[metric]), 6) for row in rows]
        stats[f"unique_{metric}"] = len(set(values))
        stats[f"min_{metric}"] = min(values) if values else 0.0
        stats[f"max_{metric}"] = max(values) if values else 0.0
    varied = any(int(stats[f"unique_{metric}"]) > 1 for metric in SENSITIVITY_METRICS[:3])
    limit_triggered = any(float(stats[f"max_{metric}"]) > 0.0 for metric in SENSITIVITY_METRICS[3:7])
    stats["inconclusive"] = bool(rows and not varied and not limit_triggered)
    return stats


def write_csv(rows: list[dict[str, float | str | bool]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _scenario_rows(rows: list[dict[str, float | str | bool]], scenario: str) -> list[dict[str, float | str | bool]]:
    return [row for row in rows if row["scenario_name"] == scenario]


def recommended_regions(rows: list[dict[str, float | str | bool]]) -> list[str]:
    lines: list[str] = []
    scenarios = sorted({str(row["scenario_name"]) for row in rows})
    for scenario in scenarios:
        subset = _scenario_rows(rows, scenario)
        if not subset:
            continue
        best_error = min(subset, key=lambda r: float(r["final_abs_x_error"]))
        best_score = max(subset, key=lambda r: float(r["combined_design_score"]))
        passing = [row for row in subset if row["pass"]]
        lines += [
            f"### {scenario}",
            "",
            f"- Best final horizontal error: {float(best_error['final_abs_x_error']):.3f} m at "
            f"{best_error['vane_angle_max_deg']} deg, {best_error['vane_rate_limit_deg_s']} deg/s, T_max_factor {best_error['T_max_factor']}.",
            f"- Best combined design score: {float(best_score['combined_design_score']):.3f} at "
            f"{best_score['vane_angle_max_deg']} deg, {best_score['vane_rate_limit_deg_s']} deg/s, T_max_factor {best_score['T_max_factor']}.",
        ]
        if passing:
            lines.append(f"- Lowest passing vane angle in this analytical grid: {min(float(r['vane_angle_max_deg']) for r in passing):g} deg.")
            lines.append(f"- Lowest passing servo rate in this analytical grid: {min(float(r['vane_rate_limit_deg_s']) for r in passing):g} deg/s.")
            lines.append(f"- Lowest passing T_max_factor in this analytical grid: {min(float(r['T_max_factor']) for r in passing):g}.")
        else:
            lines.append("- No passing cases in this grid; treat the scenario/grid combination as a stress signal, not a design requirement.")
        high_limit = [row for row in subset if max(float(row["authority_limited_percent"]), float(row["mixer_saturation_percent"])) >= 10.0]
        high_rate = [row for row in subset if float(row["servo_rate_saturation_percent"]) >= 10.0]
        bad = [row for row in subset if float(row["combined_design_score"]) < 0.35 or row["crash_reason"]]
        if high_limit:
            max_angle = max(float(row["vane_angle_max_deg"]) for row in high_limit)
            lines.append(f"- Authority or mixer limits are common in the low-authority region up to roughly {max_angle:g} deg in this sweep.")
        else:
            lines.append("- No strong mixer/authority-limited region appeared in this sweep.")
        if high_rate:
            max_rate = max(float(row["vane_rate_limit_deg_s"]) for row in high_rate)
            lines.append(f"- Servo rate saturation appears for slow rates up to roughly {max_rate:g} deg/s in this sweep.")
        else:
            lines.append("- Servo rate saturation was not a dominant limiter in this sweep.")
        if bad:
            lines.append(f"- Consistently poor cases cluster around the lowest vane angles, slowest rates, or low thrust margins; {len(bad)} cases scored below 0.35 or crashed.")
        lines.append("")
    return lines


def write_markdown(rows: list[dict[str, float | str | bool]], path: Path, scenario_label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for row in rows if row["pass"])
    total = len(rows)
    stats = sweep_sensitivity(rows)
    failure_counts: dict[str, int] = {}
    for row in rows:
        if not row["pass"]:
            reason = str(row["failure_reason"])
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    lines = [
        "# Vane Authority Mapping",
        "",
        f"Scenario selection: `{scenario_label}`",
        "",
        "This is a design-oriented analytical map for the simplified 2D single-fan simulator. It is not calibrated real-flight prediction.",
        "Use the results as relative indicators for recovery, saturation, authority margin, and tuning sensitivity.",
        "",
        f"Summary: {passed}/{total} analytical checks passed.",
        "",
    ]
    if stats.get("inconclusive"):
        lines += ["**INCONCLUSIVE:** sweep parameters did not affect important metrics or trigger limits.", ""]
    lines += [
        "## Sensitivity Check",
        "",
        f"- unique final_abs_x_error values: {stats.get('unique_final_abs_x_error', 0)}",
        f"- final_abs_x_error range: {float(stats.get('min_final_abs_x_error', 0.0)):.3f} to {float(stats.get('max_final_abs_x_error', 0.0)):.3f} m",
        f"- max_vane_cmd_deg range: {float(stats.get('min_max_vane_cmd_deg', 0.0)):.3f} to {float(stats.get('max_max_vane_cmd_deg', 0.0)):.3f} deg",
        f"- max_vane_actual_deg range: {float(stats.get('min_max_vane_actual_deg', 0.0)):.3f} to {float(stats.get('max_max_vane_actual_deg', 0.0)):.3f} deg",
        f"- combined_design_score range: {float(stats.get('min_combined_design_score', 0.0)):.3f} to {float(stats.get('max_combined_design_score', 0.0)):.3f}",
        f"- max mixer_saturation_percent: {float(stats.get('max_mixer_saturation_percent', 0.0)):.1f}%",
        f"- max authority_limited_percent: {float(stats.get('max_authority_limited_percent', 0.0)):.1f}%",
        f"- max servo_rate_saturation_percent: {float(stats.get('max_servo_rate_saturation_percent', 0.0)):.1f}%",
        f"- max motor_saturation_percent: {float(stats.get('max_motor_saturation_percent', 0.0)):.1f}%",
        "",
        "## Failure Reasons",
        "",
    ]
    if failure_counts:
        for reason, count in sorted(failure_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Recommended Design Regions",
        "",
        *recommended_regions(rows),
        "## Engineering Takeaway",
        "",
        "- Configurations below the lowest passing vane angle in each scenario are generally under-actuated in this analytical grid.",
        "- Servo rates that repeatedly show rate saturation should be treated as candidates for slower recovery or overshoot risk.",
        "- Increasing thrust margin helps most after sufficient vane angle and servo rate are available.",
        "- These are analytical trends, not calibrated flight requirements.",
        "",
        "## Limitations",
        "",
        "- The simulator is 2D and uses one equivalent pitch-axis vane/moment.",
        "- It is ArduCopter-inspired, not exact ArduPilot firmware.",
        "- It is not calibrated with bench thrust data or real flight logs.",
        "- Real SingleCopter four-flap physics are future work.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _grid(rows, scenario: str, thrust_factor: float, metric: str):
    subset = [r for r in rows if r["scenario_name"] == scenario and float(r["T_max_factor"]) == thrust_factor]
    angles = sorted({float(r["vane_angle_max_deg"]) for r in subset})
    rates = sorted({float(r["vane_rate_limit_deg_s"]) for r in subset})
    data = np.full((len(rates), len(angles)), np.nan)
    for row in subset:
        data[rates.index(float(row["vane_rate_limit_deg_s"])), angles.index(float(row["vane_angle_max_deg"]))] = float(row[metric])
    return angles, rates, data


def write_plots(rows: list[dict[str, float | str | bool]], output_dir: Path, required: bool = False) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        message = "matplotlib is not installed; CSV and Markdown were still generated."
        if required:
            raise RuntimeError(message)
        print(message)
        return []

    plot_dir = output_dir / "authority_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    scenarios = sorted({str(r["scenario_name"]) for r in rows})
    thrust_factors = sorted({float(r["T_max_factor"]) for r in rows})
    metrics = [
        ("final_abs_x_error", "Final abs x error", "m"),
        ("pass", "Pass/fail", "0 fail / 1 pass"),
        ("authority_limited_percent", "Authority limited", "%"),
        ("servo_rate_saturation_percent", "Servo rate saturation", "%"),
        ("combined_design_score", "Combined design score", "score"),
    ]
    for scenario in scenarios:
        for thrust_factor in thrust_factors:
            for metric, title, label in metrics:
                angles, rates, data = _grid(rows, scenario, thrust_factor, metric)
                if data.size == 0:
                    continue
                fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
                image = ax.imshow(data, origin="lower", aspect="auto")
                ax.set_title(f"{scenario}: {title}, T_max_factor={thrust_factor:g}")
                ax.set_xticks(range(len(angles)))
                ax.set_xticklabels([f"{v:g}" for v in angles])
                ax.set_yticks(range(len(rates)))
                ax.set_yticklabels([f"{v:g}" for v in rates])
                ax.set_xlabel("vane_angle_max_deg")
                ax.set_ylabel("vane_rate_limit_deg_s")
                fig.colorbar(image, ax=ax, label=label)
                path = plot_dir / f"{scenario}_{metric}_Tmax_{str(thrust_factor).replace('.', 'p')}.png"
                fig.savefig(path, dpi=140)
                plt.close(fig)
                paths.append(path)

    for scenario in scenarios:
        subset = _scenario_rows(rows, scenario)
        angles = sorted({float(r["vane_angle_max_deg"]) for r in subset})
        rates = sorted({float(r["vane_rate_limit_deg_s"]) for r in subset})
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for rate in rates[:5]:
            values = []
            for angle in angles:
                candidates = [r for r in subset if float(r["vane_rate_limit_deg_s"]) == rate and float(r["vane_angle_max_deg"]) == angle and r["pass"]]
                values.append(min((float(r["T_max_factor"]) for r in candidates), default=np.nan))
            ax.plot(angles, values, marker="o", label=f"{rate:g} deg/s")
        ax.set_title(f"{scenario}: minimum passing T_max_factor")
        ax.set_xlabel("vane_angle_max_deg")
        ax.set_ylabel("minimum passing T_max_factor")
        ax.legend(fontsize=7)
        path = plot_dir / f"{scenario}_minimum_passing_Tmax_summary.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths
