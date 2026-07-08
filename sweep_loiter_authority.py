from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from analysis.headless_loiter import run_headless_loiter, scenario_by_name


SWEEP_FIELDS = [
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
    "max_theta_deg",
    "max_vane_cmd_deg",
    "max_vane_actual_deg",
    "mixer_saturation_percent",
    "authority_limited_percent",
    "servo_rate_saturation_percent",
    "motor_saturation_percent",
]


SENSITIVITY_METRICS = [
    "final_abs_x_error",
    "max_vane_cmd_deg",
    "max_vane_actual_deg",
    "mixer_saturation_percent",
    "authority_limited_percent",
    "servo_rate_saturation_percent",
    "motor_saturation_percent",
]


def _parse_float_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _failure_reason(metrics: dict[str, Any]) -> str:
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


def run_sweep(args: argparse.Namespace) -> list[dict[str, float | str | bool]]:
    scenario = scenario_by_name(args.scenario, args.duration)
    rows: list[dict[str, float | str | bool]] = []
    for angle in _parse_float_list(args.vane_angle_max_deg):
        for rate in _parse_float_list(args.vane_rate_limit_deg_s):
            for thrust_factor in _parse_float_list(args.T_max_factor):
                run_scenario = replace(scenario, vane_angle_max_deg=None, vane_rate_limit_deg_s=None, T_max_factor=None)
                result = run_headless_loiter(
                    args.params,
                    run_scenario,
                    rb_overrides={
                        "vane_angle_max_deg": angle,
                        "vane_rate_limit_deg_s": rate,
                        "T_max_factor": thrust_factor,
                    },
                )
                m = result.metrics
                rows.append(
                    {
                        "vane_angle_max_deg": angle,
                        "vane_rate_limit_deg_s": rate,
                        "T_max_factor": thrust_factor,
                        "effective_vane_angle_max_deg": float(m["effective_vane_angle_max_deg"]),
                        "effective_vane_rate_limit_deg_s": float(m["effective_vane_rate_limit_deg_s"]),
                        "effective_T_max_factor": float(m["effective_T_max_factor"]),
                        "effective_T_max_N": float(m["effective_T_max_N"]),
                        "pass": bool(m["pass"]),
                        "failure_reason": _failure_reason(m),
                        "crash_reason": str(m["crash_reason"]),
                        "final_abs_x_error": float(m["final_abs_x_error"]),
                        "final_abs_z_error": float(m["final_abs_z_error"]),
                        "max_theta_deg": float(m["max_theta_deg"]),
                        "max_vane_cmd_deg": float(m["max_vane_cmd_deg"]),
                        "max_vane_actual_deg": float(m["max_vane_actual_deg"]),
                        "mixer_saturation_percent": float(m["mixer_saturation_percent"]),
                        "authority_limited_percent": float(m["authority_limited_percent"]),
                        "servo_rate_saturation_percent": float(m["servo_rate_saturation_percent"]),
                        "motor_saturation_percent": float(m["motor_saturation_percent"]),
                    }
                )
    return rows


def sweep_sensitivity(rows: list[dict[str, float | str | bool]]) -> dict[str, float | int | bool]:
    stats: dict[str, float | int | bool] = {}
    for metric in SENSITIVITY_METRICS:
        values = [round(float(row[metric]), 6) for row in rows]
        stats[f"unique_{metric}"] = len(set(values))
        stats[f"min_{metric}"] = min(values) if values else 0.0
        stats[f"max_{metric}"] = max(values) if values else 0.0
    varied = any(int(stats[f"unique_{metric}"]) > 1 for metric in SENSITIVITY_METRICS[:3])
    limit_triggered = any(float(stats[f"max_{metric}"]) > 0.0 for metric in SENSITIVITY_METRICS[3:])
    stats["inconclusive"] = bool(rows and not varied and not limit_triggered)
    return stats


def write_csv(rows: list[dict[str, float | str | bool]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown(rows: list[dict[str, float | str | bool]], path: Path, scenario_name: str) -> Path:
    passed = sum(1 for row in rows if row["pass"])
    total = len(rows)
    best = min(rows, key=lambda r: float(r["final_abs_x_error"])) if rows else None
    worst_auth = max(rows, key=lambda r: float(r["authority_limited_percent"])) if rows else None
    stats = sweep_sensitivity(rows)
    inconclusive = bool(stats.get("inconclusive", False))
    failure_counts: dict[str, int] = {}
    for row in rows:
        if not row["pass"]:
            reason = str(row["failure_reason"])
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    lines = [
        "# Preliminary LOITER Authority Sweep",
        "",
        f"Scenario: `{scenario_name}`",
        "",
        "This is a preliminary authority map from the analytical 2D simulator. It is not calibrated real-flight prediction.",
        "Use it to identify obviously under-actuated or saturated regions before expanding the sweep in a later PR.",
        "",
        f"Summary: {passed}/{total} analytical checks passed.",
        "",
    ]
    if inconclusive:
        lines += [
            "**INCONCLUSIVE:** the selected sweep parameters did not change the important metrics and did not trigger saturation or authority limits.",
            "",
        ]
    if best:
        lines.append(
            "Best final horizontal error: "
            f"angle={best['vane_angle_max_deg']} deg, rate={best['vane_rate_limit_deg_s']} deg/s, "
            f"T_max_factor={best['T_max_factor']}, final_abs_x_error={float(best['final_abs_x_error']):.3f} m."
        )
    if worst_auth:
        if float(worst_auth["authority_limited_percent"]) > 0.0:
            lines.append(
                "Highest authority-limited percentage: "
                f"angle={worst_auth['vane_angle_max_deg']} deg, rate={worst_auth['vane_rate_limit_deg_s']} deg/s, "
                f"T_max_factor={worst_auth['T_max_factor']}, authority_limited={float(worst_auth['authority_limited_percent']):.1f}%."
            )
        else:
            lines.append("No saturation or authority limits were triggered in this run.")
    lines += [
        "",
        "## Sensitivity Check",
        "",
        f"- unique final_abs_x_error values: {stats.get('unique_final_abs_x_error', 0)}",
        f"- final_abs_x_error range: {float(stats.get('min_final_abs_x_error', 0.0)):.3f} to {float(stats.get('max_final_abs_x_error', 0.0)):.3f} m",
        f"- max_vane_cmd_deg range: {float(stats.get('min_max_vane_cmd_deg', 0.0)):.3f} to {float(stats.get('max_max_vane_cmd_deg', 0.0)):.3f} deg",
        f"- max_vane_actual_deg range: {float(stats.get('min_max_vane_actual_deg', 0.0)):.3f} to {float(stats.get('max_max_vane_actual_deg', 0.0)):.3f} deg",
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
        "| vane_angle_max_deg | vane_rate_limit_deg_s | T_max_factor | pass | failure_reason | final_x_err | final_z_err | max_theta_deg | mixer_sat_% | auth_limited_% | servo_rate_sat_% | motor_sat_% |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['vane_angle_max_deg']} | {row['vane_rate_limit_deg_s']} | {row['T_max_factor']} | "
            f"{'PASS' if row['pass'] else 'FAIL'} | {row['failure_reason']} | {float(row['final_abs_x_error']):.3f} | "
            f"{float(row['final_abs_z_error']):.3f} | {float(row['max_theta_deg']):.2f} | "
            f"{float(row['mixer_saturation_percent']):.1f} | {float(row['authority_limited_percent']):.1f} | "
            f"{float(row['servo_rate_saturation_percent']):.1f} | {float(row['motor_saturation_percent']):.1f} |"
        )
    lines += [
        "",
        "Full PR #3 can expand this into denser authority maps, richer plots, and future four-vane mixer preparation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_plot(rows: list[dict[str, float | str | bool]], path: Path, required: bool = False) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        message = "matplotlib is not installed; authority sweep CSV and Markdown were still generated."
        if required:
            raise RuntimeError(message)
        print(message)
        return []
    if not rows:
        return []

    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    thrust_factors = sorted({float(row["T_max_factor"]) for row in rows})
    angles = sorted({float(row["vane_angle_max_deg"]) for row in rows})
    rates = sorted({float(row["vane_rate_limit_deg_s"]) for row in rows})
    paths: list[Path] = []
    for thrust_factor in thrust_factors:
        subset = [row for row in rows if float(row["T_max_factor"]) == thrust_factor]
        error_grid = [[float("nan") for _ in angles] for _ in rates]
        pass_grid = [[float("nan") for _ in angles] for _ in rates]
        for row in subset:
            i = rates.index(float(row["vane_rate_limit_deg_s"]))
            j = angles.index(float(row["vane_angle_max_deg"]))
            error_grid[i][j] = float(row["final_abs_x_error"])
            pass_grid[i][j] = 1.0 if row["pass"] else 0.0

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        error_image = axes[0].imshow(error_grid, origin="lower", aspect="auto")
        axes[0].set_title(f"final_abs_x_error, T_max_factor={thrust_factor:g}")
        axes[0].set_xticks(range(len(angles)))
        axes[0].set_xticklabels([f"{v:g}" for v in angles])
        axes[0].set_yticks(range(len(rates)))
        axes[0].set_yticklabels([f"{v:g}" for v in rates])
        axes[0].set_xlabel("vane_angle_max_deg")
        axes[0].set_ylabel("vane_rate_limit_deg_s")
        fig.colorbar(error_image, ax=axes[0], label="m")

        pass_image = axes[1].imshow(pass_grid, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        axes[1].set_title("pass/fail")
        axes[1].set_xticks(range(len(angles)))
        axes[1].set_xticklabels([f"{v:g}" for v in angles])
        axes[1].set_yticks(range(len(rates)))
        axes[1].set_yticklabels([f"{v:g}" for v in rates])
        axes[1].set_xlabel("vane_angle_max_deg")
        axes[1].set_ylabel("vane_rate_limit_deg_s")
        fig.colorbar(pass_image, ax=axes[1], ticks=[0, 1], label="0 fail / 1 pass")

        factor_label = str(thrust_factor).replace(".", "p")
        out = output_dir / f"loiter_authority_sweep_Tmax_{factor_label}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        paths.append(out)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a modest headless LOITER control-authority sweep.")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--params", default="params/loiter_example.json")
    parser.add_argument("--scenario", default="authority_stress")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--vane-angle-max-deg", default="0.5,1,2,3,5,8,12,20")
    parser.add_argument("--vane-rate-limit-deg-s", default="5,10,20,40,80,160")
    parser.add_argument("--T-max-factor", default="1.05,1.1,1.2,1.4,1.8,2.5")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots", action="store_true", default=None, help="Require matplotlib plots.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if any analytical threshold fails.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    rows = run_sweep(args)
    csv_path = write_csv(rows, output_dir / "loiter_authority_sweep.csv")
    md_path = write_markdown(rows, output_dir / "loiter_authority_sweep.md", args.scenario)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if not args.no_plots:
        plot_paths = write_plot(rows, output_dir / "loiter_authority_sweep.png", required=bool(args.plots))
        for plot_path in plot_paths:
            print(f"Wrote {plot_path}")
    if args.strict and any(not row["pass"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
