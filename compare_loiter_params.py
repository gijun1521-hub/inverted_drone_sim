from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analysis.headless_loiter import (
    LoiterRunResult,
    default_loiter_scenarios,
    run_headless_loiter,
    save_loiter_timeseries,
    scenario_by_name,
)


DEFAULT_PARAMS = [
    "params/loiter_sluggish_example.json",
    "params/loiter_example.json",
    "params/loiter_aggressive_example.json",
]

SUMMARY_FIELDS = [
    "param_file",
    "scenario_name",
    "pass",
    "crash_reason",
    "duration_s",
    "max_abs_x_error",
    "final_abs_x_error",
    "rms_x_error",
    "max_abs_z_error",
    "final_abs_z_error",
    "rms_z_error",
    "max_theta_deg",
    "final_theta_deg",
    "max_omega_deg_s",
    "max_vane_cmd_deg",
    "max_vane_actual_deg",
    "max_thrust_cmd_N",
    "max_thrust_actual_N",
    "motor_saturation_percent",
    "servo_angle_saturation_percent",
    "servo_rate_saturation_percent",
    "mixer_saturation_percent",
    "mixer_angle_saturation_percent",
    "authority_limited_percent",
    "final_x",
    "final_z",
    "final_vx",
    "final_vz",
    "notes",
]


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(results: list[LoiterRunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.metrics)
    return path


def write_markdown(results: list[LoiterRunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.metrics["pass"])
    total = len(results)
    lines = [
        "# Headless LOITER Parameter Comparison",
        "",
        "This report compares LOITER parameter sets in the analytical 2D single-fan simulator.",
        "It is not calibrated real-flight prediction; use the outputs as relative indicators for stability, saturation, authority margin, and tuning sensitivity.",
        "",
        f"Summary: {passed}/{total} analytical scenario checks passed.",
        "",
        "| param_file | scenario | pass | final_x_err | final_z_err | rms_x | max_theta_deg | mixer_sat_% | authority_limited_% | servo_rate_sat_% | notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        m = result.metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    str(m["param_file"]),
                    str(m["scenario_name"]),
                    _fmt(m["pass"]),
                    _fmt(m["final_abs_x_error"]),
                    _fmt(m["final_abs_z_error"]),
                    _fmt(m["rms_x_error"]),
                    _fmt(m["max_theta_deg"]),
                    _fmt(m["mixer_saturation_percent"]),
                    _fmt(m["authority_limited_percent"]),
                    _fmt(m["servo_rate_saturation_percent"]),
                    str(m["notes"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    by_param: dict[str, list[LoiterRunResult]] = {}
    for result in results:
        by_param.setdefault(result.param_file, []).append(result)
    avg_rows = []
    for param, param_results in by_param.items():
        avg_x = sum(float(r.metrics["final_abs_x_error"]) for r in param_results) / len(param_results)
        avg_sat = sum(float(r.metrics["mixer_saturation_percent"]) for r in param_results) / len(param_results)
        avg_auth = sum(float(r.metrics["authority_limited_percent"]) for r in param_results) / len(param_results)
        avg_rows.append((avg_x, avg_sat, avg_auth, param))
    most_stable = min(avg_rows, default=(0.0, 0.0, 0.0, "n/a"))[3]
    most_saturated = max(avg_rows, default=(0.0, 0.0, 0.0, "n/a"), key=lambda x: x[1] + x[2])[3]
    max_limit_activity = max((sat + auth for _x, sat, auth, _param in avg_rows), default=0.0)
    limit_line = (
        f"- Highest average saturation/authority activity in this run: `{most_saturated}`."
        if max_limit_activity > 0.0
        else "- No saturation or authority limits were triggered in this comparison run."
    )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Lowest average final horizontal error in this run: `{most_stable}`.",
        limit_line,
        "- Sluggish parameter sets should show slower recovery and larger residual errors; aggressive sets may reduce error faster but can demand more pitch/vane activity.",
        "- Saturation is not automatically a failure. It is a design signal that the requested moment or actuator motion is approaching the current model limits.",
        "",
        "## What To Inspect Manually",
        "",
        "- Time-series rows for scenarios with large final velocity or large residual error.",
        "- `mixer_saturation_percent`, `authority_limited_percent`, and `servo_rate_saturation_percent` together, not in isolation.",
        "- Attitude peaks and whether they are caused by pilot target motion, impulse recovery, or low authority.",
        "",
        "## Next Steps",
        "",
        "- Expand vane authority sweeps into richer maps.",
        "- Prepare for future four-vane SingleCopter mixer modeling.",
        "- Add experimental calibration with bench data and real flight logs later.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_plots(results: list[LoiterRunResult], output_dir: Path, required: bool = False) -> list[Path]:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    first_by_param: dict[str, LoiterRunResult] = {}
    for result in results:
        if result.scenario.name == "stick_move_release":
            first_by_param.setdefault(result.param_file, result)
    if first_by_param:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for param, result in first_by_param.items():
            t = [float(r["time"]) for r in result.rows]
            axes[0].plot(t, [float(r["x"]) for r in result.rows], label=param)
            axes[0].plot(t, [float(r["target_x"]) for r in result.rows], linestyle="--", alpha=0.5)
            axes[1].plot(t, [float(r["z"]) for r in result.rows], label=param)
            axes[1].plot(t, [float(r["target_z"]) for r in result.rows], linestyle="--", alpha=0.5)
            axes[2].plot(t, [float(r["theta"]) * 57.295779513 for r in result.rows], label=param)
        axes[0].set_ylabel("x / target_x (m)")
        axes[1].set_ylabel("z / target_z (m)")
        axes[2].set_ylabel("theta (deg)")
        axes[2].set_xlabel("time (s)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        path = output_dir / "loiter_position_response.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{Path(r.param_file).stem}\n{r.scenario.name}" for r in results]
    ax.bar(range(len(results)), [float(r.metrics["final_abs_x_error"]) for r in results], label="final_abs_x_error")
    ax.bar(range(len(results)), [float(r.metrics["final_abs_z_error"]) for r in results], bottom=[float(r.metrics["final_abs_x_error"]) for r in results], label="final_abs_z_error")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("stacked final abs error (m)")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "loiter_param_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def run_comparison(args: argparse.Namespace) -> list[LoiterRunResult]:
    params = args.params or DEFAULT_PARAMS
    scenarios = [scenario_by_name(args.scenario, args.duration)] if args.scenario else default_loiter_scenarios(args.duration)
    results: list[LoiterRunResult] = []
    for param in params:
        for scenario in scenarios:
            result = run_headless_loiter(param, scenario)
            results.append(result)
            if args.save_timeseries:
                stem = f"{Path(param).stem}_{scenario.name}.csv"
                save_loiter_timeseries(result.rows, Path(args.output_dir) / "timeseries" / stem)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare LOITER parameter sets without opening pygame.")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--params", action="append", help="Parameter JSON path. May be repeated.")
    parser.add_argument("--scenario", help="Run only one scenario by name.")
    parser.add_argument("--duration", type=float, help="Override scenario duration in seconds.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots", action="store_true", default=None, help="Require matplotlib plots.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if any analytical threshold fails.")
    parser.add_argument("--save-timeseries", action="store_true", help="Write per-run time-series CSV files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    results = run_comparison(args)
    csv_path = write_csv(results, output_dir / "loiter_param_comparison.csv")
    md_path = write_markdown(results, output_dir / "loiter_param_comparison.md")
    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = write_plots(results, output_dir, required=bool(args.plots))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    if args.strict and any(not result.metrics["pass"] for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
