from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from analysis.headless_loiter import run_headless_loiter, scenario_by_name


SWEEP_FIELDS = [
    "vane_angle_max_deg",
    "vane_rate_limit_deg_s",
    "T_max_factor",
    "pass",
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


def _parse_float_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


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
                        "pass": bool(m["pass"]),
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
    if best:
        lines.append(
            "Best final horizontal error: "
            f"angle={best['vane_angle_max_deg']} deg, rate={best['vane_rate_limit_deg_s']} deg/s, "
            f"T_max_factor={best['T_max_factor']}, final_abs_x_error={float(best['final_abs_x_error']):.3f} m."
        )
    if worst_auth:
        lines.append(
            "Highest authority-limited percentage: "
            f"angle={worst_auth['vane_angle_max_deg']} deg, rate={worst_auth['vane_rate_limit_deg_s']} deg/s, "
            f"T_max_factor={worst_auth['T_max_factor']}, authority_limited={float(worst_auth['authority_limited_percent']):.1f}%."
        )
    lines += [
        "",
        "| vane_angle_max_deg | vane_rate_limit_deg_s | T_max_factor | pass | final_x_err | final_z_err | max_theta_deg | mixer_sat_% | auth_limited_% | servo_rate_sat_% |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['vane_angle_max_deg']} | {row['vane_rate_limit_deg_s']} | {row['T_max_factor']} | "
            f"{'PASS' if row['pass'] else 'FAIL'} | {float(row['final_abs_x_error']):.3f} | "
            f"{float(row['final_abs_z_error']):.3f} | {float(row['max_theta_deg']):.2f} | "
            f"{float(row['mixer_saturation_percent']):.1f} | {float(row['authority_limited_percent']):.1f} | "
            f"{float(row['servo_rate_saturation_percent']):.1f} |"
        )
    lines += [
        "",
        "Full PR #3 can expand this into denser authority maps, richer plots, and future four-vane mixer preparation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_plot(rows: list[dict[str, float | str | bool]], path: Path, required: bool = False) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        message = "matplotlib is not installed; authority sweep CSV and Markdown were still generated."
        if required:
            raise RuntimeError(message)
        print(message)
        return None
    if not rows:
        return None
    labels = [f"{r['vane_angle_max_deg']}/{r['vane_rate_limit_deg_s']}/{r['T_max_factor']}" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.22), 5))
    colors = ["#4c9f70" if r["pass"] else "#c75d5d" for r in rows]
    ax.bar(range(len(rows)), [float(r["final_abs_x_error"]) for r in rows], color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("final_abs_x_error (m)")
    ax.set_xlabel("vane_angle_deg / vane_rate_deg_s / T_max_factor")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a modest headless LOITER control-authority sweep.")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--params", default="params/loiter_example.json")
    parser.add_argument("--scenario", default="horizontal_impulse_recovery")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--vane-angle-max-deg", default="5,10,15,20,25")
    parser.add_argument("--vane-rate-limit-deg-s", default="60,120,180,300")
    parser.add_argument("--T-max-factor", default="1.4,1.8,2.2,2.5")
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
        plot_path = write_plot(rows, output_dir / "loiter_authority_sweep.png", required=bool(args.plots))
        if plot_path:
            print(f"Wrote {plot_path}")
    if args.strict and any(not row["pass"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
