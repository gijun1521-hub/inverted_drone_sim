from __future__ import annotations

import argparse
from pathlib import Path

from analysis.moving_mass_comparison import (
    default_variants,
    resolve_comparison_scenarios,
    run_moving_mass_comparison,
    write_csv,
    write_markdown,
    write_plots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare vane-only, legacy moving-mass, and total-COM geometry variants."
    )
    parser.add_argument("--output-dir", default="results/analysis/moving_mass_comparison")
    parser.add_argument("--params", default="params/loiter_example.json")
    parser.add_argument("--scenario", default="pitch_assist_probe", help="Scenario name, or 'all'.")
    parser.add_argument("--duration", type=float, help="Override scenario duration in seconds.")
    parser.add_argument("--fixed-target-m", type=float, default=0.02)
    parser.add_argument("--assist-gain-m-per-Nm", type=float, default=0.025)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots", action="store_true", default=None, help="Require matplotlib plots.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    scenarios = resolve_comparison_scenarios(args.scenario, args.duration)
    variants = default_variants(
        fixed_target_m=args.fixed_target_m,
        proportional_gain_m_per_Nm=args.assist_gain_m_per_Nm,
    )
    results = run_moving_mass_comparison(param_path=args.params, scenarios=scenarios, variants=variants)
    csv_path = write_csv(results, output_dir / "moving_mass_comparison.csv")
    md_path = write_markdown(results, output_dir / "moving_mass_comparison.md")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if not args.no_plots:
        for path in write_plots(results, output_dir, required=bool(args.plots)):
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
