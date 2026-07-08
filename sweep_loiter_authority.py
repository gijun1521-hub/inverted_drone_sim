from __future__ import annotations

import argparse
from pathlib import Path

from analysis.vane_authority import (
    VaneAuthorityGrid,
    default_grid,
    parse_float_list,
    resolve_authority_scenarios,
    run_authority_sweep,
    sweep_sensitivity,
    write_csv,
    write_markdown,
    write_plots,
)


def build_parser() -> argparse.ArgumentParser:
    default = default_grid(False)
    parser = argparse.ArgumentParser(description="Run headless LOITER vane authority mapping.")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--params", default="params/loiter_example.json")
    parser.add_argument("--scenario", default="authority_stress", help="Authority stress scenario name, or 'all'.")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--vane-angle-max-deg", default=",".join(f"{v:g}" for v in default.vane_angle_max_deg))
    parser.add_argument("--vane-rate-limit-deg-s", default=",".join(f"{v:g}" for v in default.vane_rate_limit_deg_s))
    parser.add_argument("--T-max-factor", default=",".join(f"{v:g}" for v in default.T_max_factor))
    parser.add_argument("--quick", action="store_true", help="Use a small grid suitable for fast checks.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots", action="store_true", default=None, help="Require matplotlib plots.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero only when analytical checks fail.")
    return parser


def grid_from_args(args: argparse.Namespace) -> VaneAuthorityGrid:
    if getattr(args, "quick", False):
        return default_grid(True)
    return VaneAuthorityGrid(
        parse_float_list(args.vane_angle_max_deg),
        parse_float_list(args.vane_rate_limit_deg_s),
        parse_float_list(args.T_max_factor),
    )


def run_sweep(args: argparse.Namespace):
    scenarios = resolve_authority_scenarios(args.scenario, args.duration)
    return run_authority_sweep(param_path=args.params, scenarios=scenarios, grid=grid_from_args(args))


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    rows = run_sweep(args)
    csv_path = write_csv(rows, output_dir / "loiter_authority_sweep.csv")
    md_path = write_markdown(rows, output_dir / "loiter_authority_sweep.md", args.scenario)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if not args.no_plots:
        for plot_path in write_plots(rows, output_dir, required=bool(args.plots)):
            print(f"Wrote {plot_path}")
    stats = sweep_sensitivity(rows)
    if stats.get("inconclusive"):
        print("INCONCLUSIVE: sweep parameters did not affect important metrics.")
    if args.strict and any(not row["pass"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
