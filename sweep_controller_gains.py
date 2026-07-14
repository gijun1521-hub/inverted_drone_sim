from __future__ import annotations

import argparse
from pathlib import Path

from analysis.controller_grid_search import STAGES, WorkflowOptions, run_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run staged, deterministic controller-gain grid searches and export CSV/XLSX reports."
    )
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--output-dir", default="results/analysis/controller_grid_search")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke grids and 0.30 s scenarios.")
    parser.add_argument("--no-resume", action="store_true", help="Discard scenario rows and recompute every selected stage.")
    parser.add_argument(
        "--tail-window",
        type=float,
        default=2.0,
        help="LOITER and moving-mass tail window in seconds; RATE and attitude use stage-specific windows.",
    )
    parser.add_argument("--vane-params", default="params/loiter_example.json")
    parser.add_argument("--moving-mass-params", default="params/moving_mass_prototype_2kg.json")
    parser.add_argument(
        "--profile-output-dir",
        default=None,
        help=(
            "Tuned-profile destination. Defaults to params for full searches and "
            "<output-dir>/profiles for quick searches."
        ),
    )
    parser.add_argument("--top-pd-count", type=int, default=3)
    return parser


def run_from_args(args: argparse.Namespace) -> dict:
    return run_workflow(
        WorkflowOptions(
            stage=args.stage,
            output_dir=Path(args.output_dir),
            quick=bool(args.quick),
            resume=not bool(args.no_resume),
            tail_window_s=float(args.tail_window),
            vane_param_source=Path(args.vane_params),
            moving_mass_param_source=Path(args.moving_mass_params),
            profile_output_dir=(
                Path(args.profile_output_dir) if args.profile_output_dir is not None else None
            ),
            top_pd_count=int(args.top_pd_count),
        )
    )


def main() -> int:
    result = run_from_args(build_parser().parse_args())
    metadata = result["metadata"]
    print(f"Wrote {result['workbook_path']}")
    print(f"Wrote {result['markdown_path']}")
    if result["vane_profile_path"] is not None:
        print(f"Wrote {result['vane_profile_path']}")
    if result["moving_mass_profile_path"] is not None:
        print(f"Wrote {result['moving_mass_profile_path']}")
    print(
        f"Completed {metadata['candidate_count']} candidates and "
        f"{metadata['scenario_run_count']} unique scenario rows in {result['runtime_s']:.3f}s"
    )
    print(f"Ribbon/comet assessment: {result['ribbon_assessment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
