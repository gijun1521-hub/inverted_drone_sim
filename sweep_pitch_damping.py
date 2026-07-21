from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from analysis.pitch_damping_retune import (
    BaselineMismatchError,
    DEFAULT_OUTPUT_DIR,
    WorkflowOptions,
    run_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic staged Vane-only pitch damping retune."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="New result directory (default: results/analysis/pitch_damping_retune).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke mode only; reduced scenarios/candidates and never writes a selected profile.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Recompute selected work instead of accepting fingerprint-matched scenario rows.",
    )
    parser.add_argument(
        "--stage", choices=("baseline", "all"), default="all",
        help="Run only fresh Stage 0 diagnostics or the complete staged workflow.",
    )
    parser.add_argument(
        "--allow-baseline-mismatch", action="store_true",
        help=(
            "Explicitly allow candidate search after Stage 0 records a PR #19 behavior mismatch. "
            "Without this flag the search stops before the first candidate."
        ),
    )
    return parser


def run_from_args(args: argparse.Namespace) -> dict:
    return run_workflow(
        WorkflowOptions(
            output_dir=args.output_dir,
            quick=bool(args.quick),
            resume=not bool(args.no_resume),
            allow_baseline_mismatch=bool(args.allow_baseline_mismatch),
            stage=str(args.stage),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_from_args(args)
    except BaselineMismatchError as exc:
        print(f"baseline_mismatch: {exc}")
        return 2
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"pitch_damping_retune_error: {exc}")
        return 1
    print(json.dumps(result.get("metadata", result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
