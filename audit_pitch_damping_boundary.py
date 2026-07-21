from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from analysis.pitch_damping_boundary_audit import DEFAULT_AUDIT_DIR, run_boundary_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Targeted full-duration Stage 3C upper-boundary audit."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_boundary_audit(
            output_dir=args.output_dir, resume=not bool(args.no_resume)
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"pitch_damping_boundary_audit_error: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
