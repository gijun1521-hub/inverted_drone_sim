from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.moving_mass_gain_resweep import DEFAULT_OUTPUT_DIR, run_resweep


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic moving-mass assist-gain resweep.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    try:
        summary = run_resweep(args.output_dir, resume=not args.no_resume)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"moving_mass_gain_resweep_error: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
