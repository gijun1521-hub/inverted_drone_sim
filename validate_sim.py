from __future__ import annotations

import argparse

try:
    from .params import load_rigid_body_config
    from .validation.runner import run_validation
except ImportError:  # pragma: no cover - direct script execution
    from params import load_rigid_body_config
    from validation.runner import run_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run headless simulator validation scenarios.")
    parser.add_argument("--params", default=None, help="Optional JSON parameter override file.")
    parser.add_argument("--results-dir", default="results", help="Directory for validation outputs.")
    args = parser.parse_args()

    cfg = load_rigid_body_config(args.params)
    results, summary, report = run_validation(cfg, args.results_dir)
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'}: {result.name} - {result.details}")
    print(f"saved: {summary}")
    print(f"saved: {report}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
