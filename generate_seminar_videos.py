from __future__ import annotations

import argparse
from pathlib import Path

from analysis.seminar_video_renderer import RenderConfig, render_seminar_comparison
from analysis.seminar_moving_mass_gain_sweep import (
    assert_selected_gain,
    run_staged_gain_sweep,
    write_gain_selection_markdown,
    write_gain_sweep_csv,
)
from analysis.seminar_video_scenarios import (
    DEFAULT_OUTPUT_DIR,
    run_all_scenarios,
    validate_result_set,
    write_manifest,
    write_metrics_csv,
    write_summary_markdown,
)


REQUIRED_NON_MP4 = {
    "seminar_scenario_comparison.gif",
    "seminar_video_thumbnail.png",
    "scenario_metrics.csv",
    "scenario_summary.md",
    "moving_mass_gain_sweep.csv",
    "moving_mass_gain_selection.md",
    "manifest.json",
}
REQUIRED_MP4 = {
    "seminar_scenario_comparison.mp4",
    "loiter_locked.mp4",
    "loiter_assist.mp4",
    "forward_1m_locked.mp4",
    "forward_1m_assist.mp4",
}


def generate(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    duration_s: float = 8.0,
    render_config: RenderConfig | None = None,
    write_mp4: bool = True,
) -> tuple[list, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gain_sweep = run_staged_gain_sweep(duration_s=8.0)
    assert_selected_gain(gain_sweep)
    sweep_path = write_gain_sweep_csv(
        gain_sweep, output_dir / "moving_mass_gain_sweep.csv"
    )
    selection_path = write_gain_selection_markdown(
        gain_sweep, output_dir / "moving_mass_gain_selection.md"
    )
    results = run_all_scenarios(duration_s=duration_s)
    validation = validate_result_set(results)
    metrics_path = write_metrics_csv(results, output_dir / "scenario_metrics.csv")
    summary_path = write_summary_markdown(results, output_dir / "scenario_summary.md")
    render_report = render_seminar_comparison(
        results,
        output_dir,
        config=render_config,
        write_mp4=write_mp4,
    )
    render_report["artifacts"] = sorted(
        set(
            render_report["artifacts"]
            + [metrics_path.name, summary_path.name, sweep_path.name, selection_path.name]
        )
    )
    render_report["simulation_validation"] = validation
    manifest_path = write_manifest(results, output_dir, render_report, gain_sweep.metadata())
    render_report["artifacts"] = sorted(set(render_report["artifacts"] + [manifest_path.name]))
    verify_required_outputs(output_dir, mp4_expected=bool(write_mp4 and render_report["encoder"]["available"]))
    return results, render_report


def verify_required_outputs(output_dir: str | Path, *, mp4_expected: bool) -> None:
    output_dir = Path(output_dir)
    required = set(REQUIRED_NON_MP4)
    if mp4_expected:
        required |= REQUIRED_MP4
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"required seminar artifacts are missing: {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic locked-vs-assist seminar comparison videos."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--gif-fps", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=None)
    parser.add_argument("--panel-height", type=int, default=None)
    parser.add_argument("--no-mp4", action="store_true", help="Explicitly use the GIF/PNG-only path.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Render a 0.25 s, reduced-size workflow check in the smoke output directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR.with_name("seminar_videos_smoke")
        duration = args.duration if args.duration is not None else 0.25
        config = RenderConfig(
            fps=args.fps or 6,
            panel_width=args.panel_width or 480,
            panel_height=args.panel_height or 270,
            gif_fps=args.gif_fps or 3,
            gif_width=480,
            gif_height=270,
        )
    else:
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        duration = args.duration if args.duration is not None else 8.0
        config = RenderConfig(
            fps=args.fps or 30,
            panel_width=args.panel_width or 960,
            panel_height=args.panel_height or 540,
            gif_fps=args.gif_fps or 10,
            gif_width=960,
            gif_height=540,
        )
    results, report = generate(
        output_dir,
        duration_s=duration,
        render_config=config,
        write_mp4=not args.no_mp4,
    )
    print(f"Generated {len(results)} deterministic simulations in {Path(output_dir).resolve()}")
    print(
        f"Encoder: {report['encoder']['source']}"
        + (f" ({report['encoder']['version']})" if report["encoder"]["available"] else "")
    )
    for name in report["artifacts"]:
        path = Path(output_dir) / name
        print(f"{name}: {path.stat().st_size} bytes")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
