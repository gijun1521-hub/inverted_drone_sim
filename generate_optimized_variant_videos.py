from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any

from analysis.optimized_variant_video_scenarios import (
    DEFAULT_OUTPUT_DIR,
    PR24_OUTPUT_DIR,
    compare_to_pr25,
    directory_hashes,
    run_all_scenarios,
    validate_result_set,
    write_manifest,
    write_metrics_csv,
    write_summary_markdown,
)
from analysis.seminar_video_renderer import (
    RenderConfig,
    detect_ffmpeg,
    render_optimized_comparison,
)


REQUIRED_NON_MP4 = {
    "final_optimized_comparison.gif",
    "final_optimized_thumbnail.png",
    "scenario_metrics.csv",
    "scenario_summary.md",
    "manifest.json",
}
REQUIRED_MP4 = {
    "loiter_vane_only.mp4",
    "loiter_moving_mass_assist.mp4",
    "forward_1m_vane_only.mp4",
    "forward_1m_moving_mass_assist.mp4",
    "final_optimized_comparison_2x2.mp4",
}


def _probe_mp4(
    path: Path,
    *,
    expected_size: tuple[int, int],
    expected_fps: int,
    expected_duration_s: float,
    expected_frame_count: int,
) -> dict[str, Any]:
    encoder = detect_ffmpeg()
    if not encoder.available:
        raise RuntimeError("MP4 verification requires FFmpeg")
    info = subprocess.run(
        [encoder.executable, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if info.returncode != 0:
        raise RuntimeError(f"cannot decode {path.name}: {info.stderr.strip()}")
    video_line = next(
        (line.strip() for line in info.stderr.splitlines() if "Video:" in line), ""
    )
    size_match = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", video_line)
    fps_match = re.search(r"([0-9.]+) fps", video_line)
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info.stderr)
    frame_matches = re.findall(r"frame=\s*(\d+)", info.stderr)
    if not size_match or not fps_match or not duration_match or not frame_matches:
        raise RuntimeError(f"incomplete FFmpeg metadata for {path.name}: {video_line}")
    size = (int(size_match.group(1)), int(size_match.group(2)))
    fps = float(fps_match.group(1))
    duration = (
        3600.0 * int(duration_match.group(1))
        + 60.0 * int(duration_match.group(2))
        + float(duration_match.group(3))
    )
    decoded_frames = int(frame_matches[-1])
    checks = {
        "opens_and_decodes": info.returncode == 0,
        "nonzero_file": path.stat().st_size > 0,
        "dimensions": size == expected_size,
        "fps": abs(fps - expected_fps) <= 1e-9,
        "duration": abs(duration - expected_duration_s) <= max(0.05, 1.0 / expected_fps),
        "codec_h264": "Video: h264" in video_line,
        "pixel_format_yuv420p": "yuv420p" in video_line,
        "nonzero_frames": decoded_frames > 0,
        "frame_count": decoded_frames == expected_frame_count,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{path.name}: media checks failed: {', '.join(failed)}")
    return {
        "name": path.name,
        "size_px": list(size),
        "fps": fps,
        "duration_s": duration,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "decoded_frame_count": decoded_frames,
        "size_bytes": path.stat().st_size,
        "checks": checks,
    }


def verify_mp4_outputs(output_dir: Path, render_report: dict[str, Any]) -> dict[str, Any]:
    duration = float(render_report["duration_s"])
    fps = int(render_report["fps"])
    frames = int(render_report["frame_count"])
    individual_size = tuple(render_report["individual_size_px"])
    composite_size = tuple(render_report["composite_size_px"])
    files = []
    for name in sorted(REQUIRED_MP4):
        expected_size = composite_size if name == "final_optimized_comparison_2x2.mp4" else individual_size
        files.append(
            _probe_mp4(
                output_dir / name,
                expected_size=expected_size,
                expected_fps=fps,
                expected_duration_s=duration,
                expected_frame_count=frames,
            )
        )
    return {"passed": True, "file_count": len(files), "files": files}


def verify_required_outputs(output_dir: Path, *, mp4_expected: bool) -> None:
    required = set(REQUIRED_NON_MP4)
    if mp4_expected:
        required |= REQUIRED_MP4
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"required optimized-video artifacts are missing: {', '.join(missing)}")


def generate(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    duration_s: float = 10.0,
    render_config: RenderConfig | None = None,
    write_mp4: bool = True,
) -> tuple[list, dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pr24_hashes_before = directory_hashes(PR24_OUTPUT_DIR)
    results = run_all_scenarios(duration_s=duration_s)
    validation = validate_result_set(results)
    pr25_comparison = compare_to_pr25(results)
    metrics_path = write_metrics_csv(results, output_dir / "scenario_metrics.csv")
    summary_path = write_summary_markdown(
        results, output_dir / "scenario_summary.md", pr25_comparison
    )
    report = render_optimized_comparison(
        results,
        output_dir,
        config=render_config or RenderConfig(optimized_hud=True),
        write_mp4=write_mp4,
    )
    report["artifacts"] = sorted(
        set(report["artifacts"] + [metrics_path.name, summary_path.name])
    )
    report["simulation_validation"] = validation
    report["pr25_selected_result_comparison"] = pr25_comparison
    mp4_expected = bool(write_mp4 and report["encoder"]["available"])
    if mp4_expected:
        report["media_verification"] = verify_mp4_outputs(output_dir, report)
    else:
        report["media_verification"] = {
            "passed": not write_mp4,
            "skipped": True,
            "reason": "MP4 disabled" if not write_mp4 else "FFmpeg unavailable",
        }
    manifest_path = write_manifest(
        results,
        output_dir,
        report,
        validation,
        pr25_comparison,
        pr24_hashes_before,
    )
    report["artifacts"] = sorted(set(report["artifacts"] + [manifest_path.name]))
    verify_required_outputs(output_dir, mp4_expected=mp4_expected)
    return results, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render deterministic videos for the two independently optimized PR #25 controllers."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--gif-fps", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=None)
    parser.add_argument("--panel-height", type=int, default=None)
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 0.25 s reduced-resolution workflow check in a separate smoke directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR.with_name(
            "final_optimized_controller_videos_smoke"
        )
        duration = args.duration if args.duration is not None else 0.25
        config = RenderConfig(
            fps=args.fps or 6,
            panel_width=args.panel_width or 480,
            panel_height=args.panel_height or 270,
            gif_fps=args.gif_fps or 3,
            gif_width=480,
            gif_height=270,
            optimized_hud=True,
        )
    else:
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        duration = args.duration if args.duration is not None else 10.0
        config = RenderConfig(
            fps=args.fps or 30,
            panel_width=args.panel_width or 960,
            panel_height=args.panel_height or 540,
            gif_fps=args.gif_fps or 10,
            gif_width=960,
            gif_height=540,
            optimized_hud=True,
        )
    results, report = generate(
        output_dir,
        duration_s=duration,
        render_config=config,
        write_mp4=not args.no_mp4,
    )
    print(f"Generated {len(results)} deterministic simulations in {Path(output_dir).resolve()}")
    print(f"PR #25 comparisons: {report['pr25_selected_result_comparison']['comparison_count']} PASS")
    print(f"Encoder: {report['encoder']['source']} ({report['encoder']['version']})")
    for name in report["artifacts"]:
        path = Path(output_dir) / name
        print(f"{name}: {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
