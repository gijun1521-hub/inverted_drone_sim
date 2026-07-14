from __future__ import annotations

import math
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .seminar_video_scenarios import SeminarRunResult
except ImportError:  # pragma: no cover - direct script execution from repository root
    from analysis.seminar_video_scenarios import SeminarRunResult


@dataclass(frozen=True)
class RenderConfig:
    fps: int = 30
    panel_width: int = 960
    panel_height: int = 540
    gif_fps: int = 10
    gif_width: int = 960
    gif_height: int = 540
    x_bounds_m: tuple[float, float] = (-1.5, 2.0)
    z_bounds_m: tuple[float, float] = (0.0, 2.5)

    def validate(self) -> None:
        if self.fps <= 0 or self.gif_fps <= 0:
            raise ValueError("FPS values must be positive")
        for value in (self.panel_width, self.panel_height, self.gif_width, self.gif_height):
            if value <= 0:
                raise ValueError("render dimensions must be positive")
        if self.panel_width % 2 or self.panel_height % 2:
            raise ValueError("MP4 panel dimensions must be even for yuv420p")


@dataclass(frozen=True)
class EncoderInfo:
    available: bool
    source: str
    executable: str
    version: str


@dataclass(frozen=True)
class PreparedRun:
    result: SeminarRunResult
    timestamps: np.ndarray
    values: dict[str, np.ndarray]


def detect_ffmpeg() -> EncoderInfo:
    system_executable = shutil.which("ffmpeg")
    if system_executable:
        return EncoderInfo(True, "system ffmpeg", system_executable, _ffmpeg_version(system_executable))
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable and Path(executable).is_file():
            return EncoderInfo(True, "imageio-ffmpeg", executable, _ffmpeg_version(executable))
    except (ImportError, RuntimeError, OSError):
        pass
    return EncoderInfo(False, "unavailable", "", "")


def _ffmpeg_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return completed.stdout.splitlines()[0].strip() if completed.stdout else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def frame_timestamps(duration_s: float, fps: int) -> np.ndarray:
    frame_count = int(round(float(duration_s) * int(fps)))
    return np.arange(frame_count, dtype=float) / float(fps)


def synchronized_frame_timestamps(
    results: Iterable[SeminarRunResult], fps: int
) -> np.ndarray:
    results = list(results)
    if not results:
        raise ValueError("at least one result is required")
    durations = {float(result.scenario.config.duration_s) for result in results}
    if len(durations) != 1:
        raise ValueError(f"scenario durations are not synchronized: {sorted(durations)}")
    return frame_timestamps(durations.pop(), fps)


def _prepend_initial(
    result: SeminarRunResult,
    row_times: np.ndarray,
    values: np.ndarray,
    initial_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    if row_times.size and row_times[0] <= 1e-12:
        return row_times, values
    return np.concatenate(([0.0], row_times)), np.concatenate(([initial_value], values))


def prepare_run(result: SeminarRunResult, timestamps: np.ndarray) -> PreparedRun:
    rows = result.run.rows
    row_times = np.asarray([float(row["sim_time"]) for row in rows], dtype=float)
    scenario = result.scenario.config
    rb = result.rb_config
    initial = {
        "x_cg": scenario.initial_x,
        "z_cg": scenario.initial_z,
        "theta": math.radians(scenario.initial_theta_deg),
        "vane_angle_actual": 0.0,
        "vane_angle_cmd": 0.0,
        "moving_mass_offset_m": 0.0,
        "moving_mass_target_m": 0.0,
        "total_com_body_right_m": 0.0,
        "total_com_body_up_m": rb.moving_mass.mass_kg
        * rb.moving_mass.moving_mass_body_up_offset_m
        / rb.m,
        "thrust_actual": rb.hover_thrust,
    }
    continuous_keys = tuple(initial)
    sampled: dict[str, np.ndarray] = {}
    for key in continuous_keys:
        source = np.asarray([float(row[key]) for row in rows], dtype=float)
        source_times, source = _prepend_initial(result, row_times, source, initial[key])
        sampled[key] = np.interp(timestamps, source_times, source)

    target_x = np.full(timestamps.shape, scenario.target_x, dtype=float)
    if scenario.target_step_time_s is not None and scenario.target_step_x is not None:
        target_x[timestamps >= scenario.target_step_time_s - 1e-12] = scenario.target_step_x
    target_z = np.full(timestamps.shape, scenario.target_z, dtype=float)
    if scenario.target_step_time_s is not None and scenario.target_step_z is not None:
        target_z[timestamps >= scenario.target_step_time_s - 1e-12] = scenario.target_step_z
    sampled["target_x"] = target_x
    sampled["target_z"] = target_z
    sampled["x_error"] = target_x - sampled["x_cg"]
    return PreparedRun(result=result, timestamps=timestamps, values=sampled)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("arialbd.ttf" if bold else "arial.ttf"),
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(8, size))
        except OSError:
            continue
    return ImageFont.load_default()


def _world_mapper(config: RenderConfig, size: tuple[int, int]):
    width, height = size
    scale = min(width / 960.0, height / 540.0)
    left = max(24, int(54 * scale))
    right = max(12, int(20 * scale))
    top = max(48, int(86 * scale))
    bottom = max(22, int(42 * scale))
    x0, x1 = config.x_bounds_m
    z0, z1 = config.z_bounds_m

    def world_to_pixel(point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        x, z = float(point[0]), float(point[1])
        px = left + (x - x0) / (x1 - x0) * (width - left - right)
        py = height - bottom - (z - z0) / (z1 - z0) * (height - top - bottom)
        return int(round(px)), int(round(py))

    return world_to_pixel, (left, top, width - right, height - bottom), scale


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    mapper,
    start: np.ndarray,
    vector: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    end = start + vector
    start_px = np.asarray(mapper(start), dtype=float)
    end_px = np.asarray(mapper(end), dtype=float)
    draw.line([tuple(start_px), tuple(end_px)], fill=color, width=width)
    direction = end_px - start_px
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return
    unit = direction / norm
    normal = np.array([-unit[1], unit[0]])
    head = max(5.0, 2.5 * width)
    base = end_px - head * unit
    polygon = [tuple(end_px), tuple(base + 0.45 * head * normal), tuple(base - 0.45 * head * normal)]
    draw.polygon(polygon, fill=color)


def render_panel(
    prepared: PreparedRun,
    frame_index: int,
    config: RenderConfig,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    size = size or (config.panel_width, config.panel_height)
    width, height = size
    image = Image.new("RGB", size, (247, 249, 252))
    draw = ImageDraw.Draw(image)
    mapper, plot_rect, scale = _world_mapper(config, size)
    left, top, right, bottom = plot_rect
    small = _font(int(15 * scale))
    normal = _font(int(18 * scale))
    title_font = _font(int(20 * scale), bold=True)
    line_width = max(1, int(round(2 * scale)))

    for x_tick in np.arange(-1.5, 2.01, 0.5):
        x_px, _ = mapper((x_tick, 0.0))
        draw.line([(x_px, top), (x_px, bottom)], fill=(222, 227, 234), width=1)
        draw.text((x_px - 11 * scale, bottom + 4 * scale), f"{x_tick:g}", fill=(70, 78, 90), font=small)
    for z_tick in np.arange(0.0, 2.51, 0.5):
        _, z_px = mapper((config.x_bounds_m[0], z_tick))
        draw.line([(left, z_px), (right, z_px)], fill=(222, 227, 234), width=1)
        draw.text((3 * scale, z_px - 8 * scale), f"{z_tick:g}", fill=(70, 78, 90), font=small)
    draw.rectangle(plot_rect, outline=(102, 113, 128), width=line_width)
    ground_y = mapper((0.0, 0.0))[1]
    draw.line([(left, ground_y), (right, ground_y)], fill=(82, 87, 96), width=max(2, line_width))

    values = prepared.values
    result = prepared.result
    time_s = float(prepared.timestamps[frame_index])
    x = float(values["x_cg"][frame_index])
    z = float(values["z_cg"][frame_index])
    theta = float(values["theta"][frame_index])
    target = np.array([values["target_x"][frame_index], values["target_z"][frame_index]])
    body_up = np.array([math.sin(theta), math.cos(theta)])
    body_right = np.array([math.cos(theta), -math.sin(theta)])
    total_com = np.array([x, z])
    total_com_body = np.array(
        [
            values["total_com_body_right_m"][frame_index],
            values["total_com_body_up_m"][frame_index],
        ]
    )
    fixed_body_origin = total_com - total_com_body[0] * body_right - total_com_body[1] * body_up
    half_h = 0.5 * result.rb_config.H
    half_w = 0.5 * result.rb_config.W
    top_world = fixed_body_origin + half_h * body_up
    bottom_world = fixed_body_origin - half_h * body_up
    corners = [
        bottom_world - half_w * body_right,
        bottom_world + half_w * body_right,
        top_world + half_w * body_right,
        top_world - half_w * body_right,
    ]

    trail_points = [
        mapper((values["x_cg"][idx], values["z_cg"][idx]))
        for idx in range(frame_index + 1)
    ]
    if len(trail_points) > 1:
        draw.line(trail_points, fill=(121, 184, 228), width=max(1, line_width))

    tx, tz = mapper(target)
    marker = max(5, int(7 * scale))
    draw.ellipse((tx - marker, tz - marker, tx + marker, tz + marker), fill=(255, 220, 55), outline=(115, 93, 0), width=line_width)
    draw.line([mapper(total_com), mapper((target[0], z)), mapper(target)], fill=(195, 164, 53), width=1)

    draw.polygon([mapper(corner) for corner in corners], fill=(49, 132, 219), outline=(20, 74, 132))
    rail_center = fixed_body_origin + result.rb_config.moving_mass.moving_mass_body_up_offset_m * body_up
    rail_limit = result.rb_config.moving_mass.max_offset_m
    draw.line(
        [mapper(rail_center - rail_limit * body_right), mapper(rail_center + rail_limit * body_right)],
        fill=(205, 211, 219),
        width=max(2, int(4 * scale)),
    )
    mass_position = rail_center + values["moving_mass_offset_m"][frame_index] * body_right
    mx, mz = mapper(mass_position)
    mass_radius = max(4, int(7 * scale))
    draw.ellipse(
        (mx - mass_radius, mz - mass_radius, mx + mass_radius, mz + mass_radius),
        fill=(38, 210, 225),
        outline=(10, 97, 108),
        width=line_width,
    )

    cx, cz = mapper(total_com)
    diamond = max(5, int(7 * scale))
    draw.polygon(
        [(cx, cz - diamond), (cx + diamond, cz), (cx, cz + diamond), (cx - diamond, cz)],
        fill=(232, 58, 64),
        outline=(125, 18, 25),
    )

    vane = float(values["vane_angle_actual"][frame_index])
    hinge = bottom_world - result.ui_config.vane_visual_offset_m * body_up
    neutral = -body_up
    vane_dir = math.cos(vane * result.ui_config.vane_visual_scale) * neutral + math.sin(
        vane * result.ui_config.vane_visual_scale
    ) * body_right
    vane_length = min(0.32, result.ui_config.vane_visual_length_m)
    draw.line(
        [mapper(hinge), mapper(hinge + vane_length * vane_dir)],
        fill=(164, 61, 207),
        width=max(3, int(6 * scale)),
    )

    thrust_scale = 0.35 * float(values["thrust_actual"][frame_index]) / max(
        result.rb_config.hover_thrust, 1e-9
    )
    _draw_arrow(
        draw,
        mapper,
        fixed_body_origin,
        thrust_scale * body_up,
        (241, 143, 38),
        max(2, int(4 * scale)),
    )

    scenario_cfg = result.scenario.config
    if (
        scenario_cfg.disturbance_start_s <= time_s
        < scenario_cfg.disturbance_start_s + scenario_cfg.disturbance_duration_s
    ):
        _draw_arrow(
            draw,
            mapper,
            total_com,
            np.array([0.32, 0.0]),
            (222, 73, 63),
            max(2, int(4 * scale)),
        )

    title = f"{result.scenario.display_name} - {result.variant.display_name}"
    draw.text((left, 7 * scale), title, fill=(24, 32, 45), font=title_font)
    locked_offset = result.variant.key == "locked"
    offset_mm = 0.0 if locked_offset else 1000.0 * float(values["moving_mass_offset_m"][frame_index])
    status = (
        f"t={time_s:4.2f} s    x error={values['x_error'][frame_index]:+6.3f} m    "
        f"pitch={math.degrees(theta):+6.2f} deg"
    )
    actuator = (
        f"actual vane={math.degrees(vane):+6.2f} deg    moving-mass offset={offset_mm:+6.1f} mm"
    )
    subtitle = result.variant.subtitle
    if result.variant.key == "assist":
        subtitle += f" ({result.variant.assist_gain_m_per_Nm:.4f} m/Nm)"
    draw.text((left, 31 * scale), subtitle, fill=(82, 91, 105), font=small)
    draw.text((left, 48 * scale), status, fill=(45, 54, 68), font=normal)
    draw.text((left, 70 * scale), actuator, fill=(45, 54, 68), font=normal)

    legend = [
        ((255, 220, 55), "target"),
        ((38, 210, 225), "moving mass"),
        ((232, 58, 64), "total COM"),
        ((164, 61, 207), "vane"),
        ((241, 143, 38), "thrust"),
    ]
    legend_x = left
    legend_y = height - max(17, int(20 * scale))
    for color, label in legend:
        dot = max(3, int(4 * scale))
        draw.ellipse((legend_x, legend_y, legend_x + 2 * dot, legend_y + 2 * dot), fill=color)
        draw.text((legend_x + 2 * dot + 3, legend_y - 3 * scale), label, fill=(64, 72, 84), font=small)
        legend_x += int((len(label) * 8 + 32) * scale)
    return image


class _RawH264Writer:
    def __init__(self, encoder: EncoderInfo, path: Path, size: tuple[int, int], fps: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        command = [
            encoder.executable,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def write(self, image: Image.Image) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self.process.stdin.write(image.convert("RGB").tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        if self.process.stderr is not None:
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code != 0:
            self.path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg failed for {self.path.name}: {stderr.strip()}")
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg reported success but did not create {self.path}")


def _gif_indices(timestamps: np.ndarray, gif_fps: int, video_fps: int) -> set[int]:
    if timestamps.size == 0:
        return set()
    duration = timestamps.size / float(video_fps)
    return {
        min(timestamps.size - 1, int(round(time_s * video_fps)))
        for time_s in np.arange(0.0, duration, 1.0 / gif_fps)
    }


def _write_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("at least one GIF frame is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_single_result(
    result: SeminarRunResult,
    *,
    config: RenderConfig,
    png_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    mp4_path: str | Path | None = None,
    encoder: EncoderInfo | None = None,
) -> dict[str, Any]:
    config.validate()
    timestamps = synchronized_frame_timestamps([result], config.fps)
    prepared = prepare_run(result, timestamps)
    encoder = encoder or detect_ffmpeg()
    writer = None
    if mp4_path is not None:
        if not encoder.available:
            warnings.warn("FFmpeg unavailable; MP4 was not created. GIF/PNG output remains available.")
        else:
            writer = _RawH264Writer(
                encoder, Path(mp4_path), (config.panel_width, config.panel_height), config.fps
            )
    gif_frames: list[Image.Image] = []
    gif_indices = _gif_indices(timestamps, config.gif_fps, config.fps)
    thumbnail_index = min(len(timestamps) - 1, int(round(0.5 * len(timestamps))))
    try:
        for index in range(len(timestamps)):
            panel = render_panel(prepared, index, config)
            if writer is not None:
                writer.write(panel)
            if gif_path is not None and index in gif_indices:
                gif_frames.append(panel.resize((config.gif_width, config.gif_height), Image.Resampling.LANCZOS))
            if png_path is not None and index == thumbnail_index:
                panel.save(png_path)
    finally:
        if writer is not None:
            writer.close()
    if gif_path is not None:
        _write_gif(gif_frames, Path(gif_path), config.gif_fps)
    return {
        "frame_count": len(timestamps),
        "panel_count": 1,
        "encoder": asdict(encoder),
    }


def render_seminar_comparison(
    results: Iterable[SeminarRunResult],
    output_dir: str | Path,
    *,
    config: RenderConfig | None = None,
    write_mp4: bool = True,
) -> dict[str, Any]:
    config = config or RenderConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_map = {result.key: result for result in results}
    ordered_keys = [
        ("loiter", "locked"),
        ("loiter", "assist"),
        ("forward_1m", "locked"),
        ("forward_1m", "assist"),
    ]
    missing = [key for key in ordered_keys if key not in result_map]
    if missing:
        raise ValueError(f"missing composite panels: {missing!r}")
    ordered = [result_map[key] for key in ordered_keys]
    timestamps = synchronized_frame_timestamps(ordered, config.fps)
    prepared = [prepare_run(result, timestamps) for result in ordered]
    encoder = detect_ffmpeg()
    render_warnings: list[str] = []
    writers: dict[str, _RawH264Writer] = {}
    individual_names = {
        ("loiter", "locked"): "loiter_locked.mp4",
        ("loiter", "assist"): "loiter_assist.mp4",
        ("forward_1m", "locked"): "forward_1m_locked.mp4",
        ("forward_1m", "assist"): "forward_1m_assist.mp4",
    }
    if write_mp4 and encoder.available:
        for result in ordered:
            name = individual_names[result.key]
            writers[name] = _RawH264Writer(
                encoder,
                output_dir / name,
                (config.panel_width, config.panel_height),
                config.fps,
            )
        writers["seminar_scenario_comparison.mp4"] = _RawH264Writer(
            encoder,
            output_dir / "seminar_scenario_comparison.mp4",
            (2 * config.panel_width, 2 * config.panel_height),
            config.fps,
        )
    elif write_mp4:
        message = "FFmpeg unavailable; no MP4 files were created. GIF and PNG outputs were created."
        warnings.warn(message)
        render_warnings.append(message)
    else:
        render_warnings.append("MP4 generation was explicitly disabled.")

    gif_frames: list[Image.Image] = []
    gif_indices = _gif_indices(timestamps, config.gif_fps, config.fps)
    thumbnail_time = min(2.0, 0.5 * ordered[0].scenario.config.duration_s)
    thumbnail_index = min(len(timestamps) - 1, int(round(thumbnail_time * config.fps)))
    composite_size = (2 * config.panel_width, 2 * config.panel_height)
    try:
        for index in range(len(timestamps)):
            panels = [render_panel(run, index, config) for run in prepared]
            composite = Image.new("RGB", composite_size, (235, 238, 243))
            composite.paste(panels[0], (0, 0))
            composite.paste(panels[1], (config.panel_width, 0))
            composite.paste(panels[2], (0, config.panel_height))
            composite.paste(panels[3], (config.panel_width, config.panel_height))
            divider = ImageDraw.Draw(composite)
            divider.line(
                [(config.panel_width, 0), (config.panel_width, composite_size[1])],
                fill=(64, 72, 84),
                width=2,
            )
            divider.line(
                [(0, config.panel_height), (composite_size[0], config.panel_height)],
                fill=(64, 72, 84),
                width=2,
            )
            for result, panel in zip(ordered, panels):
                name = individual_names[result.key]
                if name in writers:
                    writers[name].write(panel)
            if "seminar_scenario_comparison.mp4" in writers:
                writers["seminar_scenario_comparison.mp4"].write(composite)
            if index in gif_indices:
                gif_frames.append(
                    composite.resize((config.gif_width, config.gif_height), Image.Resampling.LANCZOS)
                )
            if index == thumbnail_index:
                composite.save(output_dir / "seminar_video_thumbnail.png")
    finally:
        close_errors = []
        for name, writer in writers.items():
            try:
                writer.close()
            except Exception as exc:  # ensure every encoder process is reaped
                close_errors.append(f"{name}: {exc}")
        if close_errors:
            raise RuntimeError("; ".join(close_errors))

    _write_gif(gif_frames, output_dir / "seminar_scenario_comparison.gif", config.gif_fps)
    expected_names = [
        "seminar_scenario_comparison.mp4",
        "seminar_scenario_comparison.gif",
        "loiter_locked.mp4",
        "loiter_assist.mp4",
        "forward_1m_locked.mp4",
        "forward_1m_assist.mp4",
        "seminar_video_thumbnail.png",
    ]
    artifacts = [name for name in expected_names if (output_dir / name).is_file()]
    return {
        "fps": config.fps,
        "gif_fps": config.gif_fps,
        "duration_s": float(ordered[0].scenario.config.duration_s),
        "frame_count": len(timestamps),
        "panel_count": 4,
        "panel_order": [f"{scenario}/{variant}" for scenario, variant in ordered_keys],
        "world_bounds_m": {"x": list(config.x_bounds_m), "z": list(config.z_bounds_m)},
        "individual_size_px": [config.panel_width, config.panel_height],
        "composite_size_px": list(composite_size),
        "encoder": asdict(encoder),
        "mp4_created": bool(write_mp4 and encoder.available),
        "frame_timestamps": {
            "first_s": float(timestamps[0]) if timestamps.size else 0.0,
            "last_s": float(timestamps[-1]) if timestamps.size else 0.0,
            "count": len(timestamps),
        },
        "video_frame_counts": {name: len(timestamps) for name in individual_names.values()}
        | {"seminar_scenario_comparison.mp4": len(timestamps)},
        "warnings": render_warnings,
        "artifacts": artifacts,
    }
