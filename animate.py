from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Polygon

try:
    from .config import DroneConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DroneConfig


def body_geometry(state: np.ndarray, cfg: DroneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, z, theta, *_ = state
    thrust_point = np.array([x, z], dtype=float)
    u = np.array([np.sin(theta), np.cos(theta)], dtype=float)
    v = np.array([np.cos(theta), -np.sin(theta)], dtype=float)
    cg = thrust_point + cfg.l * u

    bottom_center = thrust_point
    top_center = thrust_point + cfg.H * u
    half_width = cfg.W / 2
    corners = np.array(
        [
            bottom_center - half_width * v,
            bottom_center + half_width * v,
            top_center + half_width * v,
            top_center - half_width * v,
        ]
    )
    return thrust_point, cg, corners


def save_animation(
    times: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    cfg: DroneConfig,
    results_dir: str | Path = "results",
    filename: str = "pid_animation.gif",
    fps: int = 30,
    stride: int | None = None,
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / filename

    if stride is None:
        stride = max(1, int(round(1.0 / (fps * cfg.dt))))

    frame_indices = np.arange(0, len(states), stride)
    frame_states = states[frame_indices]
    frame_actions = actions[frame_indices]

    cg_x = states[:, 0] + cfg.l * np.sin(states[:, 2])
    cg_z = states[:, 1] + cfg.l * np.cos(states[:, 2])
    x_all = np.concatenate([states[:, 0], cg_x])
    z_all = np.concatenate([states[:, 1], cg_z])

    x_min = min(np.min(x_all) - 0.8, cfg.target_x - 1.0)
    x_max = max(np.max(x_all) + 0.8, cfg.target_x + 1.0)
    z_min = max(0.0, min(np.min(z_all) - 0.4, cfg.target_z - 0.8))
    z_max = max(np.max(z_all) + 0.5, cfg.target_z + cfg.H + 0.8)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True)

    body = Polygon(np.zeros((4, 2)), closed=True, fc="tab:blue", ec="black", alpha=0.65)
    ax.add_patch(body)
    cg_dot, = ax.plot([], [], "o", color="tab:red", markersize=6, label="CG")
    base_dot, = ax.plot([], [], "o", color="black", markersize=5, label="thrust point")
    target_dot, = ax.plot([cfg.target_x], [cfg.target_z], "x", color="tab:green", markersize=8, label="target")
    base_path, = ax.plot([], [], color="tab:gray", linewidth=1, alpha=0.8, label="base path")
    thrust_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="->", mutation_scale=14, color="tab:orange")
    ax_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="->", mutation_scale=14, color="tab:purple")
    ax.add_patch(thrust_arrow)
    ax.add_patch(ax_arrow)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right")

    def update(frame: int):
        state = frame_states[frame]
        throttle, ax_cmd = frame_actions[frame]
        thrust_point, cg, corners = body_geometry(state, cfg)
        thrust = np.clip(throttle, 0.0, 1.0) * cfg.T_max

        body.set_xy(corners)
        cg_dot.set_data([cg[0]], [cg[1]])
        base_dot.set_data([thrust_point[0]], [thrust_point[1]])
        base_path.set_data(states[: frame_indices[frame] + 1, 0], states[: frame_indices[frame] + 1, 1])

        thrust_len = 0.35 * thrust / cfg.T_max
        thrust_arrow.set_positions(thrust_point, thrust_point + np.array([0.0, thrust_len]))

        ax_len = 0.35 * ax_cmd / cfg.ax_cmd_max
        ax_arrow.set_positions(thrust_point, thrust_point + np.array([ax_len, 0.0]))

        time_text.set_text(
            f"t = {times[frame_indices[frame]]:.2f} s\n"
            f"thrust = {thrust:.1f} N\n"
            f"ax_cmd = {ax_cmd:.2f} m/s^2"
        )
        return body, cg_dot, base_dot, target_dot, base_path, thrust_arrow, ax_arrow, time_text

    ani = FuncAnimation(fig, update, frames=len(frame_states), interval=1000 / fps, blit=True)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path
