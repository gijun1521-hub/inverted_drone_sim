from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

try:
    from .config import RigidBodyConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import RigidBodyConfig


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV contains no rows")

    data: dict[str, list[float] | list[str]] = {}
    for key in rows[0]:
        data[key] = []
    for row in rows:
        for key, value in row.items():
            try:
                data[key].append(float(value))
            except ValueError:
                data[key].append(value)
    return {key: np.asarray(value) for key, value in data.items()}


def save_state_plots(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    t = data["sim_time"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, data["x_cg"], label="x_cg")
    axes[0].plot(t, data["z_cg"], label="z_cg")
    if "target_x" in data:
        axes[0].plot(t, data["target_x"], "--", label="target_x")
    if "target_z" in data:
        axes[0].plot(t, data["target_z"], "--", label="target_z")
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, np.rad2deg(data["theta"]), label="theta")
    axes[1].plot(t, np.rad2deg(data["theta_target"]), label="theta target")
    axes[1].set_ylabel("angle [deg]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, data["vx"], label="vx")
    axes[2].plot(t, data["vz"], label="vz")
    axes[2].plot(t, np.rad2deg(data["omega"]), label="omega [deg/s]")
    axes[2].set_ylabel("velocity")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, data["thrust"], label="thrust")
    axes[3].plot(t, np.rad2deg(data["vane_angle"]), label="vane [deg]")
    if "vane_angle_cmd" in data:
        axes[3].plot(t, np.rad2deg(data["vane_angle_cmd"]), "--", label="vane cmd [deg]")
    if "mixer_saturated" in data:
        axes[3].plot(t, data["mixer_saturated"] * max(np.max(data["thrust"]), 1.0), ":", label="mixer saturated")
    axes[3].set_xlabel("time [s]")
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    path = output_dir / "interactive_replay_states.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_force_plots(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    t = data["sim_time"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, data["thrust_force_x"], label="thrust Fx")
    axes[0].plot(t, data["thrust_force_z"], label="thrust Fz")
    axes[0].plot(t, data["vane_force_x"], label="vane Fx")
    axes[0].plot(t, data["vane_force_z"], label="vane Fz")
    axes[0].set_ylabel("force [N]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, data["disturbance_force_x"], label="dist Fx")
    axes[1].plot(t, data["disturbance_force_z"], label="dist Fz")
    axes[1].plot(t, data["disturbance_moment"], label="dist M")
    axes[1].set_ylabel("disturbance")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, data["desired_moment"], label="desired")
    axes[2].plot(t, data["achievable_moment"], label="achievable")
    axes[2].plot(t, data["total_moment"], label="actual total")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("moment [N m]")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    path = output_dir / "interactive_replay_forces.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_controller_plots(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    t = data["sim_time"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, data["rate_error"], label="rate error")
    axes[0].plot(t, data["omega_target"], label="omega target")
    axes[0].set_ylabel("rate [rad/s]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, data["rate_p"], label="P")
    axes[1].plot(t, data["rate_i"], label="I")
    axes[1].plot(t, data["rate_d"], label="D")
    axes[1].plot(t, data["rate_ff"], label="FF")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("moment terms")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    path = output_dir / "interactive_replay_controller.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_animation(data: dict[str, np.ndarray], output_dir: Path, fps: int = 30) -> Path:
    cfg = RigidBodyConfig()
    t = data["sim_time"]
    stride = max(1, int(round(len(t) / max(1, t[-1] * fps))))
    idx = np.arange(0, len(t), stride)

    x = data["x_cg"]
    z = data["z_cg"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(x)) - 1.0, float(np.max(x)) + 1.0)
    ax.set_ylim(max(0.0, float(np.min(z)) - 0.5), float(np.max(z)) + 1.0)
    ax.grid(True)
    body = Polygon(np.zeros((4, 2)), closed=True, fc="tab:blue", ec="black", alpha=0.7)
    ax.add_patch(body)
    trace, = ax.plot([], [], color="tab:gray", lw=1)
    cg_dot, = ax.plot([], [], "o", color="tab:red")
    target_dot, = ax.plot([], [], "x", color="tab:orange")
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

    def update(frame_i: int):
        i = idx[frame_i]
        theta = data["theta"][i]
        cg = np.array([data["x_cg"][i], data["z_cg"][i]])
        body_up = np.array([np.sin(theta), np.cos(theta)])
        body_right = np.array([np.cos(theta), -np.sin(theta)])
        top = cg + cfg.l * body_up
        bottom = cg - cfg.l * body_up
        hw = 0.5 * cfg.W
        corners = np.array([bottom - hw * body_right, bottom + hw * body_right, top + hw * body_right, top - hw * body_right])
        body.set_xy(corners)
        trace.set_data(x[: i + 1], z[: i + 1])
        cg_dot.set_data([cg[0]], [cg[1]])
        if "target_x" in data and "target_z" in data:
            target_dot.set_data([data["target_x"][i]], [data["target_z"][i]])
        time_text.set_text(f"t={t[i]:.2f}s")
        return body, trace, cg_dot, target_dot, time_text

    ani = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=True)
    path = output_dir / "interactive_replay.gif"
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an interactive simulator CSV.")
    parser.add_argument("csv", help="Path to an interactive CSV log.")
    parser.add_argument("--out", default="results/replay", help="Output directory.")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF generation.")
    args = parser.parse_args()

    data = load_csv(args.csv)
    output_dir = Path(args.out)
    outputs = [
        save_state_plots(data, output_dir),
        save_force_plots(data, output_dir),
        save_controller_plots(data, output_dir),
    ]
    if not args.no_animation:
        outputs.append(save_animation(data, output_dir))

    for path in outputs:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
