from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .config import DroneConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DroneConfig


def save_plots(
    times: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    cfg: DroneConfig,
    results_dir: str | Path = "results",
    states_filename: str = "states.png",
    trajectory_filename: str = "trajectory.png",
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    x = states[:, 0]
    z = states[:, 1]
    theta_deg = np.rad2deg(states[:, 2])
    vx = states[:, 3]
    vz = states[:, 4]
    omega_deg = np.rad2deg(states[:, 5])
    throttle = actions[:, 0]
    ax_cmd = actions[:, 1]
    thrust = throttle * cfg.T_max

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, x, label="x")
    axes[0].plot(times, z, label="z")
    axes[0].axhline(cfg.target_z, color="tab:orange", linestyle="--", linewidth=1)
    axes[0].set_ylabel("position [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(times, theta_deg, label="theta")
    axes[1].axhline(np.rad2deg(cfg.target_theta), color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("theta [deg]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(times, vx, label="vx")
    axes[2].plot(times, vz, label="vz")
    axes[2].plot(times, omega_deg, label="omega [deg/s]")
    axes[2].set_ylabel("velocity")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    axes[3].plot(times, thrust, color="tab:red", linewidth=2, label="thrust [N]")
    axes[3].axhline(cfg.m * cfg.g, color="black", linestyle="--", linewidth=1, label="hover thrust")
    axes[3].set_ylabel("thrust [N]")
    thrust_margin = max(0.5, 1.2 * np.max(np.abs(thrust - cfg.m * cfg.g)))
    axes[3].set_ylim(cfg.m * cfg.g - thrust_margin, cfg.m * cfg.g + thrust_margin)
    axes[3].legend(loc="upper left")
    axes[3].grid(True)

    throttle_axis = axes[3].twinx()
    throttle_axis.plot(times, throttle, color="tab:blue", alpha=0.6, label="throttle")
    throttle_axis.set_ylabel("throttle")
    throttle_axis.set_ylim(0.0, 1.0)
    throttle_axis.legend(loc="upper right")

    axes[4].plot(times, ax_cmd, label="ax_cmd [m/s^2]")
    axes[4].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("base accel")
    axes[4].legend(loc="best")
    axes[4].grid(True)

    fig.tight_layout()
    fig.savefig(results_dir / states_filename, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, z, label="thrust point")
    ax.scatter([x[0]], [z[0]], color="tab:green", label="start")
    ax.scatter([x[-1]], [z[-1]], color="tab:red", label="end")
    ax.scatter([cfg.target_x], [cfg.target_z], color="black", marker="x", label="target")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Moving base trajectory")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / trajectory_filename, dpi=160)
    plt.close(fig)
