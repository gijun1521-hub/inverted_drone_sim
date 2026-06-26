from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .config import DroneConfig
    from .drone_model import InvertedDrone2D
    from .plots import save_plots
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DroneConfig
    from drone_model import InvertedDrone2D
    from plots import save_plots


def main() -> None:
    cfg = DroneConfig(max_time=3.0)
    drone = InvertedDrone2D(cfg)
    state = drone.reset()

    times = []
    states = []
    actions = []

    for i in range(int(cfg.max_time / cfg.dt)):
        t = i * cfg.dt
        action = np.array([cfg.hover_throttle, 0.0], dtype=float)

        times.append(t)
        states.append(state.copy())
        actions.append(action.copy())

        state = drone.step(action)
        if state[1] < 0.0 or abs(state[2]) > np.deg2rad(85):
            break

    results_dir = Path(__file__).resolve().parent / "results"
    save_plots(
        np.asarray(times),
        np.asarray(states),
        np.asarray(actions),
        cfg,
        results_dir,
        states_filename="passive_states.png",
        trajectory_filename="passive_trajectory.png",
    )

    print("Passive moving-base simulation complete")
    print(f"steps: {len(times)}")
    print(f"final theta: {np.rad2deg(states[-1][2]): .3f} deg")
    print(f"saved: {results_dir / 'passive_states.png'}")
    print(f"saved: {results_dir / 'passive_trajectory.png'}")


if __name__ == "__main__":
    main()
