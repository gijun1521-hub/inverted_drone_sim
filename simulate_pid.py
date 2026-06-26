from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .animate import save_animation
    from .config import DroneConfig
    from .drone_model import InvertedDrone2D
    from .pid_controller import PIDController
    from .plots import save_plots
except ImportError:  # pragma: no cover - supports direct script execution
    from animate import save_animation
    from config import DroneConfig
    from drone_model import InvertedDrone2D
    from pid_controller import PIDController
    from plots import save_plots


def run_simulation(cfg: DroneConfig | None = None):
    cfg = cfg or DroneConfig()
    drone = InvertedDrone2D(cfg)
    controller = PIDController(cfg)

    state = drone.reset()
    controller.reset()

    times = []
    states = []
    actions = []

    for i in range(int(cfg.max_time / cfg.dt)):
        t = i * cfg.dt
        action = controller.compute_action(state)
        clipped_action = drone.clamp_action(action)

        times.append(t)
        states.append(state.copy())
        actions.append(clipped_action.copy())

        state = drone.step(action)
        if state[1] < 0.0:
            print(f"Terminated: thrust point crossed ground at t={t:.2f} s")
            break

    return np.asarray(times), np.asarray(states), np.asarray(actions), cfg


def main() -> None:
    times, states, actions, cfg = run_simulation()
    results_dir = Path(__file__).resolve().parent / "results"
    save_plots(times, states, actions, cfg, results_dir)
    animation_path = save_animation(times, states, actions, cfg, results_dir)

    final = states[-1]
    thrust = actions[:, 0] * cfg.T_max
    ax_cmd = actions[:, 1]

    print("Moving-base PID simulation complete")
    print(f"steps: {len(times)}")
    print(f"final x: {final[0]: .3f} m")
    print(f"final z: {final[1]: .3f} m")
    print(f"final theta: {np.rad2deg(final[2]): .3f} deg")
    print(f"hover thrust: {cfg.m * cfg.g: .3f} N")
    print(f"min thrust: {np.min(thrust): .3f} N")
    print(f"mean thrust: {np.mean(thrust): .3f} N")
    print(f"max thrust: {np.max(thrust): .3f} N")
    print(f"max |ax_cmd|: {np.max(np.abs(ax_cmd)): .3f} m/s^2")
    print(f"saved: {results_dir / 'states.png'}")
    print(f"saved: {results_dir / 'trajectory.png'}")
    print(f"saved: {animation_path}")


if __name__ == "__main__":
    main()
