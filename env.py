from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Gymnasium is required for env.py. Install it with "
        "`python -m pip install gymnasium` or `pip install -r requirements.txt`."
    ) from exc

try:
    from .animate import body_geometry
    from .config import DroneConfig
    from .drone_model import InvertedDrone2D
except ImportError:  # pragma: no cover - supports direct script execution
    from animate import body_geometry
    from config import DroneConfig
    from drone_model import InvertedDrone2D


class InvertedDroneHoverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, cfg: DroneConfig | None = None):
        super().__init__()
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.cfg = cfg or DroneConfig()
        self.drone = InvertedDrone2D(self.cfg)
        self.render_mode = render_mode
        self.elapsed_steps = 0
        self.max_episode_steps = int(self.cfg.max_time / self.cfg.dt)

        self.observation_space = spaces.Box(
            low=np.array([-5, 0, -np.pi, -10, -10, -20], dtype=np.float32),
            high=np.array([5, 5, np.pi, 10, 10, 20], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.cfg.ax_cmd_max], dtype=np.float32),
            high=np.array([1.0, self.cfg.ax_cmd_max], dtype=np.float32),
            dtype=np.float32,
        )

        self._fig = None
        self._ax = None

    def _get_obs(self) -> np.ndarray:
        return self.drone.state.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.elapsed_steps = 0

        if options and "state" in options:
            state = np.asarray(options["state"], dtype=float)
        else:
            theta0 = self.np_random.uniform(np.deg2rad(-12), np.deg2rad(12))
            state = np.array([0.0, self.cfg.target_z, theta0, 0.0, 0.0, 0.0], dtype=float)

        self.drone.reset(state)
        info = {"time": 0.0, "thrust": 0.0, "ax_cmd": 0.0}

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        throttle = float(np.clip(action[0], 0.0, 1.0))
        ax_cmd = float(np.clip(action[1], -self.cfg.ax_cmd_max, self.cfg.ax_cmd_max))

        state = self.drone.step(np.array([throttle, ax_cmd], dtype=float))
        self.elapsed_steps += 1

        x, z, theta, vx, vz, omega = state
        ax_norm = ax_cmd / self.cfg.ax_cmd_max
        reward = (
            -2.0 * (z - self.cfg.target_z) ** 2
            -1.0 * (x - self.cfg.target_x) ** 2
            -3.0 * theta**2
            -0.1 * vx**2
            -0.1 * vz**2
            -0.05 * omega**2
            -0.02 * ax_norm**2
        )
        reward += 0.1

        terminated = bool(
            z < 0.0
            or abs(theta) > np.deg2rad(80)
            or abs(x) > 5.0
            or z > 5.0
        )
        truncated = self.elapsed_steps >= self.max_episode_steps
        info = {
            "time": self.elapsed_steps * self.cfg.dt,
            "thrust": float(self.drone.last_thrust),
            "ax_cmd": float(self.drone.last_ax_cmd),
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Polygon

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 5))
            if self.render_mode == "human":
                plt.ion()

        ax = self._ax
        ax.clear()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")

        thrust_point, cg, corners = body_geometry(self.drone.state, self.cfg)
        ax.add_patch(Polygon(corners, closed=True, fc="tab:blue", ec="black", alpha=0.65))
        ax.plot(cg[0], cg[1], "o", color="tab:red", label="CG")
        ax.plot(thrust_point[0], thrust_point[1], "o", color="black", label="thrust point")
        ax.plot(self.cfg.target_x, self.cfg.target_z, "x", color="tab:green", markersize=8)

        thrust_len = 0.35 * self.drone.last_thrust / self.cfg.T_max if self.cfg.T_max > 0 else 0.0
        ax.add_patch(
            FancyArrowPatch(
                thrust_point,
                thrust_point + np.array([0.0, thrust_len]),
                arrowstyle="->",
                mutation_scale=14,
                color="tab:orange",
            )
        )
        ax_len = 0.35 * self.drone.last_ax_cmd / self.cfg.ax_cmd_max
        ax.add_patch(
            FancyArrowPatch(
                thrust_point,
                thrust_point + np.array([ax_len, 0.0]),
                arrowstyle="->",
                mutation_scale=14,
                color="tab:purple",
            )
        )

        if self.render_mode == "human":
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
            return None

        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
