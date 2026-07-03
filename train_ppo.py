from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

try:
    from .env import InvertedDroneHoverEnv
except ImportError:  # pragma: no cover - supports direct script execution
    from env import InvertedDroneHoverEnv


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = InvertedDroneHoverEnv()
    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(results_dir / "tb_logs"),
    )
    model.learn(total_timesteps=200_000)
    model.save(results_dir / "ppo_inverted_drone")


if __name__ == "__main__":
    main()
