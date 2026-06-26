# -*- coding: utf-8 -*-
# 타입 힌트 처리를 유연하게 하기 위한 옵션입니다.
from __future__ import annotations

# 결과 폴더 경로를 만들기 위해 Path를 사용합니다.
from pathlib import Path

# Stable-Baselines3의 PPO 알고리즘입니다.
from stable_baselines3 import PPO
# 환경이 Gymnasium API를 제대로 따르는지 확인하는 helper입니다.
from stable_baselines3.common.env_checker import check_env

# 패키지 내부 import를 먼저 시도합니다.
try:
    # 학습에 사용할 드론 hover 환경입니다.
    from .env import InvertedDroneHoverEnv
# 직접 실행 시에는 상대 import가 실패할 수 있습니다.
except ImportError:  # pragma: no cover - supports direct script execution
    # 직접 실행용 환경 import입니다.
    from env import InvertedDroneHoverEnv


# PPO 학습을 실행하는 진입점입니다.
def main() -> None:
    # 현재 파일 위치 기준으로 results 폴더를 잡습니다.
    results_dir = Path(__file__).resolve().parent / "results"
    # 결과 폴더가 없으면 생성합니다.
    results_dir.mkdir(parents=True, exist_ok=True)

    # Gymnasium 환경 인스턴스를 만듭니다.
    env = InvertedDroneHoverEnv()
    # Stable-Baselines3 환경 checker로 API 호환성을 확인합니다.
    check_env(env)

    # PPO 모델을 MLP 정책으로 생성합니다.
    model = PPO(
        # 관측값이 벡터라서 기본 MLP policy를 사용합니다.
        "MlpPolicy",
        # 학습할 환경입니다.
        env,
        # 학습 로그를 콘솔에 표시합니다.
        verbose=1,
        # TensorBoard 로그 저장 위치입니다.
        tensorboard_log=str(results_dir / "tb_logs"),
    )
    # 지정한 timestep만큼 PPO 학습을 진행합니다.
    model.learn(total_timesteps=200_000)
    # 학습된 모델을 results 폴더에 저장합니다.
    model.save(results_dir / "ppo_inverted_drone")


# 이 파일을 직접 실행했을 때만 main을 호출합니다.
if __name__ == "__main__":
    # 실제 학습 실행 진입점입니다.
    main()
