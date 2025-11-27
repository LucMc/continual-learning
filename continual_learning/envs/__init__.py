from continual_learning.configs.envs import EnvConfig

from .base import (
    ContinualLearningEnv,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    VectorEnv,
)
from .slippery_mujoco import ContinualAnt, ContinualCheetah, ContinualHumanoid, HumanoidStand


def get_benchmark(
    seed: int,
    env_config: EnvConfig,
) -> ContinualLearningEnv | JittableContinualLearningEnv:
    match env_config.name:
        case "slippery_ant":
            return ContinualAnt(seed, env_config)
        case "slippery_humanoid":
            return ContinualHumanoid(seed, env_config)
        case "slippery_cheetah":
            return ContinualCheetah(seed, env_config)
        case "humanoid_stand":
            return HumanoidStand(seed, env_config)

    raise ValueError(f"Unknown environment: {env_config.name}")


__all__ = [
    "ContinualLearningEnv",
    "JittableVectorEnv",
    "VectorEnv",
    "JittableContinualLearningEnv",
    "ContinualAnt",
    "ContinualHumanoid",
    "ContinualCheetah",
    "HumanoidStand",
]
