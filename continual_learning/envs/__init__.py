from continual_learning.configs.envs import EnvConfig

from .base import (
    ContinualLearningEnv,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    VectorEnv,
)
from .slippery_ant import ContinualAnt


def get_benchmark(
    seed: int,
    env_config: EnvConfig,
) -> ContinualLearningEnv | JittableContinualLearningEnv:
    if env_config.name == "slippery_ant":
        return ContinualAnt(seed, env_config)
    else:
        raise ValueError(f"Unknown environment: {env_config.name}")


__all__ = [
    "ContinualLearningEnv",
    "JittableVectorEnv",
    "VectorEnv",
    "JittableContinualLearningEnv",
    "ContinualAnt",
]
