from continual_learning_2.configs.envs import EnvConfig

from .base import (
    ContinualLearningEnv,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    VectorEnv,
)


def get_benchmark(
    env_config: EnvConfig,
) -> ContinualLearningEnv | JittableContinualLearningEnv:
    # TODO
    ...


__all__ = [
    "ContinualLearningEnv",
    "JittableVectorEnv",
    "VectorEnv",
    "JittableContinualLearningEnv",
]
