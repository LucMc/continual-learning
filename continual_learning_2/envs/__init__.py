from continual_learning_2.configs.envs import EnvConfig

from .base import ContinualLearningEnv, JittableVectorEnv, VectorEnv


def get_benchmark(env_config: EnvConfig) -> ContinualLearningEnv:
    # TODO
    ...


__all__ = [
    "ContinualLearningEnv",
    "JittableVectorEnv",
    "VectorEnv",
]
