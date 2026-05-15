from continual_learning.configs.envs import EnvConfig

from .base import (
    ContinualLearningEnv,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    VectorEnv,
)
from .slippery_mujoco import ContinualAnt, ContinualHumanoid


def get_benchmark(
    seed: int,
    env_config: EnvConfig,
) -> ContinualLearningEnv | JittableContinualLearningEnv:
    if env_config.name == "slippery_ant":
        return ContinualAnt(seed, env_config)
    if env_config.name == "slippery_humanoid":
        return ContinualHumanoid(seed, env_config)
    if env_config.name == "metaworld_mt10":
        from .metaworld import MetaWorldMT10Benchmark

        return MetaWorldMT10Benchmark(seed, env_config)
    if env_config.name == "minatar":
        from .minatar import MinatarContinualEnv

        return MinatarContinualEnv(seed, env_config)
    else:
        raise ValueError(f"Unknown environment: {env_config.name}")


__all__ = [
    "ContinualLearningEnv",
    "JittableVectorEnv",
    "VectorEnv",
    "JittableContinualLearningEnv",
    "ContinualAnt",
    "ContinualHumanoid",
]
