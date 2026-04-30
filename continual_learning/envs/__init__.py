from continual_learning.configs.envs import EnvConfig

from .base import (
    ContinualLearningEnv,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    VectorEnv,
)
from .slippery_mujoco import ContinualAnt, ContinualDelayedAnt, ContinualHumanoid


def get_benchmark(
    seed: int,
    env_config: EnvConfig,
) -> ContinualLearningEnv | JittableContinualLearningEnv:
    if env_config.name == "slippery_ant":
        return ContinualAnt(seed, env_config)
    if env_config.name == "slippery_humanoid":
        return ContinualHumanoid(seed, env_config)
    if env_config.name == "delayed_ant":
        # delayed_ant requires extra delay-specific parameters, so the
        # experiment script must construct ContinualDelayedAnt directly and
        # pass it via JittedContinualPPOTrainer's `benchmark` parameter.
        raise ValueError(
            "delayed_ant cannot be constructed via get_benchmark; "
            "instantiate ContinualDelayedAnt directly and pass it to the "
            "trainer via the `benchmark` argument."
        )
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
    "ContinualDelayedAnt",
    "ContinualHumanoid",
]
