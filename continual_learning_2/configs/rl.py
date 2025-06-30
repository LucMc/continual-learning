from dataclasses import dataclass

import jax

from .optim import OptimizerConfig


@dataclass(frozen=True)
class NetworkConfig:
    optim_config: OptimizerConfig

    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]


@dataclass(frozen=True)
class PPOConfig:
    policy_config: NetworkConfig
    vf_config: NetworkConfig
    clip_eps: float = 0.2
    clip_vf_loss: bool = True
    entropy_coefficient: float = 5e-3
    vf_coefficient: float = 0.001
    normalize_advantages: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.97
    num_gradient_steps: int = 32
    num_epochs: int = 16
    num_rollout_steps: int = 100_000
    target_kl: float | None = None
