from dataclasses import dataclass

import jax

from continual_learning_2.configs.models import CNNConfig, MLPConfig, ResNetConfig
from continual_learning_2.types import StdType

from .optim import OptimizerConfig

NetworkConfigType = MLPConfig | CNNConfig | ResNetConfig


@dataclass(frozen=True)
class NetworkConfig:
    optimizer: OptimizerConfig
    network: NetworkConfigType

    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]


@dataclass(frozen=True)
class PolicyNetworkConfig:
    optimizer: OptimizerConfig
    network: NetworkConfigType
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    std_type: StdType = StdType.PARAM


@dataclass(frozen=True)
class ValueFunctionConfig:
    optimizer: OptimizerConfig
    network: NetworkConfigType


@dataclass(frozen=True)
class PPOConfig:
    policy_config: PolicyNetworkConfig
    vf_config: ValueFunctionConfig
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
