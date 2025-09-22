import jax
from flax import struct

from continual_learning.configs.models import CNNConfig, MLPConfig, ResNetConfig
from continual_learning.types import StdType

from .optim import OptimizerConfig

NetworkConfigType = MLPConfig | CNNConfig | ResNetConfig


@struct.dataclass(frozen=True)
class NetworkConfig(struct.PyTreeNode):
    optimizer: OptimizerConfig
    network: NetworkConfigType

    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]


@struct.dataclass(frozen=True)
class PolicyNetworkConfig(struct.PyTreeNode):
    optimizer: OptimizerConfig
    network: NetworkConfigType
    min_std: float = 1e-3
    var_scale: float = 1.0
    std_type: StdType = StdType.PARAM


@struct.dataclass(frozen=True)
class ValueFunctionConfig(struct.PyTreeNode):
    optimizer: OptimizerConfig
    network: NetworkConfigType


@struct.dataclass(frozen=True)
class PPOConfig(struct.PyTreeNode):
    policy_config: PolicyNetworkConfig
    vf_config: ValueFunctionConfig
    clip_eps: float = 0.3
    entropy_coefficient: float = 1e-3
    vf_coefficient: float = 0.5
    normalize_advantages: bool = True
    normalize_observations: bool = False
    gamma: float = 0.95
    gae_lambda: float = 0.97
    num_gradient_steps: int = 32
    num_epochs: int = 16
    num_rollout_steps: int = 100_000

    reset_normalizer_on_task_change: bool = False
    """Flag to reset the observation normalisation params on task change.
    This should be False to make the agent "continual-learning-agnostic",
    and is only added as a debug flag if things go wrong.
    """
