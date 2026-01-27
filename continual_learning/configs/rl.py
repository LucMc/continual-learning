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
class QNetworkConfig(struct.PyTreeNode):
    """Configuration for Q-network (critic) in SAC."""

    optimizer: OptimizerConfig
    network: NetworkConfigType

    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]


@struct.dataclass(frozen=True)
class SACConfig(struct.PyTreeNode):
    """Configuration for SAC/BRO algorithm."""

    actor_config: PolicyNetworkConfig
    critic_config: QNetworkConfig

    # SAC hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Soft target update coefficient
    alpha: float = 1.0  # Initial entropy coefficient
    alpha_lr: float = 3e-4  # Learning rate for entropy coefficient
    auto_entropy: bool = True  # Whether to auto-tune entropy coefficient
    target_entropy: float | None = None  # Target entropy (None = -action_dim)

    # Replay buffer settings
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 10_000  # Steps before training begins

    # BRO-specific settings
    replay_ratio: int = 1  # Gradient updates per environment step (BRO uses 4-16)
    reset_interval: int | None = None  # Steps between network resets (None = no resets)

    # Architecture settings
    use_layer_norm: bool = True  # Critical for stability with high replay ratios


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
