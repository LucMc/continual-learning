import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig
from continual_learning.types import StdType

tfd = tfp.distributions
tfb = tfp.bijectors

# BRO constants
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def orthogonal_init(scale: float = jnp.sqrt(2)):
    """Orthogonal initialization as used in BRO."""
    return nn.initializers.orthogonal(scale)


class BroNet(nn.Module):
    """BRO network architecture with LayerNorm and residual connections.

    This is the key architectural component of BRO that enables stable training
    with high replay ratios. Features:
    - LayerNorm after every Dense layer
    - Residual connections
    - Orthogonal initialization
    - Activation collection for reset methods (via sow)

    Note: Dense layers are named explicitly (Dense_0, Dense_1, etc.) and
    activation sow() names match these to enable reset methods.
    """
    hidden_dims: int = 256
    depth: int = 1
    activation: str = "relu"
    add_final_layer: bool = False
    output_nodes: int = 1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh
        dense_idx = 0  # Track Dense layer index for naming

        if self.depth == 1:
            # Initial layer (Dense_0)
            x = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", x)
            x = nn.LayerNorm()(x)
            x = activation_fn(x)
            self.sow("activations", f"Dense_{dense_idx}_act", x)
            dense_idx += 1

            # Residual block (Dense_1, Dense_2)
            res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
            res = nn.LayerNorm()(res)
            res = activation_fn(res)
            self.sow("activations", f"Dense_{dense_idx}_act", res)
            dense_idx += 1

            res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(res)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
            res = nn.LayerNorm()(res)
            self.sow("activations", f"Dense_{dense_idx}_act", res)  # Sow after LayerNorm for reset methods
            x = res + x  # Residual connection
            dense_idx += 1

            if self.add_final_layer:
                # Output layer
                self.sow("preactivations", "output_pre", x)
                x = nn.Dense(self.output_nodes, kernel_init=orthogonal_init(), name="output")(x)
                self.sow("activations", "output_act", x)
            return x

        elif self.depth == 2:
            # Initial layer (Dense_0)
            x = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", x)
            x = nn.LayerNorm()(x)
            x = activation_fn(x)
            self.sow("activations", f"Dense_{dense_idx}_act", x)
            dense_idx += 1

            # Residual block 1 (Dense_1, Dense_2)
            res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
            res = nn.LayerNorm()(res)
            res = activation_fn(res)
            self.sow("activations", f"Dense_{dense_idx}_act", res)
            dense_idx += 1

            res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(res)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
            res = nn.LayerNorm()(res)
            self.sow("activations", f"Dense_{dense_idx}_act", res)  # Sow after LayerNorm for reset methods
            x = res + x
            dense_idx += 1

            # Residual block 2 (Dense_3, Dense_4)
            res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
            res = nn.LayerNorm()(res)
            res = activation_fn(res)
            self.sow("activations", f"Dense_{dense_idx}_act", res)
            dense_idx += 1

            res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(res)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
            res = nn.LayerNorm()(res)
            self.sow("activations", f"Dense_{dense_idx}_act", res)  # Sow after LayerNorm for reset methods
            x = res + x
            dense_idx += 1

            if self.add_final_layer:
                self.sow("preactivations", "output_pre", x)
                x = nn.Dense(self.output_nodes, kernel_init=orthogonal_init(), name="output")(x)
                self.sow("activations", "output_act", x)
            return x

        else:  # depth == 3
            # Initial layer (Dense_0)
            x = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
            self.sow("preactivations", f"Dense_{dense_idx}_pre", x)
            x = nn.LayerNorm()(x)
            x = activation_fn(x)
            self.sow("activations", f"Dense_{dense_idx}_act", x)
            dense_idx += 1

            # Residual blocks
            for _ in range(3):
                res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(x)
                self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
                res = nn.LayerNorm()(res)
                res = activation_fn(res)
                self.sow("activations", f"Dense_{dense_idx}_act", res)
                dense_idx += 1

                res = nn.Dense(self.hidden_dims, kernel_init=orthogonal_init(), name=f"Dense_{dense_idx}")(res)
                self.sow("preactivations", f"Dense_{dense_idx}_pre", res)
                res = nn.LayerNorm()(res)
                self.sow("activations", f"Dense_{dense_idx}_act", res)  # Sow after LayerNorm for reset methods
                x = res + x
                dense_idx += 1

            if self.add_final_layer:
                self.sow("preactivations", "output_pre", x)
                x = nn.Dense(self.output_nodes, kernel_init=orthogonal_init(), name="output")(x)
                self.sow("activations", "output_act", x)
            return x


class BRONormalTanhPolicy(nn.Module):
    """BRO conservative actor with tanh squashing.

    Uses BroNet backbone and outputs a tanh-squashed normal distribution.
    This is the "conservative" policy in BRO terminology.
    """
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    log_std_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        temperature: float = 1.0,
        training: bool = False,
        return_params: bool = False
    ):
        outputs = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            add_final_layer=False
        )(observations, training=training)

        # Use unique names to avoid collision with BroNet's internal Dense layers
        means = nn.Dense(self.action_dim, kernel_init=orthogonal_init(), name="mean_output")(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=orthogonal_init(self.log_std_scale), name="logstd_output")(outputs)

        # Squash log_stds to valid range using tanh (as in official implementation)
        log_stds = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * 0.5 * (1 + nn.tanh(log_stds))
        stds = jnp.exp(log_stds) * temperature

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=stds)

        if not return_params:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), means, stds


class BRODualTanhPolicy(nn.Module):
    """BRO optimistic actor (dual policy).

    Takes the conservative actor's mean/std as input and outputs a shifted policy.
    This enables optimistic exploration while staying close to the conservative policy.
    """
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    scale_means: float = 0.01

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        means: jax.Array,
        stds: jax.Array,
        std_multiplier: float,
        training: bool = False,
        return_params: bool = True
    ):
        # Concatenate observations with conservative means
        inputs = jnp.concatenate([observations, means], axis=-1)

        outputs = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            add_final_layer=False
        )(inputs, training=training)

        # Predict action shift (small scale initialization)
        # Use unique name to avoid collision with BroNet's internal Dense layers
        action_shift = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal_init(scale=self.scale_means),
            use_bias=False,
            name="action_shift_output"
        )(outputs)

        optimistic_means = means + action_shift
        optimistic_stds = stds * std_multiplier

        base_dist = tfd.MultivariateNormalDiag(loc=optimistic_means, scale_diag=optimistic_stds)

        if not return_params:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), optimistic_means, optimistic_stds


class BRODistributionalCritic(nn.Module):
    """BRO distributional critic using quantile regression.

    Outputs n_quantiles values for each Q-network instead of a single value.
    This provides uncertainty estimates used for pessimism/optimism.
    """
    n_quantiles: int = 100
    hidden_dims: int = 256
    depth: int = 1

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([observations, actions], axis=-1)

        # Q1 network
        q1 = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            add_final_layer=True,
            output_nodes=self.n_quantiles
        )(x, training=training)

        # Q2 network (separate)
        q2 = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            add_final_layer=True,
            output_nodes=self.n_quantiles
        )(x, training=training)

        return q1, q2


class Temperature(nn.Module):
    """Learnable temperature (entropy coefficient) for SAC."""
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jax.Array:
        log_temp = self.param(
            'log_temp',
            lambda key: jnp.full((), jnp.log(self.initial_temperature))
        )
        return jnp.exp(log_temp)


class Adjustment(nn.Module):
    """Learnable adjustment coefficient (optimism/regularizer) with bounded output.

    Uses tanh to bound the log value, then exponentiates.
    """
    init_value: float = 1.0
    log_val_min: float = -10.0
    log_val_max: float = 7.5

    @nn.compact
    def __call__(self) -> jax.Array:
        log_value = self.param(
            'log_value',
            lambda key: jnp.full((), jnp.log(self.init_value))
        )
        # Bound using tanh
        log_value = self.log_val_min + (self.log_val_max - self.log_val_min) * 0.5 * (1 + nn.tanh(log_value))
        return jnp.exp(log_value)


class Policy(nn.Module):
    network: type[nn.Module]
    config: PolicyNetworkConfig

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> distrax.Distribution:
        network_config = self.config.network
        if self.config.std_type == StdType.MLP_HEAD:
            network_config = network_config.replace(output_size=network_config.output_size * 2)

        x = self.network(config=network_config, name="main")(x, training=training)

        # Actor head
        if self.config.std_type == StdType.MLP_HEAD:
            mean, std = jnp.split(x, 2, axis=-1)
            mean, std = mean.astype(jnp.float32), std.astype(jnp.float32)
        elif self.config.std_type == StdType.PARAM:
            mean = x
            std = self.param(  # init std to 1
                "std", nn.initializers.ones_init(), (self.config.network.output_size,)
            )
            std = jnp.broadcast_to(std, mean.shape)
        else:
            raise ValueError("Invalid std_type: %s" % self.config.std_type)

        # NOTE: STD could be different in the brax code

        std = (jax.nn.softplus(std) + self.config.min_std) * self.config.var_scale
        return distrax.MultivariateNormalDiag(mean, std)


class TanhPolicy(nn.Module):
    """SAC-style policy with tanh squashing for bounded action spaces."""

    network: type[nn.Module]
    config: PolicyNetworkConfig

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> distrax.Distribution:
        network_config = self.config.network
        # Always use MLP head for mean and log_std
        network_config = network_config.replace(output_size=network_config.output_size * 2)

        x = self.network(config=network_config, name="main")(x, training=training)

        mean, log_std = jnp.split(x, 2, axis=-1)
        mean, log_std = mean.astype(jnp.float32), log_std.astype(jnp.float32)

        # Clamp log_std for stability
        log_std = jnp.clip(log_std, -20.0, 2.0)
        std = jnp.exp(log_std)

        # Create base normal distribution
        base_dist = distrax.MultivariateNormalDiag(mean, std)

        # Apply tanh squashing
        return distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), ndims=1))


class QNetwork(nn.Module):
    """Twin Q-network for SAC with optional layer normalization.

    This implements the critic for SAC, consisting of two separate Q-networks
    (Q1 and Q2) that take (observation, action) pairs and output Q-values.
    The minimum of the two Q-values is used for updates to reduce overestimation.
    """

    network: type[nn.Module]
    config: QNetworkConfig

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array, training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass through both Q-networks.

        Args:
            obs: Observations of shape (batch, obs_dim)
            action: Actions of shape (batch, action_dim)
            training: Whether in training mode (for dropout)

        Returns:
            Tuple of (q1_values, q2_values), each of shape (batch, 1)
        """
        # Concatenate observation and action
        x = jnp.concatenate([obs, action], axis=-1)

        # Q1 network - modify config to output scalar
        q1_config = self.config.network.replace(output_size=1)
        q1 = self.network(config=q1_config, name="q1")(x, training=training)

        # Q2 network - separate parameters
        q2_config = self.config.network.replace(output_size=1)
        q2 = self.network(config=q2_config, name="q2")(x, training=training)

        return q1, q2
