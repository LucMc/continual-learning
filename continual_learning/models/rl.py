import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig
from continual_learning.types import StdType


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


class QNetwork(nn.Module):
    """Twin Q-network for SAC."""

    network: type[nn.Module]
    config: QNetworkConfig

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array, training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([obs, action], axis=-1)

        q1_config = self.config.network.replace(output_size=1)
        q1 = self.network(config=q1_config, name="q1")(x, training=training)

        q2_config = self.config.network.replace(output_size=1)
        q2 = self.network(config=q2_config, name="q2")(x, training=training)

        return q1, q2


class TanhPolicy(nn.Module):
    """SAC-style policy with tanh squashing for bounded action spaces."""

    network: type[nn.Module]
    config: PolicyNetworkConfig

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> distrax.Distribution:
        network_config = self.config.network
        network_config = network_config.replace(output_size=network_config.output_size * 2)

        x = self.network(config=network_config, name="main")(x, training=training)

        mean, log_std = jnp.split(x, 2, axis=-1)
        mean, log_std = mean.astype(jnp.float32), log_std.astype(jnp.float32)

        log_std = jnp.clip(log_std, -20.0, 2.0)
        std = jnp.exp(log_std)

        base_dist = distrax.MultivariateNormalDiag(mean, std)
        return distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), ndims=1))
