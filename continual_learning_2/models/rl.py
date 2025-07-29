import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from continual_learning_2.configs.rl import PolicyNetworkConfig
from continual_learning_2.types import StdType


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
