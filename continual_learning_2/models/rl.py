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
        x = self.network(config=self.config.network, name="main")(x, training=training)

        # Actor head
        if self.config.std_type == StdType.MLP_HEAD:
            mean, log_std = jnp.split(x, 2, axis=-1)
        elif self.config.std_type == StdType.PARAM:
            mean = x
            log_std = self.param(  # init std to 1
                "log_std", nn.initializers.zeros_init(), (self.config.network.output_size,)
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
        else:
            raise ValueError("Invalid std_type: %s" % self.config.std_type)

        log_std = jnp.clip(
            log_std, a_min=self.config.log_std_min, a_max=self.config.log_std_max
        )

        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
