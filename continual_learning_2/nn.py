from dataclasses import dataclass
from functools import partial

import distrax
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

from continual_learning_2.types import Activation, LayerNorm, LayerNormPosition, StdType


@dataclass(frozen=True)
class ValueFunctionConfig:
    width: int = 128
    depth: int = 2
    activation: Activation = Activation.ReLU
    weight_init: nn.initializers.Initializer = nn.initializers.he_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    layer_norm: LayerNorm = LayerNorm.NONE
    layer_norm_position: LayerNormPosition = LayerNormPosition.POST


class ValueFunction(nn.Module):
    config: ValueFunctionConfig

    @nn.compact
    def __call__(self, x: Float[Array, "... obs_dim"]) -> Float[Array, "... 1"]:
        Dense = partial(
            nn.Dense, kernel_init=self.config.weight_init, bias_init=self.config.bias_init
        )

        # Trunk
        for i in range(self.config.depth):
            if self.layer_norm_position == LayerNormPosition.PRE and i > 0:
                x = self.layer_norm(name=f"layer_norm_{i}")(x)

            x = Dense(self.config.width, name=f"dense_{i}")(x)
            x = self.config.activation(x)

            if self.layer_norm_position == LayerNormPosition.POST:
                x = self.layer_norm(name=f"layer_norm_{i}")(x)

            self.sow("intermediates", f"activations_{i}", x)

        # Head
        return Dense(1, name="out_layer")(x)


@dataclass(frozen=True)
class PolicyConfig:
    action_dim: int
    std_type: StdType = StdType.PARAM

    width: int = 64
    depth: int = 2
    activation: Activation = Activation.ReLU
    weight_init: nn.initializers.Initializer = nn.initializers.he_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    layer_norm: LayerNorm = LayerNorm.NONE
    layer_norm_position: LayerNormPosition = LayerNormPosition.PRE


class Policy(nn.Module):
    config: PolicyConfig

    @nn.compact
    def __call__(self, x: Float[Array, "... obs_dim"]) -> distrax.Distribution:
        Dense = partial(
            nn.Dense, kernel_init=self.config.weight_init, bias_init=self.config.bias_init
        )

        # Trunk
        for i in range(self.config.depth):
            if self.config.layer_norm_position == LayerNormPosition.PRE and i > 1:
                x = self.layer_norm(name=f"layer_norm_{i}")(x)

            x = Dense(self.config.width, name=f"dense_{i}")(x)
            x = self.config.activation(x)

            if self.config.layer_norm_position == LayerNormPosition.POST:
                x = self.layer_norm(name=f"layer_norm_{i}")(x)

            self.sow("intermediates", f"activations_{i}", x)

        # Actor head
        if self.config.std_type == StdType.MLP_HEAD:
            x = Dense(self.config.action_dim * 2, name="mu_and_log_std")(x)
            mean, log_std = jnp.split(x, 2, axis=-1)
        elif self.config.std_type == StdType.PARAM:
            mean = x
            log_std = self.param(  # init std to 1
                "log_std", nn.initializers.zeros_init(), (self.action_dim,)
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
        else:
            raise ValueError("Invalid std_type: %s" % self.config.std_type)

        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
