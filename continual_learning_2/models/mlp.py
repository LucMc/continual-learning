from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from continual_learning_2.configs.models import MLPConfig


class MLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        Dense = partial(
            nn.Dense,
            use_bias=self.config.use_bias,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dtype=self.config.dtype,
        )

        for i in range(self.config.num_layers):
            x = Dense(self.config.hidden_size, name=f"layer_{i}")(x)
            x = self.config.activation_fn(x)
            if self.config.dropout is not None:
                x = nn.Dropout(self.config.dropout, deterministic=not training)(x)
            self.sow("intermediates", f"layer_{i}", x)

        x = nn.Dense(self.config.output_size, name="output")(x)
        self.sow("intermediates", "outputs", x)
        return x.astype(jnp.float32)
