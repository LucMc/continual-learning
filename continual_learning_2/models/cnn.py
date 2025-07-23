from functools import partial

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

from continual_learning_2.configs.models import CNNConfig
from continual_learning_2.utils.nn import flatten_last


class CNN(nn.Module):
    """Based on VGG."""

    config: CNNConfig

    @nn.compact
    def __call__(
        self, x: Float[Array, "... H W C"], training: bool = True
    ) -> Float[Array, "... O"]:
        Conv = partial(
            nn.Conv,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            strides=self.config.strides,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
        )
        Dense = partial(
            nn.Dense,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
        )
        idx = 0

        # ConvNet feature extractor
        for feature in self.config.features:
            for layer in range(self.config.num_convs_per_layer):
                x = Conv(features=feature, name=f"{idx}_conv_{feature}_{layer}")(x) # Make variable and sync with others
                self.sow("preactivations", f"{idx}_conv_{feature}_{layer}_pre", flatten_last(x))
                x = self.config.activation_fn(x)

                if self.config.dropout is not None:
                    x = nn.Dropout(self.config.dropout, deterministic=not training)(x)

                self.sow("activations", f"{idx}_conv_{feature}_{layer}_act", x)
                idx += 1 # To fix alphabetical ordering

            if self.config.use_max_pooling:
                x = nn.max_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding=self.config.padding
                )

        # MLP head
        x = x.reshape((x.shape[0], -1))
        for layer in range(self.config.num_dense_layers):
            x = Dense(self.config.dense_hidden_size, name=f"{idx}_Dense_{layer}")(x)
            self.sow("preactivations", f"{idx}_Dense_{layer}_pre", x)
            x = self.config.activation_fn(x)
            if self.config.dropout is not None:
                x = nn.Dropout(self.config.dropout, deterministic=not training)(x)
            self.sow("activations", f"{idx}_Dense_{layer}_act", x)
            idx += 1

        # "output" is already last in the order and can stay the same for consistancy with MLP
        self.sow("preactivations", "output_pre", x)
        x = Dense(self.config.output_size, name=f"output")(x)
        self.sow("activations", f"output_act", x)
        return x.astype(jnp.float32)
