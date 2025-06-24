# Based on https://github.com/google/flax/blob/main/examples/imagenet/models.py
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from continual_learning_2.configs.models import ResNetConfig
from continual_learning_2.utils.nn import flatten_last


class ResNetBlock(nn.Module):
    conv: Any
    norm: Any
    activation_fn: Callable[[jax.Array], jax.Array]

    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: Float[Array, "... H W C"]) -> Float[Array, "... H W C"]:
        residual = x

        y = self.conv(kernel_size=3, strides=self.strides, name=f"{self.name}_conv1")(x)
        self.sow("preactivations", f"{self.name}_conv1_pre", flatten_last(x))
        y = self.norm()(y)
        y = self.activation_fn(y)
        self.sow("activations", f"{self.name}_conv1_act", flatten_last(y))

        y = self.conv(kernel_size=3, strides=1, name=f"{self.name}_conv2")(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)
        self.sow("preactivations", f"{self.name}_conv2_pre", flatten_last(x))

        if residual.shape != y.shape:
            residual = self.conv(kernel_size=1, strides=self.strides)(residual)
            residual = self.norm()(residual)

        y = self.activation_fn(y + residual)
        self.sow("activations", f"{self.name}_conv2_act", flatten_last(y))
        return y


class ResNet(nn.Module):
    config: ResNetConfig

    @nn.compact
    def __call__(
        self, x: Float[Array, "... H W C"], training: bool = True
    ) -> Float[Array, "... O"]:
        Conv = partial(
            nn.Conv,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            use_bias=False,
            dtype=self.config.dtype,
        )
        Norm = partial(
            nn.BatchNorm,
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.config.dtype,
            axis_name="batch",
        )
        Dense = partial(
            nn.Dense,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            use_bias=False,
            dtype=self.config.dtype,
        )

        x = Conv(
            self.config.num_filters,
            kernel_size=7,
            strides=2,
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)
        x = Norm(name="norm_init")(x)
        x = self.config.activation_fn(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.config.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetBlock(
                    conv=partial(Conv, num_features=self.num_filters * 2**i),
                    norm=Norm,
                    activation_fn=self.config.activation_fn,
                    strides=strides,
                    name=f"block_{i}_{j}",
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        self.sow("preactivations", "output_pre", x)
        x = Dense(self.config.output_size, name="output")(x)
        self.sow("activations", "output_act", x)
        return x.astype(jnp.float32)
