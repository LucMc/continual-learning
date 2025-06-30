from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp


@dataclass
class MLPConfig:
    output_size: int = 1
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float | None = None

    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16


@dataclass
class CNNConfig:
    output_size: int = 1

    # ConvNet feature extractor
    features: Sequence[int] = (64, 128, 256)
    num_convs_per_layer: int = 2
    kernel_size: int | Sequence[int] = (3, 3)
    strides: int | Sequence[int] = 1
    padding: Literal["SAME", "VALID", "CIRCULAR"] = "SAME"
    use_max_pooling: bool = True

    # MLP head
    dense_hidden_size: int = 256
    num_dense_layers: int = 2

    dropout: float | None = None

    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16


@dataclass
class ResNetConfig:
    output_size: int = 1

    # ConvNet feature extractor
    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    num_filters: int = 64

    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros  # pyright: ignore[reportAssignmentType]
    dtype: jnp.dtype = jnp.bfloat16
