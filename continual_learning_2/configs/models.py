from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class MLPConfig:
    output_size: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float | None = None

    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16
