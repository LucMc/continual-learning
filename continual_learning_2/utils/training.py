from typing import Callable, Self

import jax
from flax import struct
import flax.core
from flax.training.train_state import TrainState as FlaxTrainState
from jaxtyping import PRNGKeyArray


class TrainState(FlaxTrainState):
    kernel_init: jax.nn.initializers.Initializer = struct.field(pytree_node=False)
    bias_init: jax.nn.initializers.Initializer = struct.field(pytree_node=False)

    def reset_layer(self, rng_key: PRNGKeyArray, layer: str) -> Self:
        layer_params = self.params["params"][layer]
        kernel, bias = layer_params["kernel"], layer_params["bias"]
        assert isinstance(kernel, jax.Array) and isinstance(bias, jax.Array)
        new_layer_kernel = self.kernel_init(rng_key, kernel.shape, kernel.dtype)
        new_layer_bias = self.bias_init(rng_key, bias.shape, bias.dtype)

        new_params = self.params
        new_params["params"][layer] = {"kernel": new_layer_kernel, "bias": new_layer_bias}

        return self.replace(params=new_params)
