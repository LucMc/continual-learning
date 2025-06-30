from typing import Generator, Never, Self

import jax
import numpy as np
from flax import struct
from flax.training.train_state import TrainState as FlaxTrainState
from jaxtyping import PRNGKeyArray

from continual_learning_2.types import Rollout


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


    # Pass features to optimizer and set params with update
    def apply_gradients(self, *, grads, features=None, **kwargs):
        assert features, "Features must be provided to apply_gradients()"

        new_params, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, features=features
        )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def to_minibatch_iterator(
    data: Rollout, num: int, seed: int, flatten_batch_dims: bool = True
) -> Generator[Rollout, None, Never]:
    # Flatten batch dims
    rollouts = data
    if flatten_batch_dims:
        rollouts = Rollout(
            *map(
                lambda x: x.reshape(-1, x.shape[-1]) if x is not None else None,
                data,
            )  # pyright: ignore[reportArgumentType]
        )

    rollout_size = rollouts.observations.shape[0]
    minibatch_size = rollout_size // num

    rng = np.random.default_rng(seed)
    rng_state = rng.bit_generator.state

    while True:
        for field in rollouts:
            rng.bit_generator.state = rng_state
            if field is not None:
                rng.shuffle(field, axis=0)
        rng_state = rng.bit_generator.state
        for start in range(0, rollout_size, minibatch_size):
            end = start + minibatch_size
            yield Rollout(
                *map(
                    lambda x: x[start:end] if x is not None else None,  # pyright: ignore[reportArgumentType]
                    rollouts,
                )
            )
