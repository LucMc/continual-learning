import flax.linen as nn
import jax


class Identity(nn.Module):
    """Identity function as a Flax module"""

    def __call__(self, x):
        return x


def flatten_last(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0], -1))
