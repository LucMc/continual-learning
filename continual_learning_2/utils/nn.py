import flax.linen as nn


class Identity(nn.Module):
    """Identity function as a Flax module"""

    def __call__(self, x):
        return x
