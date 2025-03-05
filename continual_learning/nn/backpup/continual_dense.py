from typing import Any, Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax.random import PRNGKey
from flax.linen.initializers import lecun_normal  # Custom one in initializers.py?
from jax.lax import Precision, dot_general
from flax.linen.initializers import zeros
from flax.linen.dtypes import promote_dtype

PrecisionLike = None | str | Precision | Tuple[str, str] | Tuple[Precision, Precision]
default_kernel_init = lecun_normal()


class ContinualDense(nn.Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Any | None = None
    param_dtype: Any = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Tuple[int, ...], Any], Any] = default_kernel_init
    bias_init: Callable[[PRNGKey, Tuple[int, ...], Any], Any] = zeros
    replacement_rate: float = 1e-4
    maturity_threshold=100
    util_type='contribution'
    decay_rate=0

    @nn.compact
    def __call__(self, inputs: Any) -> Any:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        # cbp params

        age = self.param(
            "age", zeros, (self.features,), self.param_dtype,
        )
        util = self.param(
            "utility", zeros, (self.features,), self.param_dtype,
        )
        n_features_to_replace = self.param(
            "n_feat_replace", zeros, (1), self.param_dtype,
        )
        # bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))
        if self.replacement_rate > 0:
            # Reinit every time gradients
            pass # Register backwardsnd forward hook here using custom vjp
        #

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias, age = promote_dtype(inputs, kernel, bias, age, dtype=self.dtype)
        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y
    
    def reininit_featues(self):
        print("Working")
        return 10.
