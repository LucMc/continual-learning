from flax.linen.module import capture_call_intermediates
from jaxtyping import Array
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial

'''
TODO:
 - Compare pre and post activation normalisation as different sources say differently i.e. Disentangling the causes of plasticity in NNs paper
'''
# Reinforcement Learning base
class ValueNet(nn.Module):
    h_size: int = 128 # Size of hidden dimension
    layer_names: tuple = ("dense1", "dense2")

    @nn.compact
    def __call__(self, x) -> Array:
        intermediates = {}

        for i, layer_name in enumerate(self.layer_names):
            x = nn.Dense(features=self.h_size, name=layer_name)(x)
            x = nn.relu(x)
            intermediates[layer_name] = x

        value = nn.Dense(1, name="out_layer")(x)

        self.sow("intermediates", "activations", intermediates)
        return value

    @partial(jax.jit, static_argnames="self")
    def apply_w_features(self, params, x):
        return self.apply(params, x, capture_intermediates=True)


class ActorNet(nn.Module):
    n_actions: int
    h_size: int = 64 # Size of hidden dimension
    layer_names: tuple = ("dense1", "dense2")

    @nn.compact
    def __call__(self, x) -> distrax.Distribution:
        intermediates = {}

        for i, layer_name in enumerate(self.layer_names):
            x = nn.Dense(features=self.h_size, name=layer_name)(x)
            x = nn.relu(x)
            intermediates[layer_name] = x

        mean = nn.Dense(self.n_actions, name="out_layer")(x)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (
                1,
                self.n_actions,
            ),
        )
        logstd_batch = jnp.broadcast_to(
            log_std, mean.shape
        )  # Make logstd the same shape as actions

        self.sow("intermediates", "activations", intermediates)
        return mean, jnp.exp(logstd_batch)

    @partial(jax.jit, static_argnames="self")
    def apply_w_features(self, params, x):
        return self.apply(params, x, capture_intermediates=True)


# Reinforcement Learning Layer Norm
class ActorNetLayerNorm(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x) -> distrax.Distribution:
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.LayerNorm(name=f"layer_norm_1")(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.LayerNorm(name=f"layer_norm_2")(x)

        mean = nn.Dense(self.n_actions, name="mu")(x)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (
                1,
                self.n_actions,
            ),
        )
        logstd_batch = jnp.broadcast_to(
            log_std, mean.shape
        )  # Make logstd the same shape as actions
        return distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(logstd_batch)
        )

    @partial(jax.jit, static_argnames="self")
    def apply_w_features(self, params, x):
        return self.apply(params, x, capture_intermediates=True)

class ValueNetLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x) -> Array:
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.LayerNorm(name=f"layer_norm_1")(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.LayerNorm(name=f"layer_norm_2")(x)
        q_value = nn.Dense(1)(x)
        return q_value

    @partial(jax.jit, static_argnames="self")
    def apply_w_features(self, params, x):
        return self.apply(params, x, capture_intermediates=True)

class SimpleNet(nn.Module):
    n_out: int = 1
    h_size: int = 128

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        layers = [
            "dense1",
            "dense2",
            "dense3",
        ]

        for i, layer_name in enumerate(layers):
            x = nn.Dense(features=self.h_size, name=layer_name)(x)
            x = nn.relu(x)
            intermediates[layer_name] = x

        x = nn.Dense(features=self.n_out, name="out_layer")(x)
        # intermediates["out_layer"] = x

        self.sow("intermediates", "activations", intermediates)
        return x

    @jax.jit
    def predict(self, params, x):
        return self.apply({"params": params}, x, capture_intermediates=True)


class SimpleNetLayerNorm(nn.Module):
    n_out: int = 1
    h_size: int = 128

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        layers = [
            "dense1",
            "dense2",
            "dense3",
        ]

        for i, layer_name in enumerate(layers):
            x = nn.Dense(features=self.h_size, name=layer_name)(x)
            x = nn.relu(x)
            x = nn.LayerNorm(name=f"layer_norm_{i}")(x)
            intermediates[layer_name] = x

        # x = nn.LayerNorm(name=f"layer_norm_out")(x)
        x = nn.Dense(features=self.n_out, name="out_layer")(x)
        # intermediates["out_layer"] = x

        self.sow("intermediates", "activations", intermediates)
        return x

    @jax.jit
    def predict(self, params, x):
        return self.apply({"params": params}, x, capture_intermediates=True)


class SimpleTestNet(nn.Module):
    k_init = nn.initializers.zeros_init()
    b_init = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        layers = [
            "dense1",
            "dense2",
            "dense3",
        ]

        for i, layer_name in enumerate(layers):
            x = nn.Dense(
                features=4,
                name=layer_name,
                # kernel_init=self.k_init,
                # bias_init=self.b_init
            )(x)
            x = nn.relu(x)
            intermediates[layer_name] = x

        # Single output for regression
        x = nn.Dense(
            features=1,
            name="out_layer",
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )(x)

        self.sow("intermediates", "activations", intermediates)
        return x

    @jax.jit
    def predict(self, params, x):
        return self.apply({"params": params}, x, capture_intermediates=True)


