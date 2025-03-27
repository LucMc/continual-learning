import flax.linen as nn
import jax

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

class OnlineNormNet(nn.Module):
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

class TestNet(nn.Module):
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
