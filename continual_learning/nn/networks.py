import flax.linen as nn
import jax

class SimpleNet(nn.Module):
    """Simple Flax neural network for sine wave regression"""
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

