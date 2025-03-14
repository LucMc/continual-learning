import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training.train_state import TrainState

# from continual_learning.nn.continual_dense import ContinualDense
from continual_learning.optim.continual_backprop_full import (
    continual_backprop,
    CBPTrainState,
)


class FFNN(nn.Module):
    """Simple Flax neural network"""

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        layers = [
            "dense1",
            "dense2",
            "dense3",
        ]  # Could make list of dicts if I wanna changes sizes per layer

        for i in range(len(layers)):
            x = nn.Dense(features=128, name=layers[i])(x)
            x = nn.relu(x)
            intermediates[layers[i]] = x

        # Output layer
        x = nn.Dense(features=10, name="out_layer")(x)
        intermediates["out_layer"] = x
        self.sow(
            "intermediates", "activations", intermediates
        )  # Only really want to reset layers after an activation


        return x

    # @jax.jit
    def predict(self, params, x):
        return self.apply({"params": params}, x, capture_intermediates=True)


### testing ###
def test_optim():
    # Initialize random key
    key = random.PRNGKey(0)

    # Create dummy data (batch_size=2, input_dim=784 for MNIST)
    batch_size = 1
    input_dim = 784
    dummy_input = random.normal(key, (batch_size, input_dim))
    dummy_labels = jax.nn.one_hot(random.randint(key, (batch_size,), 0, 10), 10)

    net_custom = FFNN()
    params = net_custom.init(key, dummy_input)

    # tx = optax.chain(
    #     continual_backprop(),
    #     # optax.adam(1e-1)  # Base optimiser
    # )
    tx_adam = optax.adam(0)

    net_ts = CBPTrainState.create(apply_fn=net_custom.predict, params=params, tx=tx_adam, maturity_threshold=4)
    net_ts_adam = TrainState.create(
        apply_fn=net_custom.predict, params=params, tx=tx_adam
    )

    # print weight layer sizes
    for name, param in net_ts.params["params"].items(): print(name, param["kernel"].shape)

    # Test backwards pass
    def loss_fn(params, inputs, labels):
        logits, features = net_custom.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
        return loss, (logits, features)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, (logits, features)), grads = grad_fn(
        net_ts.params, dummy_input, dummy_labels
    )

    # Update parameters
    initial_weights = net_ts.params["params"]["dense1"]["kernel"]

    for i in range(10):  # 10 grad steps
        net_ts = net_ts.apply_gradients(grads=grads, features=features)
        net_ts_adam = net_ts_adam.apply_gradients(grads=grads)

    updated_weights = net_ts.params["params"]["dense1"]["kernel"]
    updated_weights_adam = net_ts_adam.params["params"]["dense1"]["kernel"]

    print(f"Loss: {loss}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    # print("features", features)
    print(f"Number of intermediate features: {len(features['intermediates'])}")

    # for i, feat in enumerate(features):
    #     print(f"Feature {i} shape: {feat.shape}")

    # Print weight changes
    weight_diff = jnp.abs(updated_weights - initial_weights)
    weight_diff_adam = jnp.abs(updated_weights_adam - initial_weights)

    print(f"Max weight change: {weight_diff.max()}")
    print(f"Mean weight change: {weight_diff.mean()}")
    print(f"Mean weight value: {updated_weights.mean()}")
    print(f"weights > 0: {(updated_weights > 0).sum()}")
    print(updated_weights.shape)

    print(f"Max weight change adam: {weight_diff_adam.max()}")
    print(f"Mean weight change adam: {weight_diff_adam.mean()}")
    print(f"Mean weight value adam: {updated_weights_adam.mean()}")
    print(f"weights > 0 adam: {(updated_weights_adam > 0).sum()}")


if __name__ == "__main__":  # test neural network
    test_optim()
