from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from continual_learning_2.configs.dataset import DatasetConfig
from continual_learning_2.data import ContinualLearningDataset, SplitMNIST


class Network(nn.Module):
    num_classes: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(256, dtype=self.dtype)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(256, dtype=self.dtype)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


@partial(jax.jit, donate_argnames=("network_state"))
def update_network(network_state: TrainState, x, y) -> tuple[TrainState, float]:
    def loss_fn(params):
        logits = network_state.apply_fn(params, x)
        return optax.softmax_cross_entropy(logits, y).mean()

    loss, grads = jax.value_and_grad(loss_fn)(network_state.params)
    new_network_state = network_state.apply_gradients(grads=grads)
    return new_network_state, loss


def train(dataset: ContinualLearningDataset):
    key = jax.random.PRNGKey(42)

    network = Network(dataset.NUM_CLASSES)
    optimizer = optax.adam(learning_rate=3e-4)
    init_params = network.lazy_init(
        key, jax.ShapeDtypeStruct((1, 28 * 28), jnp.float32)
    )  # TODO: Find a way to get this from the dataset lol
    x = jnp.ones((1, 28 * 28), jnp.float32)
    print(network.tabulate(key, x))

    network_state = TrainState.create(
        apply_fn=jax.jit(network.apply), params=init_params, tx=optimizer
    )

    steps = 0
    for i, task in enumerate(dataset.tasks):
        for batch in task:
            x, y = batch
            network_state, loss = update_network(network_state, x, y)
            if steps % 100 == 0:
                print(f"Step: {steps}, Loss: {loss}")
            steps += 1

        logs = dataset.evaluate(
            lambda x: network_state.apply_fn(network_state.params, x), forgetting=False
        )
        print(f"Task: {i}")
        print(logs)
        break


if __name__ == "__main__":
    import os
    import time

    start = time.time()
    dataset_config = DatasetConfig(
        num_tasks=5,
        num_epochs_per_task=5,
        batch_size=32,
        seed=42,
        num_workers=(os.cpu_count() or 0) // 2,
    )
    dataset = SplitMNIST(dataset_config)
    train(dataset)
    print(f"Training time: {time.time() - start:.2f} seconds")
