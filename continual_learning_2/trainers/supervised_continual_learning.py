import os
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray

from continual_learning_2.configs.dataset import DatasetConfig
from continual_learning_2.configs.optim import OptimizerConfig
from continual_learning_2.data import ContinualLearningDataset, get_dataset
from continual_learning_2.models import get_model
from continual_learning_2.optim import get_optimizer
from continual_learning_2.types import LogDict
from continual_learning_2.utils.monitoring import prefix_dict, pytree_histogram

os.environ["XLA_FLAGS"] = (
    " --xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)


@partial(jax.jit, donate_argnames=("network_state", "key"))
def update_network(
    network_state: TrainState, key: PRNGKeyArray, x, y
) -> tuple[TrainState, PRNGKeyArray, LogDict]:
    key, dropout_key = jax.random.split(key)

    def loss_fn(params):
        logits = network_state.apply_fn(
            params, x, training=True, rngs={"dropout": dropout_key}
        )
        return optax.softmax_cross_entropy(logits, y).mean(), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn)(network_state.params)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)

    grads_flat, _ = jax.flatten_util.ravel_pytree(grads)
    grads_hist_dict = pytree_histogram(grads["params"])

    network_state = network_state.apply_gradients(grads=grads)
    network_params_flat, _ = jax.flatten_util.ravel_pytree(network_state.params["params"])
    network_param_hist_dict = pytree_histogram(network_state.params["params"])

    return (
        network_state,
        key,
        {
            "metrics/train_accuracy": accuracy,
            "metrics/train_loss": loss,
            "nn/gradient_norm": jnp.linalg.norm(grads_flat),
            "nn/parameter_norm": jnp.linalg.norm(network_params_flat),
            **prefix_dict("nn/gradients", grads_hist_dict),
            **prefix_dict("nn/parameters", network_param_hist_dict),
        },
    )


class SupervisedContinualLearningTrainer:
    network: TrainState
    dataset: ContinualLearningDataset
    key: PRNGKeyArray

    def __init__(
        self,
        seed: int,
        model_config,
        optimizer_config: OptimizerConfig,
        dataset_config: DatasetConfig,
    ):
        self.key, model_init_key = jax.random.split(jax.random.PRNGKey(seed))

        flax_module = get_model(model_config)
        optimizer = get_optimizer(optimizer_config)
        self.network = TrainState.create(
            apply_fn=jax.jit(flax_module.apply, static_argnames=("training")),
            params=flax_module.lazy_init(model_init_key, self.dataset.spec, training=False),
            tx=optimizer,
        )

        self.dataset = get_dataset(dataset_config)

    def train(self):
        total_steps = 0
        for i, task in enumerate(self.dataset.tasks):
            for step, batch in enumerate(task):
                x, y = batch
                self.network, self.key, logs = update_network(self.network, self.key, x, y)
                if step % 100 == 0:
                    print(f"Step: {step}, Loss: {logs['metrics/train_loss']}")
                total_steps += 1

            logs = self.dataset.evaluate(
                lambda x: self.network.apply_fn(self.network.params, x), forgetting=True
            )
            print(f"Task: {i}")
            print(logs)


if __name__ == "__main__":
    # import os
    # import time

    # start = time.time()
    # dataset_config = DatasetConfig(
    #     num_tasks=5,
    #     num_epochs_per_task=10,
    #     batch_size=64,
    #     seed=42,
    #     num_workers=(os.cpu_count() or 0) // 2,
    # )
    # dataset = SplitMNIST(dataset_config)
    # train(dataset)
    # print(f"Training time: {time.time() - start:.2f} seconds")
    ...
