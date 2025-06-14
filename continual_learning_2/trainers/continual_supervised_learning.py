import abc
import os
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray

from continual_learning_2.configs import (
    AdamConfig,
    DatasetConfig,
    LoggingConfig,
    MLPConfig,
    OptimizerConfig,
)
from continual_learning_2.data import ContinualLearningDataset, get_dataset
from continual_learning_2.models import get_model
from continual_learning_2.optim import get_optimizer
from continual_learning_2.types import LogDict
from continual_learning_2.utils.monitoring import Logger, prefix_dict, pytree_histogram

os.environ["XLA_FLAGS"] = (
    " --xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)


class CSLTrainerBase(abc.ABC):
    network: TrainState
    key: PRNGKeyArray

    dataset: ContinualLearningDataset
    logger: Logger

    def __init__(
        self,
        seed: int,
        model_config,
        optimizer_config: OptimizerConfig,
        dataset_config: DatasetConfig,
        logging_config: LoggingConfig,
    ):
        self.key, model_init_key = jax.random.split(jax.random.PRNGKey(seed))
        self.logger = Logger(
            logging_config,
            run_config={
                "model": model_config,
                "optimizer": optimizer_config,
                "dataset": dataset_config,
            },
        )
        self.dataset = get_dataset(dataset_config)

        flax_module = get_model(model_config)
        optimizer = get_optimizer(optimizer_config)
        self.network = TrainState.create(
            apply_fn=jax.jit(flax_module.apply, static_argnames=("training")),
            params=flax_module.lazy_init(model_init_key, self.dataset.spec, training=False),
            tx=optimizer,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("network_state", "key"))
    def update_network(
        network_state: TrainState, key: PRNGKeyArray, x: jax.Array, y: jax.Array
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        raise NotImplementedError

    def train(self):
        total_steps = 0
        for i, task in enumerate(self.dataset.tasks):
            for step, batch in enumerate(task):
                x, y = batch
                self.network, self.key, logs = self.update_network(
                    self.network, self.key, x, y
                )
                self.logger.accumulate(logs)

                if step % self.logger.cfg.interval == 0:
                    self.logger.push(total_steps)

                if (
                    self.logger.cfg.eval_during_training
                    and step % self.logger.cfg.eval_interval == 0
                ):
                    logs = self.dataset.evaluate(self.network, forgetting=False)
                    self.logger.log(logs, step=total_steps)

                total_steps += 1

            logs = self.dataset.evaluate(self.network, forgetting=True)
            self.logger.push(total_steps)  # Flush logger
            self.logger.log(logs, step=total_steps)

        self.logger.close()


class ClassificationCSLTrainer(CSLTrainerBase):
    @staticmethod
    @partial(jax.jit, donate_argnames=("network_state", "key"))
    def update_network(
        network_state: TrainState,
        key: PRNGKeyArray,
        x: jax.Array,
        y: jax.Array,
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        key, dropout_key = jax.random.split(key)

        def loss_fn(params):
            logits = network_state.apply_fn(
                params, x, training=True, rngs={"dropout": dropout_key}
            )
            return optax.softmax_cross_entropy(logits, y).mean(), logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(network_state.params)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y.argmax(axis=-1))

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


class MaskedClassificationCSLTrainer(CSLTrainerBase):
    @staticmethod
    @partial(jax.jit, donate_argnames=("network_state", "key"))
    def update_network(
        network_state: TrainState,
        key: PRNGKeyArray,
        x: jax.Array,
        y: jax.Array,
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        key, dropout_key = jax.random.split(key)

        active_labels = jnp.any(y.astype(bool), axis=0)
        loss_mask = jnp.broadcast_to(active_labels, y.shape)

        def loss_fn(params):
            logits = network_state.apply_fn(
                params, x, training=True, rngs={"dropout": dropout_key}
            )
            return optax.softmax_cross_entropy(logits, y, where=loss_mask).mean(), logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(network_state.params)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y.argmax(axis=-1))

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


if __name__ == "__main__":
    import os
    import time

    SEED = 42

    start = time.time()
    trainer = MaskedClassificationCSLTrainer(
        seed=SEED,
        model_config=MLPConfig(output_size=10),
        optimizer_config=AdamConfig(learning_rate=1e-3),
        dataset_config=DatasetConfig(
            name="split_mnist",
            seed=SEED,
            batch_size=64,
            num_tasks=5,
            num_epochs_per_task=10,
            num_workers=(os.cpu_count() or 0) // 2,
        ),
        logging_config=LoggingConfig(
            run_name="split_mnist_debug_0",
            wandb_entity="evangelos-ch",
            wandb_project="continual_learning_2",
            interval=100,
            eval_during_training=False,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")
