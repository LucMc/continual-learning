import abc
import os
from functools import partial
from sys import prefix
from typing import override

import flax.traverse_util
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from flax.core import DenyList
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
from continual_learning_2.utils.monitoring import (
    Logger,
    compute_srank,
    get_dormant_neuron_logs,
    get_linearised_neuron_logs,
    prefix_dict,
    pytree_histogram,
)
from continual_learning_2.utils.training import TrainState

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
            apply_fn=jax.jit(flax_module.apply, static_argnames=("training", "mutable")),
            params=flax_module.lazy_init(
                model_init_key,
                self.dataset.spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=optimizer,
            kernel_init=model_config.kernel_init,
            bias_init=model_config.bias_init,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("network_state", "key"))
    def update_network(
        network_state: TrainState, key: PRNGKeyArray, x: jax.Array, y: jax.Array
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        raise NotImplementedError

    def train(self):
        total_steps = 0
        for _, task in enumerate(self.dataset.tasks):
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

            logs = self.dataset.evaluate(
                self.network, forgetting=self.logger.cfg.catastrophic_forgetting
            )
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
            logits, intermediates = network_state.apply_fn(
                params,
                x,
                training=True,
                rngs={"dropout": dropout_key},
                mutable=("activations", "preactivations"),
            )
            return optax.softmax_cross_entropy(logits, y).mean(), (logits, intermediates)

        (loss, (logits, intermediates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            network_state.params
        )
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y.argmax(axis=-1))

        activations = intermediates["activations"]
        activations_hist_dict = pytree_histogram(activations)
        activations_flat = {
            k: v[0]  # pyright: ignore[reportIndexIssue]
            for k, v in flax.traverse_util.flatten_dict(activations, sep="/").items()
        }  # pyright: ignore[reportIndexIssue]
        dormant_neuron_logs = get_dormant_neuron_logs(activations_flat)  # pyright: ignore[reportArgumentType]
        srank_logs = jax.tree.map(compute_srank, activations_flat)

        preactivations = intermediates["preactivations"]
        preactivations_flat = {
            k: v[0]  # pyright: ignore[reportIndexIssue]
            for k, v in flax.traverse_util.flatten_dict(preactivations, sep="/").items()
        }  # pyright: ignore[reportIndexIssue]
        linearised_neuron_logs = get_linearised_neuron_logs(preactivations_flat)  # pyright: ignore[reportArgumentType]

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
                **prefix_dict("nn/activations", activations_hist_dict),
                **prefix_dict("nn/dormant_neurons", dormant_neuron_logs),
                **prefix_dict("nn/linearised_neurons", linearised_neuron_logs),
                **prefix_dict("nn/srank", srank_logs),
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

        # NOTE: this is what's different to the normal ClassificationCSLTrainer
        active_labels = jnp.any(y.astype(bool), axis=0)
        loss_mask = jnp.broadcast_to(active_labels, y.shape)

        def loss_fn(params):
            logits, intermediates = network_state.apply_fn(
                params,
                x,
                training=True,
                rngs={"dropout": dropout_key},
                mutable=("activations", "preactivations"),
            )
            # NOTE:                         we use the mask here vvvvv
            return optax.softmax_cross_entropy(logits, y, where=loss_mask).mean(), (
                logits,
                intermediates,
            )

        (loss, (logits, intermediates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            network_state.params
        )
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y.argmax(axis=-1))

        activations = intermediates["activations"]
        activations_hist_dict = pytree_histogram(activations)
        activations_flat = {
            k: v[0]  # pyright: ignore[reportIndexIssue]
            for k, v in flax.traverse_util.flatten_dict(activations, sep="/").items()
        }
        dormant_neuron_logs = get_dormant_neuron_logs(activations_flat)  # pyright: ignore[reportArgumentType]
        srank_logs = jax.tree.map(compute_srank, activations_flat)

        preactivations = intermediates["preactivations"]
        preactivations_flat = {
            k: v[0]  # pyright: ignore[reportIndexIssue]
            for k, v in flax.traverse_util.flatten_dict(preactivations, sep="/").items()
        }
        linearised_neuron_logs = get_linearised_neuron_logs(preactivations_flat)  # pyright: ignore[reportArgumentType]

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
                **prefix_dict("nn/activations", activations_hist_dict),
                **prefix_dict("nn/dormant_neurons", dormant_neuron_logs),
                **prefix_dict("nn/linearised_neurons", linearised_neuron_logs),
                **prefix_dict("nn/srank", srank_logs),
                **prefix_dict("nn/gradients", grads_hist_dict),
                **prefix_dict("nn/parameters", network_param_hist_dict),
            },
        )


class HeadResetClassificationCSLTrainer(ClassificationCSLTrainer):
    @override
    def train(self):
        total_steps = 0
        for _, task in enumerate(self.dataset.tasks):
            for step, batch in enumerate(task):
                x, y = batch
                breakpoint()
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

            logs = self.dataset.evaluate(
                self.network, forgetting=self.logger.cfg.catastrophic_forgetting
            )
            self.logger.push(total_steps)  # Flush logger
            self.logger.log(logs, step=total_steps)

            # NOTE: this is the difference to the base training loop
            # we reset the last layer of the network
            self.key, head_reset_key = jax.random.split(self.key, num=2)
            self.network = self.network.reset_layer(head_reset_key, "output")

        self.logger.close()


if __name__ == "__main__":
    import os
    import time

    SEED = 42

    start = time.time()
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=MLPConfig(output_size=10),
        optimizer_config=AdamConfig(learning_rate=1e-3),
        dataset_config=DatasetConfig(
            name="split_mnist",
            seed=SEED,
            batch_size=64,
            num_tasks=5,
            num_epochs_per_task=1,
            # num_workers=(os.cpu_count() or 0) // 2,
            num_workers=0,
            dataset_kwargs={
                "flatten": False,
            },
        ),
        logging_config=LoggingConfig(
            run_name="split_mnist_debug_1",
            wandb_entity="evangelos-ch",
            wandb_project="continual_learning_2",
            wandb_mode="disabled",
            interval=100,
            eval_during_training=False,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")
