# pyright: reportCallIssue=false
import abc
import os
from functools import partial
from typing import override

import flax.traverse_util
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.core import DenyList
from grain import python as grain
from jaxtyping import PRNGKeyArray

from continual_learning_2.configs import (
    AdamConfig,
    DatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    TrainingConfig,
)
from continual_learning_2.configs.models import CNNConfig
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

    ckpt_mgr: ocp.CheckpointManager

    def __init__(
        self,
        seed: int,
        model_config,
        optim_cfg: OptimizerConfig,
        data_cfg: DatasetConfig,
        train_cfg: TrainingConfig,
        logs_cfg: LoggingConfig,
    ):
        self.key, model_init_key = jax.random.split(jax.random.PRNGKey(seed))
        self.logger = Logger(
            logs_cfg,
            run_config={
                "model": model_config,
                "optimizer": optim_cfg,
                "dataset": data_cfg,
            },
        )
        self.dataset = get_dataset(data_cfg)
        self.train_cfg = train_cfg

        flax_module = get_model(model_config)
        optimizer = get_optimizer(optim_cfg)
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
        self.total_steps = 0

        self.ckpt_mgr = ocp.CheckpointManager(
            logs_cfg.checkpoint_dir / f"{logs_cfg.run_name}_{seed}",
            options=ocp.CheckpointManagerOptions(
                max_to_keep=5,
                # TODO: there's lots more to add here
            ),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("network_state", "key"))
    def update_network(
        network_state: TrainState, key: PRNGKeyArray, x: jax.Array, y: jax.Array
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        raise NotImplementedError

    def save(self, dataloader: grain.DataLoader, metrics: dict[str, float] | None):
        self.ckpt_mgr.save(
            self.total_steps,
            args=ocp.args.Composite(
                nn=ocp.args.StandardSave(self.network),
                key=ocp.args.JaxRandomKeySave(self.key),
                dataloader=grain.PyGrainCheckpointSave(dataloader),
                dataset=ocp.args.JsonSave(self.dataset.state),
            ),
            metrics=metrics,
        )

    def load(self, step: int):
        assert self.total_steps == 0, "Load was called before training started"
        dummy_dataloader = next(self.dataset.tasks)

        if step == -1:
            latest_step = self.ckpt_mgr.latest_step()
            assert latest_step is not None, "No checkpoint found"
            step = latest_step

        ckpt = self.ckpt_mgr.restore(
            self.ckpt_mgr.latest_step() if step == -1 else step,
            args=ocp.args.Composite(
                nn=ocp.args.StandardRestore(self.network),
                key=ocp.args.JaxRandomKeyRestore(self.key),
                dataloader=grain.PyGrainCheckpointRestore(dummy_dataloader),
                dataset=ocp.args.JsonRestore(),
            ),
        )
        self.network = ckpt["nn"]
        self.key = ckpt["key"]
        self.dataset.load(ckpt["dataset"])
        self.total_steps = step + 1

        # TODO: feed the resumed loader to the dataset

    def train(self):
        if self.train_cfg.resume:
            self.load(step=self.train_cfg.resume_from_step)

        for _, task in enumerate(self.dataset.tasks):
            for step, batch in enumerate(task):
                x, y = batch
                self.network, self.key, logs = self.update_network(
                    self.network, self.key, x, y
                )
                self.logger.accumulate(logs)

                metrics = None
                if (
                    self.logger.cfg.eval_during_training
                    and step % self.logger.cfg.eval_interval == 0
                ):
                    metrics = self.dataset.evaluate(self.network, forgetting=False)
                    self.logger.log(metrics, step=self.total_steps)

                if step % self.logger.cfg.interval == 0:
                    self.logger.push(self.total_steps)
                    self.save(dataloader=task, metrics=metrics)

                self.total_steps += 1

            self.logger.push(self.total_steps)  # Flush logger

            logs = self.dataset.evaluate(
                self.network, forgetting=self.logger.cfg.catastrophic_forgetting
            )
            self.logger.log(logs, step=self.total_steps)

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
    # trainer = HeadResetClassificationCSLTrainer(
    #     seed=SEED,
    #     model_config=MLPConfig(output_size=10),
    #     optimizer_config=AdamConfig(learning_rate=1e-3),
    #     dataset_config=DatasetConfig(
    #         name="split_mnist",
    #         seed=SEED,
    #         batch_size=64,
    #         num_tasks=5,
    #         num_epochs_per_task=1,
    #         # num_workers=(os.cpu_count() or 0) // 2,
    #         num_workers=0,
    #         dataset_kwargs={
    #             "flatten": False,
    #         },
    #     ),
    #     logging_config=LoggingConfig(
    #         run_name="split_mnist_debug_1",
    #         wandb_entity="evangelos-ch",
    #         wandb_project="continual_learning_2",
    #         wandb_mode="disabled",
    #         interval=100,
    #         eval_during_training=False,
    #     ),
    # )
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=CNNConfig(output_size=10),
        optim_cfg=AdamConfig(learning_rate=1e-3),
        data_cfg=DatasetConfig(
            name="split_cifar10",
            seed=SEED,
            batch_size=64,
            num_tasks=5,
            num_epochs_per_task=1,
            # num_workers=(os.cpu_count() or 0) // 2,
            dataset_kwargs={
                "flatten": False,
            },
        ),
        logs_cfg=LoggingConfig(
            run_name="split_cifar10_debug_1",
            wandb_entity="evangelos-ch",
            wandb_project="continual_learning_2",
            wandb_mode="disabled",
            interval=100,
            eval_during_training=False,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")
