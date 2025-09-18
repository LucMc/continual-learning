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
import orbax.checkpoint.checkpoint_managers as ocp_mgrs
from flax.core import DenyList
from grain import python as grain
from jaxtyping import PRNGKeyArray

from continual_learning.configs import (
    AdamConfig,
    DatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    TrainingConfig,
)
from continual_learning.configs.models import CNNConfig
from continual_learning.data import ContinualLearningDataset, get_dataset
from continual_learning.models import get_model
from continual_learning.optim import get_optimizer
from continual_learning.types import LogDict
from continual_learning.utils.monitoring import (
    Logger,
    compute_srank,
    get_dormant_neuron_logs,
    get_linearised_neuron_logs,
    prefix_dict,
    pytree_histogram,
)
from continual_learning.utils.training import TrainState
from continual_learning.utils.nn import flatten_last

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
                "training": train_cfg,
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
                save_decision_policy=ocp_mgrs.AnySavePolicy(
                    [
                        ocp_mgrs.FixedIntervalPolicy(logs_cfg.save_interval),
                        ocp_mgrs.save_decision_policy.PreemptionCheckpointingPolicy(),
                    ]
                ),
                preservation_policy=ocp_mgrs.AnyPreservationPolicy(
                    [
                        ocp_mgrs.LatestN(n=1),
                        ocp_mgrs.EveryNSeconds(60 * 60),  # Hourly checkpoints
                        ocp_mgrs.BestN(  # Top 3
                            n=3,
                            get_metric_fn=lambda x: x["metrics/test_accuracy"],
                            reverse=True,
                        ),
                    ]
                ),
            ),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("network_state", "key"))
    def update_network(
        network_state: TrainState, key: PRNGKeyArray, x: jax.Array, y: jax.Array
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        del network_state, key, x, y
        raise NotImplementedError

    def save(self, dataloader: grain.DataLoader, metrics: dict[str, float] | None):
        del dataloader, metrics
        return  # TODO: Fix checkpointing
        # self.ckpt_mgr.save(
        #     self.total_steps,
        #     args=ocp.args.Composite(
        #         nn=ocp.args.StandardSave(self.network),
        #         key=ocp.args.JaxRandomKeySave(self.key),
        #         dataloader=grain.PyGrainCheckpointSave(dataloader),
        #         dataset=ocp.args.JsonSave(self.dataset.state),
        #     ),
        #     metrics=metrics,
        # )

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
        self.dataset.load(ckpt["dataset"], resumed_loader=ckpt["dataloader"])
        self.total_steps = step + 1

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
        self.ckpt_mgr.close()


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

        activations_full = intermediates["activations"]
        activations = jax.tree.map(flatten_last, activations_full)

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

        network_state = network_state.apply_gradients(grads=grads, features=activations_full)
        network_params_flat, _ = jax.flatten_util.ravel_pytree(network_state.params["params"])
        network_param_hist_dict = pytree_histogram(network_state.params["params"])

        optim_logs = network_state.opt_state["reset_method"].logs  # pyright: ignore[reportAttributeAccessIssue,reportIndexIssue]

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
                **prefix_dict("optimizer", optim_logs),
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

        # Batch-level task mask (union of labels in the batch).
        # If you mix tasks within a batch, provide a per-example mask instead.
        active_labels = jnp.any(y.astype(bool), axis=0)  # [K]
        loss_mask = jnp.broadcast_to(active_labels, y.shape)  # [B, K]

        def loss_fn(params):
            logits, intermediates = network_state.apply_fn(
                params,
                x,
                training=True,
                rngs={"dropout": dropout_key},
                mutable=("activations", "preactivations"),
            )
            # Mask the denominator by setting inactive logits to -inf.
            masked_logits = jnp.where(loss_mask, logits, -jnp.inf)
            assert isinstance(masked_logits, jax.Array)
            loss = optax.safe_softmax_cross_entropy(masked_logits, y).mean()
            # Return aux so value_and_grad(has_aux=True) works and we can log.
            return loss, (logits, intermediates)

        (loss, (logits, intermediates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            network_state.params
        )

        # Compute accuracy with the same mask used in the loss.
        masked_logits = jnp.where(loss_mask, logits, -jnp.inf)
        assert isinstance(masked_logits, jax.Array)
        accuracy = jnp.mean(jnp.argmax(masked_logits, axis=-1) == jnp.argmax(y, axis=-1))

        # ----- logging -----
        activations_full = intermediates["activations"]
        activations = jax.tree_util.tree_map(flatten_last, activations_full)

        activations_hist_dict = pytree_histogram(activations)
        activations_flat = {
            k: v[0]  # pyright: ignore[reportIndexIssue], first sample's flattened features,
            for k, v in flax.traverse_util.flatten_dict(activations, sep="/").items()
        }
        dormant_neuron_logs = get_dormant_neuron_logs(activations_flat)
        srank_logs = jax.tree_util.tree_map(compute_srank, activations_flat)

        preactivations = intermediates["preactivations"]
        preactivations_flat = {
            k: v[0]  # pyright: ignore[reportIndexIssue]
            for k, v in flax.traverse_util.flatten_dict(preactivations, sep="/").items()
        }
        linearised_neuron_logs = get_linearised_neuron_logs(preactivations_flat)

        # Flatten only the parameter grads for norms/histograms.
        grads_params = grads["params"]
        grads_flat, _ = jax.flatten_util.ravel_pytree(grads_params)
        grads_hist_dict = pytree_histogram(grads_params)

        # Keep features=activations_full for your optimizer’s feature use.
        network_state = network_state.apply_gradients(grads=grads, features=activations_full)
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
        if self.train_cfg.resume:
            self.load(step=self.train_cfg.resume_from_step)

        for _, task in enumerate(self.dataset.tasks):
            for step, batch in enumerate(task):
                x, y = batch
                self.network, self.key, logs = self.update_network(
                    self.network, self.key, x, y
                )
                self.logger.accumulate(logs)

                if step % self.logger.cfg.interval == 0:
                    self.logger.push(self.total_steps)

                metrics = None
                if (
                    self.logger.cfg.eval_during_training
                    and step % self.logger.cfg.eval_interval == 0
                ):
                    metrics = self.dataset.evaluate(self.network, forgetting=False)
                    self.logger.log(metrics, step=self.total_steps)

                self.save(dataloader=task, metrics=metrics)
                self.total_steps += 1

            self.logger.push(self.total_steps)  # Flush logger

            logs = self.dataset.evaluate(
                self.network, forgetting=self.logger.cfg.catastrophic_forgetting
            )
            self.logger.log(logs, step=self.total_steps)

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
    # optim_conf = RedoConfig(
    #         tx=AdamConfig(learning_rate=1e-3)
    #     )
    optim_conf = AdamConfig(learning_rate=1e-3)
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=CNNConfig(output_size=10),
        optim_cfg=optim_conf,
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
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name="split_cifar10_debug_1",
            wandb_entity="evangelos-ch",
            wandb_project="continual_learning",
            wandb_mode="disabled",
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")
