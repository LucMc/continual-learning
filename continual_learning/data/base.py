# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import abc
from typing import Generator

import datasets
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxtyping import Array, Int32

from continual_learning.configs import DatasetConfig
from continual_learning.types import DatasetItem
from continual_learning.utils.monitoring import accumulate_metrics, prefix_dict


class ContinualLearningDataset(abc.ABC):
    NUM_CLASSES: int

    @abc.abstractmethod
    def __init__(self, config: DatasetConfig): ...

    @property
    @abc.abstractmethod
    def state(self) -> dict: ...

    @abc.abstractmethod
    def load(self, state: dict, resumed_loader: grain.DataLoader) -> None: ...

    @property
    @abc.abstractmethod
    def tasks(self) -> Generator[grain.DataLoader, None, None]: ...

    @property
    @abc.abstractmethod
    def operations(self) -> list[grain.Transformation]: ...

    @abc.abstractmethod
    def evaluate(self, model: TrainState, forgetting: bool = False) -> dict[str, float]: ...

    @property
    @abc.abstractmethod
    def spec(self) -> jax.ShapeDtypeStruct: ...


# @jax.jit
# def _eval_model(model: TrainState, x: jax.Array, y: jax.Array) -> dict[str, float]:
#     logits = model.apply_fn(model.params, x, training=False)
#     loss = optax.softmax_cross_entropy(logits, y).mean()
#     accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y.argmax(axis=-1))
#     return {"eval_loss": loss, "eval_accuracy": accuracy} # Removed float() as this breaks mnist


@jax.jit
def _eval_model(
    model: TrainState,
    x: jax.Array,
    y: jax.Array,
    class_mask: jax.Array | None = None,
) -> dict[str, float]:
    logits = model.apply_fn(model.params, x, training=False)

    ci_loss = optax.softmax_cross_entropy(logits, y).mean()
    ci_acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))

    out = {"eval_loss_ci": ci_loss, "eval_accuracy_ci": ci_acc}

    if class_mask is not None:
        loss_mask = jnp.broadcast_to(class_mask, y.shape)  # [B, K]
        masked_logits = jnp.where(loss_mask, logits, -jnp.inf)
        ta_loss = optax.safe_softmax_cross_entropy(masked_logits, y).mean()
        ta_acc = jnp.mean(jnp.argmax(masked_logits, axis=-1) == jnp.argmax(y, axis=-1))
        out.update({"eval_loss_task": ta_loss, "eval_accuracy_task": ta_acc})

    return out


class SplitDataset(ContinualLearningDataset):
    current_task: int
    data_label: str

    NUM_CLASSES: int
    KEEP_IN_MEMORY: bool | None = None

    DATASET_PATH: str

    def __init__(self, config: DatasetConfig):
        if not self.NUM_CLASSES % config.num_tasks == 0:
            raise ValueError(
                f"Number of classes ({self.NUM_CLASSES}) must be divisible by number of tasks ({config.num_tasks})."
            )
        self.num_tasks = config.num_tasks
        self.num_epochs = config.num_epochs_per_task
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.num_workers = config.num_workers
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH, keep_in_memory=self.KEEP_IN_MEMORY
        ).with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        self.current_task = 0
        self.resumed_loader = None
        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

    @property
    def tasks(self) -> Generator[grain.DataLoader, None, None]:
        for task_id in range(self.current_task, self.num_tasks):
            self.current_task = task_id

            if self.resumed_loader is not None:
                yield self.resumed_loader
                self.resumed_loader = None
            else:
                yield self._get_task(task_id)

    @property
    def state(self) -> dict:
        return {
            "current_task": self.current_task,
        }

    def load(self, state: dict, resumed_loader: grain.DataLoader) -> None:
        self.current_task = state["current_task"]
        self.resumed_loader = resumed_loader

    def evaluate(self, model: TrainState, forgetting: bool = False) -> dict[str, float]:
        metrics = {}

        if forgetting:
            for task in range(self.current_task):
                test_set = self._get_task_test(task)
                task_mask = self._task_class_mask(task)
                task_metrics = self._eval_task(model, test_set, class_mask=task_mask)
                metrics.update(prefix_dict(f"metrics/task_{task}", task_metrics))

        print(f"- Evaluating on task {self.current_task}")
        task_mask = self._task_class_mask(self.current_task)
        latest = self._eval_task(
            model, self._get_task_test(self.current_task), class_mask=task_mask
        )

        metrics.update(
            prefix_dict(
                "metrics",
                {
                    "eval_loss": latest["eval_loss_task"],
                    "eval_accuracy": latest["eval_accuracy_task"],
                    "eval_loss_ci": latest["eval_loss_ci"],
                    "eval_accuracy_ci": latest["eval_accuracy_ci"],
                },
            )
        )

        metrics.update(prefix_dict(f"metrics/task_{self.current_task}", latest))
        return metrics

    def _task_class_mask(self, task_id: int) -> jax.Array:
        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes = jnp.arange(
            num_classes_in_task * task_id, num_classes_in_task * (task_id + 1)
        )
        mask = jnp.zeros((self.NUM_CLASSES,), dtype=bool).at[classes].set(True)
        return mask

    def _eval_task(
        self,
        model: TrainState,
        test_set: grain.DataLoader,
        class_mask: jax.Array | None = None,
    ) -> dict[str, float]:
        logs = []
        for data in test_set:
            x, y = data
            logs.append(_eval_model(model, x, y, class_mask))
        return accumulate_metrics(logs)

    def _get_task(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes_in_task = list(
            range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1))
        )

        ds = self._dataset_train.filter(lambda x: x[self.data_label] in classes_in_task)

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=True,
                num_epochs=self.num_epochs,
                seed=self.seed,
            ),
            operations=[
                *self.operations,
                grain.Batch(batch_size=self.batch_size, drop_remainder=True),
            ],
            worker_count=self.num_workers,
        )

    def _get_task_test(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes_in_task = list(
            range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1))
        )

        ds = self._dataset_test.filter(lambda x: x[self.data_label] in classes_in_task)

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds), shuffle=False, num_epochs=1, seed=self.seed
            ),
            operations=[
                *self.operations,
                grain.Batch(batch_size=self.batch_size, drop_remainder=False),
            ],
            worker_count=self.num_workers,
        )


class PermuteInputs(grain.MapTransform):
    permutation: Int32[Array, " data_dim"]

    def __init__(self, permutation: Int32[Array, " data_dim"]):
        self.permutation = permutation

    def map(self, element: DatasetItem) -> DatasetItem:
        x, y = element
        x = x.reshape(-1)[self.permutation].reshape(x.shape)
        return x, y


class PermutedDataset(ContinualLearningDataset):
    current_task: int

    NUM_CLASSES: int
    DATA_DIM: int
    DATASET_PATH: str

    def __init__(self, config: DatasetConfig):
        self.num_tasks = config.num_tasks
        self.num_epochs = config.num_epochs_per_task
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.num_workers = config.num_workers
        self.dataset = datasets.load_dataset(self.DATASET_PATH).with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        rng = np.random.default_rng(self.seed)
        self.permutations = [rng.permutation(self.DATA_DIM) for _ in range(self.num_tasks)]
        self.seeds = [rng.integers(0, 99999) for _ in range(self.num_tasks)]
        self.current_task = 0
        self.resumed_loader = None

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

    @property
    def state(self) -> dict:
        return {
            "current_task": self.current_task,
        }

    def load(self, state: dict, resumed_loader: grain.DataLoader) -> None:
        self.current_task = state["current_task"]
        self.resumed_loader = resumed_loader

    @property
    def tasks(self) -> Generator[grain.DataLoader, None, None]:
        for task_id in range(self.current_task, self.num_tasks):
            self.current_task = task_id

            if self.resumed_loader is not None:
                yield self.resumed_loader
                self.resumed_loader = None
            else:
                yield self._get_task(task_id)

    def evaluate(self, model: TrainState, forgetting: bool = False) -> dict[str, float]:
        metrics = {}
        if forgetting:
            for task in range(self.current_task):
                test_set = self._get_task_test(task)
                print(f"- Evaluating on task {task}")
                metrics.update(
                    prefix_dict(f"metrics/task_{task}", self._eval_task(model, test_set))
                )
        print(f"- Evaluating on task {self.current_task}")
        latest_metrics = self._eval_task(model, self._get_task_test(self.current_task))
        metrics.update(prefix_dict("metrics", latest_metrics))
        metrics.update(prefix_dict(f"metrics/task_{self.current_task}", latest_metrics))
        return metrics

    def _eval_task(self, model: TrainState, test_set: grain.DataLoader) -> dict[str, float]:
        logs = []
        for data in test_set:
            x, y = data
            logs.append(_eval_model(model, x, y))

        return accumulate_metrics(logs)

    def _get_task(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        permutation = self.permutations[task_id]
        seed = self.seeds[task_id]

        ds = self._dataset_train

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=True,
                num_epochs=self.num_epochs,
                seed=int(seed),
            ),
            operations=[
                *self.operations,
                PermuteInputs(permutation),
                grain.Batch(batch_size=self.batch_size, drop_remainder=True),
            ],
            worker_count=self.num_workers,
        )

    def _get_task_test(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")
        permutation = self.permutations[task_id]
        seed = self.seeds[task_id]

        ds = self._dataset_test

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=True,
                seed=int(seed),
                num_epochs=1,
            ),
            operations=[
                *self.operations,
                PermuteInputs(permutation),
                grain.Batch(batch_size=self.batch_size),
            ],
            worker_count=self.num_workers,
        )


class ClassIncrementalDataset(SplitDataset):
    def __init__(self, config: DatasetConfig):
        if not config.num_tasks <= self.NUM_CLASSES:
            raise ValueError(
                f"Number of tasks ({config.num_tasks}) must be less than or equal to the number of classes ({self.NUM_CLASSES})."
            )
        if not self.NUM_CLASSES % config.num_tasks == 0:
            raise ValueError(
                f"Number of classes ({self.NUM_CLASSES}) must be divisible by the number of tasks ({config.num_tasks})."
            )
        self.num_tasks = config.num_tasks
        self.num_epochs = config.num_epochs_per_task
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.num_workers = config.num_workers
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH, keep_in_memory=self.KEEP_IN_MEMORY
        ).with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        rng = np.random.default_rng(self.seed)
        self.class_increment = self.NUM_CLASSES // self.num_tasks
        self.class_order = rng.permutation(self.NUM_CLASSES)
        self.current_task = 0
        self.resumed_loader = None

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

    def _cumulative_class_mask(self, task_id: int) -> jax.Array:
        """Create mask for all classes seen up to and including task_id."""
        num_classes_seen = self.class_increment * (task_id + 1)
        classes_seen = self.class_order[:num_classes_seen]
        mask = jnp.zeros((self.NUM_CLASSES,), dtype=bool).at[classes_seen].set(True)
        return mask

    def evaluate(self, model: TrainState, forgetting: bool = False) -> dict[str, float]:
        metrics = {}

        if forgetting:
            for task in range(self.current_task):
                test_set = self._get_task_test(task)
                task_mask = self._cumulative_class_mask(task)
                task_metrics = self._eval_task(model, test_set, class_mask=task_mask)
                metrics.update(prefix_dict(f"metrics/task_{task}", task_metrics))

        print(f"- Evaluating on task {self.current_task}")
        task_mask = self._cumulative_class_mask(self.current_task)
        latest = self._eval_task(
            model, self._get_task_test(self.current_task), class_mask=task_mask
        )

        metrics.update(
            prefix_dict(
                "metrics",
                {
                    "eval_loss": latest["eval_loss_task"],
                    "eval_accuracy": latest["eval_accuracy_task"],
                    "eval_loss_ci": latest["eval_loss_ci"],
                    "eval_accuracy_ci": latest["eval_accuracy_ci"],
                },
            )
        )

        # Also store the full set under the task-specific prefix
        metrics.update(prefix_dict(f"metrics/task_{self.current_task}", latest))
        return metrics

    def _get_task(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes = self.class_increment * (task_id + 1)
        ds = self._dataset_train.filter(
            lambda x: x[self.data_label] in self.class_order[:num_classes]
        )

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=True,
                num_epochs=self.num_epochs,
                seed=self.seed,
            ),
            operations=[
                *self.operations,
                grain.Batch(batch_size=self.batch_size, drop_remainder=True),
            ],
            worker_count=self.num_workers,
        )

    def _get_task_test(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        ds = self._dataset_test

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds), shuffle=False, num_epochs=1, seed=self.seed
            ),
            operations=[
                *self.operations,
                grain.Batch(batch_size=self.batch_size, drop_remainder=False),
            ],
            worker_count=self.num_workers,
        )
