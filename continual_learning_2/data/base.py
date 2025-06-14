# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import abc
import sys
from typing import Generator

import datasets
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxtyping import Array, Int32

from continual_learning_2.configs import DatasetConfig
from continual_learning_2.types import DatasetItem, LogDict, PredictorModel
from continual_learning_2.utils.monitoring import accumulate_metrics, prefix_dict


class ContinualLearningDataset(abc.ABC):
    NUM_CLASSES: int

    @abc.abstractmethod
    def __init__(self, config: DatasetConfig): ...

    @abc.abstractproperty
    def tasks(self) -> Generator[grain.DataLoader, None, None]: ...

    @abc.abstractmethod
    def evaluate(self, model: TrainState, forgetting: bool = False) -> LogDict: ...

    @abc.abstractproperty
    def spec(self) -> jax.ShapeDtypeStruct: ...


@jax.jit
def _eval_model(model: TrainState, x: jax.Array, y: jax.Array) -> LogDict:
    logits = model.apply_fn(model.params, x, training=False)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y.argmax(axis=-1))
    return {"eval_loss": loss, "eval_accuracy": accuracy}


class SplitDataset(ContinualLearningDataset):
    CURRENT_TASK: int
    NUM_CLASSES: int
    KEEP_IN_MEMORY: bool | None = None

    DATASET_PATH: str
    OPERATIONS: list[grain.Transformation] = []

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

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

    @property
    def tasks(self) -> Generator[grain.DataLoader, None, None]:
        for task_id in range(self.num_tasks):
            self.CURRENT_TASK = task_id
            yield self._get_task(task_id)

    def evaluate(self, model: TrainState, forgetting: bool = False) -> LogDict:
        metrics = {}
        if forgetting:
            for task in range(self.CURRENT_TASK):
                test_set = self._get_task_test(task)
                print(f"- Evaluating on task {task}")
                metrics.update(
                    prefix_dict(f"metrics/task_{task}", self._eval_task(model, test_set))
                )
        print(f"- Evaluating on task {self.CURRENT_TASK}")
        latest_metrics = self._eval_task(model, self._get_task_test(self.CURRENT_TASK))
        metrics.update(prefix_dict("metrics", latest_metrics))
        metrics.update(prefix_dict(f"metrics/task_{self.CURRENT_TASK}", latest_metrics))
        return metrics

    def _eval_task(self, model: TrainState, test_set: grain.DataLoader) -> LogDict:
        logs = []
        for data in test_set:
            x, y = data
            logs.append(_eval_model(model, x, y))

        return accumulate_metrics(logs)

    def _get_task(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes_in_task = list(
            range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1))
        )

        ds = self._dataset_train.filter(lambda x: x["label"] in classes_in_task)

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=True,
                num_epochs=self.num_epochs,
                seed=self.seed,
            ),
            operations=[
                *self.OPERATIONS,
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

        ds = self._dataset_test.filter(lambda x: x["label"] in classes_in_task)

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=False,
                num_epochs=1,
            ),
            operations=[
                *self.OPERATIONS,
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
    CURRENT_TASK: int
    NUM_CLASSES: int

    DATA_DIM: int

    DATASET_PATH: str
    OPERATIONS: list[grain.Transformation] = []

    def __init__(self, config: DatasetConfig):
        self.num_tasks = config.num_tasks
        self.num_epochs = config.num_epochs_per_task
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.num_workers = config.num_workers
        self.dataset = datasets.load_dataset(self.DATASET_PATH).with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        self.rng = np.random.default_rng(self.seed)
        self.permutations = [
            self.rng.permutation(self.DATA_DIM) for _ in range(self.num_tasks)
        ]
        self.seeds = [self.rng.integers(0, sys.maxsize) for _ in range(self.num_tasks)]

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

    @property
    def tasks(self) -> Generator[grain.DataLoader, None, None]:
        for task_id in range(self.num_tasks):
            self.CURRENT_TASK = task_id
            yield self._get_task(task_id)

    def evaluate(self, model: TrainState, forgetting: bool = False) -> LogDict:
        metrics = {}
        if forgetting:
            for task in range(self.CURRENT_TASK):
                test_set = self._get_task_test(task)
                print(f"- Evaluating on task {task}")
                metrics.update(
                    prefix_dict(f"metrics/task_{task}", self._eval_task(model, test_set))
                )
        print(f"- Evaluating on task {self.CURRENT_TASK}")
        latest_metrics = self._eval_task(model, self._get_task_test(self.CURRENT_TASK))
        metrics.update(prefix_dict("metrics", latest_metrics))
        metrics.update(prefix_dict(f"metrics/task_{self.CURRENT_TASK}", latest_metrics))
        return metrics

    def _eval_task(self, model: TrainState, test_set: grain.DataLoader) -> LogDict:
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
                seed=seed,
            ),
            operations=[
                *self.OPERATIONS,
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
                seed=seed,
            ),
            operations=[
                *self.OPERATIONS,
                PermuteInputs(permutation),
                grain.Batch(batch_size=self.batch_size),
            ],
            worker_count=self.num_workers,
        )
