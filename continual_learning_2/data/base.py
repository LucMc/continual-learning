# pyright: reportArgumentType=false
import abc
from typing import Generator

import datasets
import grain.python as grain
import numpy as np

from continual_learning_2.types import LogDict, PredictorModel


class ContinualLearningDataset(abc.ABC):
    NUM_CLASSES: int

    @abc.abstractmethod
    def __init__(
        self, seed: int, num_tasks: int = 5, num_epochs: int = 20, batch_size: int = 32
    ): ...

    @abc.abstractproperty
    def tasks(self) -> Generator[grain.DataLoader, None, None]: ...

    @abc.abstractmethod
    def evaluate(self, model: PredictorModel, forgetting: bool = False) -> LogDict: ...


class SplitDataset(ContinualLearningDataset):
    CURRENT_TASK: int
    NUM_CLASSES: int

    DATASET_PATH: str
    OPERATIONS: list[grain.Transformation] = []

    def __init__(self, seed: int, num_tasks: int, num_epochs: int, batch_size: int):
        if not self.NUM_CLASSES % num_tasks == 0:
            raise ValueError(
                f"Number of classes ({self.NUM_CLASSES}) must be divisible by number of tasks ({num_tasks})."
            )
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.dataset = datasets.load_dataset(self.DATASET_PATH).with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

    @property
    def tasks(self) -> Generator[grain.DataLoader, None, None]:
        for task_id in range(self.num_tasks):
            self.CURRENT_TASK = task_id
            yield self._get_task(task_id)

    def evaluate(self, model: PredictorModel, forgetting: bool = False) -> LogDict:
        metrics = {}
        if forgetting:
            for task in range(self.CURRENT_TASK):
                test_set = self._get_task_test(task)
                metrics[f"task_{task}_accuracy"] = self._eval_task(model, test_set)
        metrics["accuracy"] = self._eval_task(model, self._get_task_test(self.CURRENT_TASK))
        metrics[f"task_{self.CURRENT_TASK}_accuracy"] = metrics["accuracy"]
        return metrics

    def _eval_task(self, model: PredictorModel, test_set: grain.DataLoader) -> float:
        accuracies = []
        for data in test_set:
            x, y = data
            pred = model(x)
            accuracies.append((pred.argmax(axis=1) == y.argmax(axis=1)).mean().item())
        return float(np.mean(accuracies))

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
                shuffle=True,
                seed=self.seed,
            ),
            operations=[
                *self.OPERATIONS,
                grain.Batch(batch_size=self.batch_size),
            ],
        )
