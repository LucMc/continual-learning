# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
from typing import Generator

import datasets
import grain.python as grain
import numpy as np
from jaxtyping import Array, Int32

from continual_learning_2.data.base import ContinualLearningDataset
from continual_learning_2.types import DatasetItem, LogDict, PredictorModel


class ProcessMNIST(grain.MapTransform):
    def map(self, element: dict) -> DatasetItem:
        x, y = element["image"], element["label"]
        x = np.array(x, dtype=np.int32)
        y_one_hot = np.zeros(10, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class PermuteMNIST(grain.MapTransform):
    permutation: Int32[Array, "28*28"]

    def __init__(self, permutation: Int32[Array, "28*28"]):
        self.permutation = permutation

    def map(self, element: DatasetItem) -> DatasetItem:
        x, y = element
        x = x.reshape(-1)[self.permutation].reshape(x.shape)
        return x, y


class SplitMNIST(ContinualLearningDataset):
    CURRENT_TASK: int
    NUM_CLASSES: int = 10

    def __init__(
        self, seed: int, num_tasks: int = 5, num_epochs: int = 20, batch_size: int = 32
    ):
        if not self.NUM_CLASSES % num_tasks == 0:
            raise ValueError(
                f"Number of classes ({self.NUM_CLASSES}) must be divisible by number of tasks ({num_tasks})."
            )
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.dataset = datasets.load_dataset("mnist").with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        self._train_loader = grain.MapDataset.source(self.dataset["train"]).seed(self.seed)
        self._test_loader = grain.MapDataset.source(self.dataset["test"]).seed(self.seed)

    @property
    def tasks(self) -> Generator[grain.IterDataset, None, None]:
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

    def _eval_task(self, model: PredictorModel, test_set: grain.IterDataset) -> float:
        accuracies = []
        for data in test_set:
            x, y = data
            pred = model(x)
            breakpoint()
            accuracies.append((pred.argmax(axis=1) == y.argmax(axis=1)).mean().item())
        return np.mean(accuracies)

    def _get_task(self, task_id: int) -> grain.IterDataset:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes_in_task = list(
            range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1))
        )

        return (
            self._train_loader.filter(lambda x: x["label"] in classes_in_task)
            .map(ProcessMNIST())
            .shuffle()
            .repeat(self.num_epochs)
            .to_iter_dataset()
            .batch(self.batch_size, drop_remainder=True)
        )

    def _get_task_test(self, task_id: int) -> grain.IterDataset:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes_in_task = list(
            range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1))
        )

        return (
            self._test_loader.filter(lambda x: x["label"] in classes_in_task)
            .map(ProcessMNIST())
            .shuffle()
            .to_iter_dataset()
            .batch(self.batch_size)
        )


class PermutedMNIST(ContinualLearningDataset):
    CURRENT_TASK: int
    NUM_CLASSES: int = 10

    def __init__(
        self, seed: int, num_tasks: int = 5, num_epochs: int = 20, batch_size: int = 32
    ):
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.dataset = datasets.load_dataset("mnist").with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        self.rng = np.random.default_rng(seed)
        self.permutations = [self.rng.permutation(28 * 28) for _ in range(self.num_tasks)]

        self._train_loader = grain.MapDataset.source(self.dataset["train"]).seed(self.seed)
        self._test_loader = grain.MapDataset.source(self.dataset["test"]).seed(self.seed)

    @property
    def tasks(self) -> Generator[grain.IterDataset, None, None]:
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

    def _eval_task(self, model: PredictorModel, test_set: grain.IterDataset) -> float:
        accuracies = []
        for data in test_set:
            x, y = data
            pred = model(x)
            accuracies.append((pred.argmax(axis=1) == y.argmax(axis=1)).mean().item())
        return np.mean(accuracies)

    def _get_task(self, task_id: int) -> grain.IterDataset:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        permutation = self.permutations[task_id]

        return (
            self._train_loader.map(ProcessMNIST())
            .map(PermuteMNIST(permutation))
            .shuffle()
            .repeat(self.num_epochs)
            .to_iter_dataset()
            .batch(self.batch_size, drop_remainder=True)
        )

    def _get_task_test(self, task_id: int) -> grain.IterDataset:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        permutation = self.permutations[task_id]

        return (
            self._test_loader.map(ProcessMNIST())
            .map(PermuteMNIST(permutation))
            .shuffle()
            .to_iter_dataset()
            .batch(self.batch_size)
        )
