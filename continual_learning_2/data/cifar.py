# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
from typing import Generator

import datasets
import grain.python as grain
import numpy as np

from continual_learning_2.data.base import ContinualLearningDataset
from continual_learning_2.types import DatasetItem, LogDict, PredictorModel


class ProcessCIFAR(grain.MapTransform):
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def map(self, element: dict) -> DatasetItem:
        x, y = element["img"], element["label"]
        x = np.array(x, dtype=np.int32)
        y_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class SplitCIFAR10(ContinualLearningDataset):
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
        self.dataset = datasets.load_dataset("cifar10").with_format("numpy")
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
            accuracies.append((pred.argmax(axis=1) == y.argmax(axis=1)).mean().item())
        return float(np.mean(accuracies))

    def _get_task(self, task_id: int) -> grain.IterDataset:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        num_classes_in_task = self.NUM_CLASSES // self.num_tasks
        classes_in_task = list(
            range(num_classes_in_task * task_id, num_classes_in_task * (task_id + 1))
        )

        return (
            self._train_loader.filter(lambda x: x["label"] in classes_in_task)
            .map(ProcessCIFAR(self.NUM_CLASSES))
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
            .map(ProcessCIFAR(self.NUM_CLASSES))
            .shuffle()
            .to_iter_dataset()
            .batch(self.batch_size)
        )


class SplitCIFAR100(SplitCIFAR10):
    CURRENT_TASK: int
    NUM_CLASSES: int = 100

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
        self.dataset = datasets.load_dataset("cifar100").with_format("numpy")
        assert isinstance(self.dataset, datasets.DatasetDict)

        self._train_loader = grain.MapDataset.source(self.dataset["train"]).seed(self.seed)
        self._test_loader = grain.MapDataset.source(self.dataset["test"]).seed(self.seed)
