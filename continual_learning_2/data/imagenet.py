# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import grain.python as grain
import numpy as np
import datasets

from continual_learning_2.configs import DatasetConfig
from continual_learning_2.data.base import SplitDataset
from continual_learning_2.types import DatasetItem


class ProcessImageNet(grain.MapTransform):
    ... # TODO


class ContinualImageNet(SplitDataset):
    NUM_CLASSES: int = 1000
    DATASET_PATH = "ILSVRC/imagenet-1k"
    KEEP_IN_MEMORY: bool | None = False

    def __init__(self, config: DatasetConfig):
        self.num_tasks = config.num_tasks
        self.num_epochs = config.num_epochs_per_task
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.rng = np.random.default_rng(self.seed)
        self.num_workers = config.num_workers
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH, keep_in_memory=False, streaming=True, use_auth_token=True, trust_remote_code=True
        ).with_format("numpy")
        assert isinstance(self.dataset, datasets.IterableDatasetDict)

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

        self._tasks = [self.rng.choice(self.NUM_CLASSES, size=2, replace=False) for _ in range(self.num_tasks)]

    def _get_task(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        classes_in_task = self._tasks[task_id]
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

        classes_in_task = self._tasks[task_id]
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
