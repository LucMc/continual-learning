# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import itertools

import datasets
import grain.python as grain
import numpy as np
from PIL import Image

from continual_learning.configs import DatasetConfig
from continual_learning.data.base import SplitDataset
from continual_learning.types import DatasetItem


class ProcessImageNet(grain.MapTransform):
    NUM_CLASSES = 1000

    def map(self, element) -> DatasetItem:
        x, y = element["image"], element["label"]
        x = x.resize((256, 256), resample=Image.Resampling.BILINEAR)

        # Normalise data between -1 and 1
        # That's how I've seen it in a few repositories and Gemini says
        # it should be better in general
        x = 2 * (np.array(x, dtype=np.float32) / 255) - 1

        y_one_hot = np.zeros(self.NUM_CLASSES, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class ContinualImageNet(SplitDataset):
    NUM_CLASSES: int = 1000
    DATASET_PATH = "ILSVRC/imagenet-1k"
    KEEP_IN_MEMORY: bool | None = False

    def __init__(self, config: DatasetConfig, num_classes_per_task: int = 2):
        self.num_tasks = config.num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.num_epochs = config.num_epochs_per_task
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.rng = np.random.default_rng(self.seed)
        self.num_workers = config.num_workers
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH,
            keep_in_memory=False,
            streaming=True,
            use_auth_token=True,
            trust_remote_code=True,
        ).with_format("numpy")
        assert isinstance(self.dataset, datasets.IterableDatasetDict)

        self._dataset_train, self._dataset_test = self.dataset["train"], self.dataset["test"]

        class_stream = self.rng.permutation(self.NUM_CLASSES).tolist() * (
            self.num_classes_per_task * self.num_tasks // self.NUM_CLASSES
        )
        # NOTE: Python 3.12+ only, but come on it's $CURRENT_YEAR
        self._tasks = list(itertools.batched(class_stream, self.num_classes_per_task))

    def _get_task(self, task_id: int) -> grain.DataLoader:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Invalid task id: {task_id}")

        ds = self._dataset_train.filter(lambda x: x["label"] in self._tasks[task_id])

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

        ds = self._dataset_test.filter(lambda x: x["label"] in self._tasks[task_id])

        return grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=False,
                num_epochs=1,
            ),
            operations=[
                *self.operations,
                grain.Batch(batch_size=self.batch_size, drop_remainder=False),
            ],
            worker_count=self.num_workers,
        )
