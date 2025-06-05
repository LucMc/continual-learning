# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import grain.python as grain
import numpy as np

from continual_learning_2.data.base import (
    PermutedDataset,
    SplitDataset,
)
from continual_learning_2.types import DatasetItem


class ProcessMNIST(grain.MapTransform):
    def map(self, element: dict) -> DatasetItem:
        x, y = element["image"], element["label"]
        x = np.array(x, dtype=np.int32)
        y_one_hot = np.zeros(10, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class SplitMNIST(SplitDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    OPERATIONS = [ProcessMNIST()]


class PermutedMNIST(PermutedDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    OPERATIONS = [ProcessMNIST()]
    DATA_DIM: int = 28 * 28
