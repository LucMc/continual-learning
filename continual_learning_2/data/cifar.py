# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import grain.python as grain
import numpy as np

from continual_learning_2.data.base import ClassIncrementalDataset, SplitDataset
from continual_learning_2.types import DatasetItem


class ProcessCIFAR(grain.MapTransform):
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def map(self, element: dict) -> DatasetItem:
        x, y = element["img"], element["label"]
        x = np.array(x, dtype=np.int32)
        y_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class SplitCIFAR10(SplitDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH = "cifar10"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessCIFAR(10)]


class ClassIncrementalCIFAR10(ClassIncrementalDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH = "cifar10"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessCIFAR(10)]


class SplitCIFAR100(SplitDataset):
    NUM_CLASSES: int = 100
    DATASET_PATH = "cifar100"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessCIFAR(100)]


class ClassIncrementalCIFAR100(ClassIncrementalDataset):
    NUM_CLASSES: int = 100
    DATASET_PATH = "cifar100"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessCIFAR(100)]
