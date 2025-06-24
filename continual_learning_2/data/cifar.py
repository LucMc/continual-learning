# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from continual_learning_2.configs.dataset import DatasetConfig
from continual_learning_2.data.base import ClassIncrementalDataset, SplitDataset
from continual_learning_2.types import DatasetItem


class ProcessCIFAR(grain.MapTransform):
    num_classes: int
    flatten: bool

    def __init__(self, num_classes: int = 10, flatten: bool = False):
        self.num_classes = num_classes
        self.flatten = flatten

    def map(self, element: dict) -> DatasetItem:
        x, y = element["img"], element["label"]
        x = np.array(x, dtype=np.float32) / 255
        if self.flatten:
            x = x.flatten()
        y_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class CIFAR:
    num_classes: int
    flatten: bool = False

    def __init__(self, num_classes: int, flatten: bool = False):
        self.num_classes = num_classes
        self.flatten = flatten

    @property
    def operations(self):
        return [ProcessCIFAR(self.num_classes, self.flatten)]

    @property
    def spec(self) -> jax.ShapeDtypeStruct:
        # fmt: off
        if self.flatten:
            return jax.ShapeDtypeStruct((1, 32 * 32), dtype=jnp.float32)
        else:
            return jax.ShapeDtypeStruct((1, 3, 32, 32), dtype=jnp.float32)
        # fmt: on


class SplitCIFAR10(CIFAR, SplitDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH = "cifar10"
    KEEP_IN_MEMORY: bool | None = True

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        SplitDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten)


class ClassIncrementalCIFAR10(CIFAR, ClassIncrementalDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH = "cifar10"
    KEEP_IN_MEMORY: bool | None = True

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        ClassIncrementalDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten)


class SplitCIFAR100(CIFAR, SplitDataset):
    NUM_CLASSES: int = 100
    DATASET_PATH = "cifar100"
    KEEP_IN_MEMORY: bool | None = True

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        SplitDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten)


class ClassIncrementalCIFAR100(CIFAR, ClassIncrementalDataset):
    NUM_CLASSES: int = 100
    DATASET_PATH = "cifar100"
    KEEP_IN_MEMORY: bool | None = True

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        ClassIncrementalDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten)
