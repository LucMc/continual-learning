# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from continual_learning_2.configs.dataset import DatasetConfig
from continual_learning_2.data.base import (
    ClassIncrementalDataset,
    PermutedDataset,
    SplitDataset,
)
from continual_learning_2.types import DatasetItem


class ProcessMNIST(grain.MapTransform):
    flatten: bool

    def __init__(self, flatten: bool):
        self.flatten = flatten

    def map(self, element: dict) -> DatasetItem:
        x, y = element["image"], element["label"]
        x = np.array(x, dtype=np.float32) / 255
        if self.flatten:
            x = x.flatten()
        else:
            x = x[..., None]  # MNIST is (H,W) by default, make it (H,W,C)
        y_one_hot = np.zeros(10, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class MNIST:
    flatten: bool = True

    def __init__(self, flatten: bool):
        self.flatten = flatten
        self.data_label = "label"

    @property
    def operations(self):
        return [ProcessMNIST(self.flatten)]

    @property
    def spec(self) -> jax.ShapeDtypeStruct:
        # fmt: off
        if self.flatten:
            return jax.ShapeDtypeStruct((1, 28 * 28), dtype=jnp.float32)
        else:
            return jax.ShapeDtypeStruct((1, 1, 28, 28), dtype=jnp.float32)
        # fmt: on


class SplitMNIST(MNIST, SplitDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    KEEP_IN_MEMORY: bool | None = True

    def __init__(self, config: DatasetConfig, flatten: bool = True):
        SplitDataset.__init__(self, config)
        MNIST.__init__(self, flatten)


class PermutedMNIST(MNIST, PermutedDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    KEEP_IN_MEMORY: bool | None = True
    DATA_DIM: int = 28 * 28

    def __init__(self, config: DatasetConfig, flatten: bool = True):
        PermutedDataset.__init__(self, config)
        MNIST.__init__(self, flatten)


class ClassIncrementalMNIST(MNIST, ClassIncrementalDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    KEEP_IN_MEMORY: bool | None = True

    def __init__(self, config: DatasetConfig, flatten: bool = True):
        ClassIncrementalDataset.__init__(self, config)
        MNIST.__init__(self, flatten)
