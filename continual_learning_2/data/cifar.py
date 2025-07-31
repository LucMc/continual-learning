# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
from typing import Any

import dm_pix as pix
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from continual_learning_2.configs.dataset import DatasetConfig
from continual_learning_2.data.base import ClassIncrementalDataset, SplitDataset
from continual_learning_2.types import DatasetItem


class ProcessCIFAR(grain.RandomMapTransform):
    num_classes: int
    flatten: bool

    def __init__(self, num_classes: int = 10, flatten: bool = False):
        self.num_classes = num_classes
        self.flatten = flatten

    @staticmethod
    @jax.jit
    def process_image(seed: int, image: np.ndarray) -> Any:
        """Follows the transforms in https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/incremental_cifar/incremental_cifar_experiment.py#L323"""
        keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        x = pix.random_flip_left_right(keys[0], image, probability=0.5)
        x = pix.pad_to_size(x, target_height=32 + 4, target_width=32 + 4, mode="reflect")
        x = pix.random_crop(keys[1], x, crop_sizes=(32, 32, 3))
        x = pix.rotate(x, angle=jax.random.randint(keys[2], shape=(), minval=0, maxval=15))
        return x

    def random_map(self, element: dict, rng: np.random.Generator) -> DatasetItem:
        x, y = element["img"], element["fine_label"]

        # Max normalisation
        x = np.array(x, dtype=np.float32) / 255

        # Standardisation
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        mean, std = np.broadcast_to(mean, x.shape), np.broadcast_to(std, x.shape)
        x = x - mean / std

        # Misc transforms
        x = self.process_image(rng.integers(0, np.iinfo(np.int64).max), x)

        if self.flatten:
            x = x.flatten()

        y_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot  # pyright: ignore[reportReturnType]


class CIFAR:
    num_classes: int
    flatten: bool = False

    def __init__(self, num_classes: int, flatten: bool = False, data_label: str = "label"):
        self.num_classes = num_classes
        self.flatten = flatten
        self.data_label = data_label

    @property
    def operations(self):
        return [ProcessCIFAR(self.num_classes, self.flatten)]

    @property
    def spec(self) -> jax.ShapeDtypeStruct:
        # fmt: off
        if self.flatten:
            return jax.ShapeDtypeStruct((1, 32 * 32), dtype=jnp.float32)
        else:
            return jax.ShapeDtypeStruct((1, 32, 32, 3), dtype=jnp.float32)
        # fmt: on


class SplitCIFAR10(CIFAR, SplitDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH = "cifar10"
    KEEP_IN_MEMORY: bool | None = True
    DATA_LABEL: str = "label"

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        SplitDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten, self.DATA_LABEL)


class ClassIncrementalCIFAR10(CIFAR, ClassIncrementalDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH = "cifar10"
    KEEP_IN_MEMORY: bool | None = True
    DATA_LABEL: str = "label"

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        ClassIncrementalDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten, self.DATA_LABEL)


class SplitCIFAR100(CIFAR, SplitDataset):
    NUM_CLASSES: int = 100
    DATASET_PATH = "cifar100"
    KEEP_IN_MEMORY: bool | None = True
    DATA_LABEL: str = "fine_label" # As oposed to corse_label

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        SplitDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten, self.DATA_LABEL)


class ClassIncrementalCIFAR100(CIFAR, ClassIncrementalDataset):
    NUM_CLASSES: int = 100
    DATASET_PATH = "cifar100"
    KEEP_IN_MEMORY: bool | None = True 
    DATA_LABEL: str = "fine_label"

    def __init__(self, config: DatasetConfig, flatten: bool = False):
        ClassIncrementalDataset.__init__(self, config)
        CIFAR.__init__(self, self.NUM_CLASSES, flatten, self.DATA_LABEL)
