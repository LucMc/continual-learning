# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from continual_learning_2.data.base import (
    ClassIncrementalDataset,
    PermutedDataset,
    SplitDataset,
)
from continual_learning_2.types import DatasetItem


class ProcessMNIST(grain.MapTransform):
    def map(self, element: dict) -> DatasetItem:
        x, y = element["image"], element["label"]
        x = np.array(x, dtype=np.float32).flatten() / 255
        y_one_hot = np.zeros(10, dtype=np.float32)
        y_one_hot[y] = 1.0
        return x, y_one_hot


class SplitMNIST(SplitDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessMNIST()]

    @property
    def spec(self) -> jax.ShapeDtypeStruct:
        # fmt: off
        return jax.ShapeDtypeStruct((1, 28 * 28), dtype=jnp.float32)
        # fmt: on


class PermutedMNIST(PermutedDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessMNIST()]
    DATA_DIM: int = 28 * 28

    @property
    def spec(self) -> jax.ShapeDtypeStruct:
        # fmt: off
        return jax.ShapeDtypeStruct((1, 28 * 28), dtype=jnp.float32)
        # fmt: on

class ClassIncrementalMNIST(ClassIncrementalDataset):
    NUM_CLASSES: int = 10
    DATASET_PATH: str = "mnist"
    KEEP_IN_MEMORY: bool | None = True
    OPERATIONS = [ProcessMNIST()]

    @property
    def spec(self) -> jax.ShapeDtypeStruct:
        # fmt: off
        return jax.ShapeDtypeStruct((1, 28 * 28), dtype=jnp.float32)
        # fmt: on
