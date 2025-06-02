from flax import struct
from flax.core import FrozenDict
from jaxtyping import (
    Array,
    Float,
    PRNGKeyArray,
    PyTree,
)
from flax.training.train_state import TrainState
from typing import Literal
from chex import dataclass
import optax

