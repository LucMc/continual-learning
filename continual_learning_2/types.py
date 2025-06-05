import enum
from typing import Callable, NamedTuple

import flax.linen
import flax.struct
import jax
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float

from continual_learning_2.utils.nn import Identity

Action = Float[np.ndarray, "... action_dim"]
Value = Float[np.ndarray, "... 1"]
LogProb = Float[np.ndarray, "... 1"]
Observation = Float[np.ndarray, "... obs_dim"]
type LogDict = dict[str, float | Float[Array, ""] | Histogram]
type Input = Float[np.ndarray, " ... *input_dim"]
type Label = Float[np.ndarray, " ... num_classes"]
type DatasetItem = tuple[Input, Label]

type PredictorModel = Callable[[DatasetItem], Label]


class Histogram(flax.struct.PyTreeNode):
    data: Float[npt.NDArray | Array, "..."] | None = None
    np_histogram: tuple | None = None


class Activation(enum.Enum):
    ReLU = enum.member(jax.nn.relu)
    Tanh = enum.member(jax.nn.tanh)
    LeakyReLU = enum.member(jax.nn.leaky_relu)
    PReLU = enum.member(lambda x: flax.linen.PReLU()(x))  # noqa: E731
    ReLU6 = enum.member(jax.nn.relu6)
    SiLU = enum.member(jax.nn.silu)
    GELU = enum.member(jax.nn.gelu)
    GLU = enum.member(jax.nn.glu)
    Identity = enum.member(lambda x: x)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class LayerNorm(enum.Enum):
    LayerNorm = enum.member(flax.linen.LayerNorm)
    RMSNorm = enum.member(flax.linen.RMSNorm)
    NONE = enum.member(Identity)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class LayerNormPosition(enum.Enum):
    PRE = enum.auto()
    POST = enum.auto()


class StdType(enum.Enum):
    MLP_HEAD = enum.auto()
    PARAM = enum.auto()


class Rollout(NamedTuple):
    # Standard timestep data
    observations: Float[Observation, "timestep task"]
    actions: Float[Action, "timestep task"]
    rewards: Float[np.ndarray, "timestep task 1"]
    dones: Float[np.ndarray, "timestep task 1"]

    # Auxiliary policy outputs
    log_probs: Float[LogProb, "timestep task"] | None = None
    means: Float[Action, "timestep task"] | None = None
    stds: Float[Action, "timestep task"] | None = None
    values: Float[np.ndarray, "timestep task 1"] | None = None

    # Computed statistics about observed rewards
    returns: Float[np.ndarray, "timestep task 1"] | None = None
    advantages: Float[np.ndarray, "timestep task 1"] | None = None
    valids: Float[np.ndarray, "episode timestep 1"] | None = None
