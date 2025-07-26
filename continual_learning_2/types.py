import enum
from typing import Callable, NamedTuple

import flax.linen
import flax.struct
import jax
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Bool, Float, Int, PyTree

from continual_learning_2.utils.nn import Identity

# RL
Action = Float[np.ndarray | Array, "... action_dim"]
Value = Float[np.ndarray | Array, "... 1"]
LogProb = Float[np.ndarray | Array, "... 1"]
Observation = Float[np.ndarray | Array, "... obs_dim"]
Reward = Float[np.ndarray | Array, "... 1"]
Done = Bool[np.ndarray | Array, "... 1"]
EpisodeStarted = Bool[np.ndarray | Array, "... 1"]
EpisodeLengths = Int[np.ndarray | Array, "... 1"]
EpisodeReturns = Float[np.ndarray | Array, "... 1"]
EnvState = PyTree
type LogDict = dict[str, float | Float[Array, ""] | Histogram]

# Supervised Learning
type Input = Float[np.ndarray, " ... *input_dim"]
type Label = Float[np.ndarray, " ... num_classes"]
type DatasetItem = tuple[Input, Label]

type PredictorModel = Callable[[DatasetItem], Label]

LayerActivations = Float[Array, "batch_size layer_dim"]
type LayerActivationsDict = dict[str, Float[Array, "batch_size layer_dim"]]
type Intermediates = dict[str, tuple[LayerActivations, ...] | "Intermediates"]


class Histogram(flax.struct.PyTreeNode):
    total_events: int
    data: Float[npt.NDArray | Array, "..."] | None = None
    np_histogram: tuple | None = None


class Activation(enum.Enum):
    ReLU = enum.member(jax.nn.relu)
    Tanh = enum.member(jax.nn.tanh)
    LeakyReLU = enum.member(jax.nn.leaky_relu)
    PReLU = enum.member(lambda x: flax.linen.PReLU()(x))  # noqa: E731
    ReLU6 = enum.member(jax.nn.relu6)
    SiLU = enum.member(jax.nn.silu)
    Swish = enum.member(jax.nn.swish)
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
    observations: Float[Observation, "timestep env"]
    actions: Float[Action, "timestep env"]
    rewards: Float[Reward, "timestep env 1"]
    terminated: Bool[npt.NDArray | Array, "timestep env 1"]
    truncated: Bool[npt.NDArray | Array, "timestep env 1"]
    next_observations: Float[Observation, "timestep env"]

    infos: dict

    # Auxiliary policy outputs
    log_probs: Float[LogProb, "timestep env"] | None = None
    values: Float[Value, "timestep env 1"] | None = None
