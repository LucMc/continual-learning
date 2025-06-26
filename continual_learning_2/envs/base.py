import abc
from typing import Generator, NamedTuple, Protocol

import jax

from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.types import (
    Action,
    Done,
    EnvState,
    EpisodeLengths,
    EpisodeReturns,
    EpisodeStarted,
    Observation,
    Reward,
)


class Agent(Protocol):
    def eval_action(self, observation: Observation) -> Action: ...


class Timestep(NamedTuple):
    next_observation: Observation
    reward: Reward
    terminated: Done
    truncated: Done
    final_episode_returns: EpisodeReturns
    final_episode_lengths: EpisodeLengths
    final_observation: Observation


class VectorEnv(abc.ABC):
    @abc.abstractmethod
    def init(self) -> tuple[Observation, EpisodeStarted]: ...

    @abc.abstractmethod
    def step(self, action: Action) -> Timestep: ...

    @abc.abstractmethod
    def save(self) -> dict: ...

    @abc.abstractmethod
    def load(self, checkpoint: dict): ...

    @abc.abstractproperty
    def num_envs(self) -> int: ...


class JittableVectorEnv(abc.ABC):
    @abc.abstractmethod
    def init(self) -> tuple[EnvState, Observation]: ...

    @abc.abstractmethod
    def step(self, state: EnvState, action: Action) -> tuple[EnvState, Timestep]: ...

    @abc.abstractmethod
    def save(self) -> dict: ...

    @abc.abstractmethod
    def load(self, checkpoint: dict): ...


class ContinualLearningEnv(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config: EnvConfig): ...

    @abc.abstractproperty
    def num_envs(self) -> int: ...

    @abc.abstractproperty
    def tasks(self) -> Generator[VectorEnv, None, None]: ...

    @abc.abstractproperty
    def observation_spec(self) -> jax.ShapeDtypeStruct: ...

    @abc.abstractproperty  # TODO: This assumes continuous action space
    def action_dim(self) -> int: ...

    @abc.abstractmethod
    def evaluate(self, agent: Agent, forgetting: bool = False) -> dict[str, float]: ...

    @abc.abstractmethod
    def save(self) -> dict: ...

    @abc.abstractmethod
    def load(self, checkpoint: dict, envs_checkpoint: dict): ...
