import abc
from typing import Generic, Self, TypeVar

from flax import struct

from continual_learning_2.envs.base import ContinualLearningEnv
from continual_learning_2.types import Action, LogDict, Observation

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout)


class Algorithm(abc.ABC, Generic[AlgorithmConfigType, DataType], struct.PyTreeNode):
    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env: ContinualLearningEnv, seed: int = 1
    ) -> "Algorithm": ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...
