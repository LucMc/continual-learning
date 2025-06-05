import abc
from typing import TYPE_CHECKING, Generator

from continual_learning_2.types import LogDict, PredictorModel

if TYPE_CHECKING:
    import grain.python as grain


class ContinualLearningDataset(abc.ABC):
    NUM_CLASSES: int

    @abc.abstractmethod
    def __init__(
        self, seed: int, num_tasks: int = 5, num_epochs: int = 20, batch_size: int = 32
    ): ...

    @abc.abstractproperty
    def tasks(self) -> Generator[grain.IterDataset, None, None]: ...

    @abc.abstractmethod
    def evaluate(self, model: PredictorModel, forgetting: bool = False) -> LogDict: ...
