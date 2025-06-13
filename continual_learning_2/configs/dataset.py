from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    """The name of the dataset."""

    seed: int
    """The seed to use for random shuffling etc."""

    batch_size: int
    """The batch size for training and testing."""

    num_epochs_per_task: int
    """How many epochs to repeat the dataset over for each task."""

    num_tasks: int
    """The number of continual learning tasks."""

    num_workers: int = 0
    """The number of workers to use for data loading. 0 = no multi-processing."""
