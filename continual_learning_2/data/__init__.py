from continual_learning_2.configs.dataset import DatasetConfig

from .base import ContinualLearningDataset
from .cifar import SplitCIFAR10, SplitCIFAR100
from .mnist import PermutedMNIST, SplitMNIST


def get_dataset(dataset_config: DatasetConfig):
    if dataset_config.name == "permuted_mnist":
        return PermutedMNIST(dataset_config)
    elif dataset_config.name == "split_mnist":
        return SplitMNIST(dataset_config)
    elif dataset_config.name == "split_cifar10":
        return SplitCIFAR10(dataset_config)
    elif dataset_config.name == "split_cifar100":
        return SplitCIFAR100(dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")


__all__ = [
    "ContinualLearningDataset",
    "PermutedMNIST",
    "SplitMNIST",
    "SplitCIFAR10",
    "SplitCIFAR100",
]
