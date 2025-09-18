from continual_learning_2.configs.dataset import DatasetConfig

from .base import ContinualLearningDataset
from .cifar import (
    ClassIncrementalCIFAR10,
    ClassIncrementalCIFAR100,
    SplitCIFAR10,
    SplitCIFAR100,
)
from .mnist import ClassIncrementalMNIST, PermutedMNIST, SplitMNIST


def get_dataset(dataset_config: DatasetConfig):
    if dataset_config.name == "permuted_mnist":
        return PermutedMNIST(dataset_config, **dataset_config.dataset_kwargs)
    elif dataset_config.name == "split_mnist":
        return SplitMNIST(dataset_config, **dataset_config.dataset_kwargs)
    elif dataset_config.name == "classinc_mnist":
        return ClassIncrementalMNIST(dataset_config, **dataset_config.dataset_kwargs)
    elif dataset_config.name == "split_cifar10":
        return SplitCIFAR10(dataset_config, **dataset_config.dataset_kwargs)
    elif dataset_config.name == "split_cifar100":
        return SplitCIFAR100(dataset_config, **dataset_config.dataset_kwargs)
    elif dataset_config.name == "classinc_cifar10":
        return ClassIncrementalCIFAR10(dataset_config, **dataset_config.dataset_kwargs)
    elif dataset_config.name == "classinc_cifar100":
        return ClassIncrementalCIFAR100(dataset_config, **dataset_config.dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")


__all__ = [
    "ContinualLearningDataset",
    "PermutedMNIST",
    "SplitMNIST",
    "SplitCIFAR10",
    "SplitCIFAR100",
    "ClassIncrementalMNIST",
    "ClassIncrementalCIFAR10",
    "ClassIncrementalCIFAR100",
]
