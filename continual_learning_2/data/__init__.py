from .base import ContinualLearningDataset
from .cifar import SplitCIFAR10, SplitCIFAR100
from .mnist import PermutedMNIST, SplitMNIST

__all__ = [
    "ContinualLearningDataset",
    "PermutedMNIST",
    "SplitMNIST",
    "SplitCIFAR10",
    "SplitCIFAR100",
]
