import flax.linen as nn

from continual_learning.configs.models import CNNConfig, MLPConfig

from .cnn import CNN
from .mlp import MLP


def get_model(cfg) -> nn.Module:
    if isinstance(cfg, MLPConfig):
        return MLP(cfg)
    elif isinstance(cfg, CNNConfig):
        return CNN(cfg)
    else:
        raise ValueError(f"Unknown model configuration: {cfg}")


def get_model_cls(cfg) -> type[nn.Module]:
    if isinstance(cfg, MLPConfig):
        return MLP
    elif isinstance(cfg, CNNConfig):
        return CNN
    else:
        raise ValueError(f"Unknown model configuration: {cfg}")
