import flax.linen as nn

from continual_learning_2.configs.models import CNNConfig, MLPConfig, ResNetConfig

from .cnn import CNN
from .mlp import MLP
from .resnet import ResNet


def get_model(cfg) -> nn.Module:
    if isinstance(cfg, MLPConfig):
        return MLP(cfg)
    elif isinstance(cfg, CNNConfig):
        return CNN(cfg)
    elif isinstance(cfg, ResNetConfig):
        return ResNet(cfg)
    else:
        raise ValueError(f"Unknown model configuration: {cfg}")


def get_model_cls(cfg) -> type[nn.Module]:
    if isinstance(cfg, MLPConfig):
        return MLP
    elif isinstance(cfg, CNNConfig):
        return CNN
    elif isinstance(cfg, ResNetConfig):
        return ResNet
    else:
        raise ValueError(f"Unknown model configuration: {cfg}")
