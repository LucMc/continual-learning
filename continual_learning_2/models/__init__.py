import flax.linen as nn

from continual_learning_2.configs.models import MLPConfig

from .mlp import MLP


def get_model(cfg) -> nn.Module:
    if isinstance(cfg, MLPConfig):
        return MLP(cfg)
    else:
        raise ValueError(f"Unknown model configuration: {cfg}")
