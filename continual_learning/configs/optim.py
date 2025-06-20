from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    learning_rate: float


class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
