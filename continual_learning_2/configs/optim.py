from dataclasses import dataclass

import optax


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float | optax.Schedule


@dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
