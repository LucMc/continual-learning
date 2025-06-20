from dataclasses import dataclass
import jax
import flax.linen as nn
from typing import Callable
import chex

@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float

@dataclass(frozen=True)
class ResetMethod:
    tx: OptimizerConfig

class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

@dataclass(frozen=True)
class ShrinkAndPerterbConfig(ResetMethod):
    param_noise_fn: Callable = nn.initializers.xavier_normal()
    seed: int = 42
    shrink: float = 0.8
    perturb: float = 0.01
    every_n: int = 1

@dataclass(frozen=True)
class RedoConfig(ResetMethod):
    update_frequency: int = 10
    score_threshold: float = 0.1
    
@dataclass(frozen=True)
class CBPConfig(ResetMethod):
    replacement_rate: float
    decay_rate: float
    maturity_threshold: float
    accumulate: float # TODO
    
@dataclass(frozen=True)
class CCBPConfig(ResetMethod):
    replacement_rate: float
    decay_rate: float
    maturity_threshold: float
    
@dataclass(frozen=True)
class CCBP2Config(ResetMethod):
    replacement_rate: float
    decay_rate: float
    maturity_threshold: float
    
