from dataclasses import dataclass
import jax
import flax.linen as nn
from typing import Callable
import chex
import optax

# Base configs
@dataclass
class OptimizerConfig:
    learning_rate: float | optax.Schedule

@dataclass(frozen=True)
class ResetMethodConfig:
    tx: OptimizerConfig

# Standard optimizer configs
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

# Reset method configs
@dataclass(frozen=True)
class ShrinkAndPerterbConfig(ResetMethodConfig):
    param_noise_fn: Callable = nn.initializers.xavier_normal()
    seed: int = 42
    shrink: float = 0.8
    perturb: float = 0.01
    every_n: int = 1

@dataclass(frozen=True)
class RedoConfig(ResetMethodConfig):
    update_frequency: int = 100
    score_threshold: float = 0.1
    
@dataclass(frozen=True)
class CBPConfig(ResetMethodConfig):
    replacement_rate: float = 0.1
    decay_rate: float = 0.9
    maturity_threshold: int = 20
    accumulate: bool = False # TODO
    
@dataclass(frozen=True)
class CCBPConfig(ResetMethodConfig):
    replacement_rate: float
    decay_rate: float
    maturity_threshold: float
    
@dataclass(frozen=True)
class CCBP2Config(ResetMethodConfig):
    replacement_rate: float
    decay_rate: float
    maturity_threshold: float
    
