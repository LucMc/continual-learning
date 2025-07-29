from dataclasses import dataclass
import jax
import flax.linen as nn
from typing import Callable
import chex
import optax

<<<<<<< HEAD
# Base configs
@dataclass
=======

@dataclass(frozen=True)
>>>>>>> origin/ltnt-15-clean-up-the-codebase
class OptimizerConfig:
    learning_rate: float | optax.Schedule

@dataclass(frozen=True)
class ResetMethodConfig:
    tx: OptimizerConfig

# Standard optimizer configs
@dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

# Reset method configs
@dataclass(frozen=True)
class ShrinkAndPerterbConfig(ResetMethodConfig):
    param_noise_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    shrink: float = 0.8
    perturb: float = 0.01
    every_n: int = 1

@dataclass(frozen=True)
class RedoConfig(ResetMethodConfig):
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    update_frequency: int = 100
    score_threshold: float = 0.1
    
@dataclass(frozen=True)
class CBPConfig(ResetMethodConfig):
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    replacement_rate: float = 0.1
    decay_rate: float = 0.99
    maturity_threshold: int = 20
    
@dataclass(frozen=True)
class CCBPConfig(ResetMethodConfig):
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    replacement_rate: float = 0.1
    decay_rate: float = 0.99
    maturity_threshold: float = 20
