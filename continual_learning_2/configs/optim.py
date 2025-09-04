from dataclasses import dataclass, field
import jax
import flax.linen as nn
from typing import Callable
import chex
import optax

# Base configs
@dataclass(frozen=True)
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

@dataclass(frozen=True)
class AdamwConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    decay: float = 1e-4

@dataclass(frozen=True)
class MuonConfig(OptimizerConfig):
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
    update_frequency: int
    score_threshold: float
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    

@dataclass(frozen=True)
class RegramaConfig(ResetMethodConfig):
    update_frequency: int
    score_threshold: float
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42

@dataclass(frozen=True)
class CbpConfig(ResetMethodConfig):
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    replacement_rate: float = 0.1
    decay_rate: float = 0.99
    maturity_threshold: int = 20
    
@dataclass(frozen=True)
class CcbpConfig(ResetMethodConfig):
    weight_init_fn: Callable = jax.nn.initializers.he_uniform()
    seed: int = 42
    replacement_rate: float = 0.1
    sharpness: float = -1
    threshold: float = -1
    decay_rate: float = 0.99
    update_frequency: int = 100
