import optax
import jax
import flax.linen as nn
from continual_learning_2.configs.optim import (
    AdamConfig,
    CBPConfig,
    CCBPConfig,
    CCBP2Config,
    RedoConfig,
    ShrinkAndPerterbConfig,
    OptimizerConfig,
)
from continual_learning_2.utils.optim import attach_reset_method

from .redo import redo
from .ccbp import ccbp
from .ccbp_2 import ccbp2
from .shrink_perturb import shrink_perturb
from .cbp import cbp
from .identity_reset import identity_reset



def get_optimizer(config: OptimizerConfig, is_inner=False):
    if isinstance(config, AdamConfig):
        tx = optax.adam(
            config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.epsilon
        )
        if is_inner:
            return tx
        else:
            return attach_reset_method(("tx", tx),
                                   ("reset_method", identity_reset(**config.__dict__)))


    elif isinstance(config, ShrinkAndPerterbConfig):
        return attach_reset_method(("tx", get_optimizer(config.__dict__.pop("tx"), is_inner=True)),
                                   ("reset_method", shrink_perturb(**config.__dict__)))

    elif isinstance(config, RedoConfig):
        return attach_reset_method(("tx", get_optimizer(config.__dict__.pop("tx"), is_inner=True)),
                                   ("reset_method", redo(**config.__dict__)))

    elif isinstance(config, CBPConfig):
        return attach_reset_method(("tx", get_optimizer(config.__dict__.pop("tx"), is_inner=True)),
                                   ("reset_method", cbp(**config.__dict__)))

    elif isinstance(config, CCBPConfig):
        return attach_reset_method(("tx", get_optimizer(config.__dict__.pop("tx"), is_inner=True)),
                                   ("reset_method", ccbp(**config.__dict__)))

    elif isinstance(config, CCBP2Config):
        return attach_reset_method(("tx", get_optimizer(config.__dict__.pop("tx"), is_inner=True)),
                                   ("reset_method", ccbp2(**config.__dict__)))
    else:
        raise ValueError(f"Unsupported optimizer config type: {type(config)}")

