import optax
from continual_learning_2.configs.optim import (
    AdamConfig,
    AdamwConfig,
    MuonConfig,
    CbpConfig,
    CcbpConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
    OptimizerConfig,
)
from continual_learning_2.utils.optim import attach_reset_method

from .redo import redo
from .regrama import regrama
from .ccbp import ccbp
from .shrink_perturb import shrink_perturb
from .cbp import cbp
from .identity_reset import identity_reset


def get_optimizer(config: OptimizerConfig, is_inner=False):
    rm_config = config.__dict__.copy()

    # Inner optimizers
    if isinstance(config, AdamConfig):
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(
                config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.epsilon
            ),
        )
        if is_inner:
            return tx
        else:
            return attach_reset_method(tx=tx, reset_method=identity_reset(**config.__dict__))

    if isinstance(config, AdamwConfig):
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(
                config.learning_rate,
                b1=config.beta1,
                b2=config.beta2,
                eps=config.epsilon,
                weight_decay=config.decay,
            ),
        )
        if is_inner:
            return tx
        else:
            return attach_reset_method(tx=tx, reset_method=identity_reset(**config.__dict__))

    if isinstance(config, MuonConfig):
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.contrib.muon(
                config.learning_rate,
                adam_b1=config.beta1,
                adam_b2=config.beta2,
                adam_eps_root=config.epsilon,
            ),
        )
        if is_inner:
            return tx
        else:
            return attach_reset_method(tx=tx, reset_method=identity_reset(**config.__dict__))

    # Outer/reset methods
    elif isinstance(config, ShrinkAndPerterbConfig):
        return attach_reset_method(
            tx=get_optimizer(rm_config.pop("tx"), is_inner=True),
            reset_method=shrink_perturb(**rm_config),
        )

    elif isinstance(config, RedoConfig):
        return attach_reset_method(
            tx=get_optimizer(rm_config.pop("tx"), is_inner=True),
            reset_method=redo(**rm_config),
        )

    elif isinstance(config, RegramaConfig):
        return attach_reset_method(
            tx=get_optimizer(rm_config.pop("tx"), is_inner=True),
            reset_method=regrama(**rm_config),
        )

    elif isinstance(config, CbpConfig):
        return attach_reset_method(
            tx=get_optimizer(rm_config.pop("tx"), is_inner=True),
            reset_method=cbp(**rm_config),
        )

    elif isinstance(config, CcbpConfig):
        return attach_reset_method(
            tx=get_optimizer(rm_config.pop("tx"), is_inner=True),
            reset_method=ccbp(**rm_config),
        )
    else:
        raise ValueError(f"Unsupported optimizer config type: {type(config)}")
