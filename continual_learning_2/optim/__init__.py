import optax
import jax
import flax.linen as nn
from continual_learning_2.configs.optim import (
    AdamConfig,
    ShrinkAndPerterbConfig,
    OptimizerConfig,
)

# from .redo import redo
# from .ccbp import ccbp
# from .ccbp_2 import ccbp2
from .shrink_perturb import shrink_perturb
# from .cbp import cbp


def attach_reset_method(
    *args: tuple[str, optax.GradientTransformation],
) -> optax.GradientTransformationExtraArgs:
    names = [name for name, _ in args]

    if len(names) != len(set(names)):
        raise ValueError(f"Named transformations must have unique names, but got {names}")

    transforms = [(name, optax.with_extra_args_support(t)) for name, t in args]

    def init_fn(params):
        states = {}
        for name, tx in transforms:
            states[name] = tx.init(params)
        return states

    def update_fn(updates, state, params=None, features=None, **extra_args):
        """Updated named chain update from Optax to enable resetting of base optim running stats"""

        new_state = {}
        assert len(transforms) == 2, "chain the optim with the reset method only"
        assert "tx" == args[0][0], "'tx' is the first part of this chain"
        assert "reset_method" == args[1][0], "'reset_method' is the second part of this chain"

        # TX update
        tx = transforms[0][1]
        reset_method = transforms[1][1]

        updates, new_state["tx"] = tx.update(updates, state["tx"], params, **extra_args)
        new_params_with_opt = optax.apply_updates(params, updates)

        # Reset method
        new_params_with_reset, new_state["reset_method"], new_state["tx"] = (
            reset_method.update(
                updates,
                state["reset_method"],
                new_params_with_opt,
                features=features,
                tx_state=new_state["tx"],
            )
        )

        return new_params_with_reset, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def get_optimizer(config: OptimizerConfig):
    if isinstance(config, AdamConfig):
        return optax.adam(
            config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.epsilon
        )
    elif isinstance(config, ShrinkAndPerterbConfig):
        # ADD RESET ARGS
        reset_args = dict(rng=jax.random.PRNGKey(config.seed),
                          param_noise_fn=nn.initializers.xavier_normal())

        return attach_reset_method(("tx", get_optimizer(config.tx)),
                                   ("reset_method",
                                   shrink_perturb(**reset_args)))

    # elif isinstance(config, AdamConfig):
    #     return optax.adam(
    #         config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.epsilon
    #     )
    # elif isinstance(config, AdamConfig):
    #     return optax.adam(
    #         config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.epsilon
    #     )
    else:
        raise ValueError(f"Unsupported optimizer config type: {type(config)}")

