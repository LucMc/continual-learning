from flax import struct
from flax.core import FrozenDict
from flax.typing import FrozenVariableDict
from jax.random import PRNGKey
from jaxtyping import (
    Array,
    Float,
    Bool,
    PRNGKeyArray,
    PyTree,
    jaxtyped,
    TypeCheckError,
    Scalar,
    Int,
)
from beartype import beartype as typechecker
from flax.training.train_state import TrainState
from typing import Tuple, Callable
from chex import dataclass
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from dataclasses import field
import abc

import continual_learning.optim as optim
import continual_learning.utils.optim as utils

"""
TODO: Investigate using optax partition as a cleaner solution to applying updates to only weights/
biases, and potentially ignoring the last layer. Could also consider only layers w features as targets
"""

# Put in utils
def identity_reset(*args, **kwargs):
    
    def init_fn(params, *args, **kwargs):
        return {}
    
    def update_fn(updates, state, params, features, tx_state, *args, **extra_args):
        return params, state, tx_state
    
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

RESET_METHOD_MAP = {"cbp": optim.cbp,
                    "redo": optim.redo,
                    "ccbp": optim.ccbp,
                    "ccbp2": optim.ccbp2,
                    "none": identity_reset
                    }



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
        new_params_with_reset, new_state["reset_method"], new_state["tx"] = reset_method.update(
            updates,
            state["reset_method"],
            new_params_with_opt,
            features=features,
            tx_state=new_state["tx"],
        )

        return new_params_with_reset, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

# -------------- Overall optimizer TrainState ---------------
class ResettingTrainState(TrainState):
    """
    TrainState for attaching dormant neuron resetting methods. Ensure each element in tx
    is wrapped with optax.with_extra_args_support and take in features to apply_gradients.
    """
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, reset_method, reset_method_kwargs={}, **kwargs):

        # Attach reset method
        reset_method_fn = RESET_METHOD_MAP[reset_method](**reset_method_kwargs)
        tx = attach_reset_method(("tx", tx), ("reset_method", reset_method_fn))
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            opt_state=opt_state,
            tx=tx,
            **kwargs)

    def apply_gradients(self, *, grads, features=None, **kwargs):
        assert features, "Features must be provided to apply_gradients()"

        new_params, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, features=features
        )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
