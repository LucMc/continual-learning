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


RESET_METHOD_MAP = {"cbp": optim.cbp,
                    "redo": optim.redo,
                    "ccbp": optim.ccbp,
                    "ccbp2": optim.ccbp2,
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
        assert len(transforms) == 2, (
            "This chain be at the end and chain the optim with the reset method only"
        )
        assert "tx" == args[0][0], "'tx' is the first part of this chain"
        assert "reset_method" == args[1][0], "'reset_method' is the second part of this chain"

        # TX Update
        tx = transforms[0][1]
        reset_method = transforms[1][1]
        updates, new_state["tx"] = tx.update(updates, state["tx"], params, **extra_args)

        # Reset method
        updates, new_state["reset_method"], new_state["tx"] = reset_method.update(
            updates,
            state["reset_method"],
            params,
            features=features,
            tx_state=state["tx"],
            **extra_args,
        )

        return updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

# -------------- Overall optimizer TrainState ---------------
class ResettingTrainState(TrainState):
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, reset_method=None, reset_method_kwargs={}, **kwargs):
        assert reset_method in RESET_METHOD_MAP.keys(), f"reset method must be one of: {RESET_METHOD_MAP.keys()}"

        # Attach reset method
        reset_method_fn = RESET_METHOD_MAP[reset_method](**reset_method_kwargs)
        tx = attach_reset_method(("tx", tx), ("reset_method", reset_method_fn))

        opt_state = tx.init(params)
        ts_args = dict(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs)
        
        return cls(**ts_args) if reset_method != "none" else TrainState(**ts_args)

    def apply_gradients(self, *, grads, features=None, **kwargs):
        assert features, "Features must be provided to apply_gradients()"

        grads_with_opt = grads
        params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, features=features
        )
        # new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        new_params_with_opt = updates
        # Set params with params given by last optim (dormant reset method)

        new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


# -------------- Weight reset ---------------
# class BaseOptimiser(abc.ABC):
#     @staticmethod
#
#     def reset_weights(
#         reset_mask: PyTree[Bool[Array, "#neurons"]],
#         layer_w: PyTree[Float[Array, "..."]],
#         key_tree: PyTree[PRNGKeyArray],
#         initial_weights: PyTree[Float[Array, "..."]],
#         replacement_rate: Float[Array, ""] = None,
#     ):
#         layer_names = list(reset_mask.keys())
#         logs = {}
#
#         for i in range(len(layer_names) - 1):
#             in_layer = layer_names[i]
#             out_layer = layer_names[i + 1]
#
#             assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"
#             assert len(reset_mask[in_layer].flatten()) == layer_w[out_layer].shape[0], (
#                 f"Reset mask shape incorrect: {len(reset_mask[in_layer].flatten())} should be {layer_w[out_layer].shape[0]}"
#             )
#
#             in_reset_mask = reset_mask[in_layer].reshape(-1)  # [1, out_size]
#             _in_layer_w = jnp.where(in_reset_mask, initial_weights[in_layer], layer_w[in_layer])
#
#             _out_layer_w = jnp.where(
#                 in_reset_mask, jnp.zeros_like(layer_w[out_layer]), layer_w[out_layer]
#             )
#             n_reset = reset_mask[in_layer].sum()
#
#             layer_w[in_layer] = _in_layer_w
#             layer_w[out_layer] = _out_layer_w
#
#             logs[in_layer] = {"nodes_reset": n_reset}
#
#         logs[out_layer] = {"nodes_reset": 0}
#
#         return layer_w, logs
#
#     @staticmethod
#     def reset_method(
#         updates: optax.Updates,
#     ) -> Tuple[optax.Updates, BaseOptimState]:
#         @abc.abstractmethod
#         def init(params: optax.Params, **kwargs):
#             pass
#
#         @abc.abstractmethod
#         def update(
#             updates: optax.Updates,  # Gradients
#             state: CBPOptimState,
#             params: optax.Params | None = None,
#             features: Array | None = None,
#         ) -> tuple[optax.Updates, BaseOptimState, Any]:
#             pass
