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

import continual_learning.utils.optim as utils


@dataclass
class BaseOptimState:
    initial_weights: PyTree[Float[Array, "..."]]
    utilities: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, ""]
    ages: Array
    accumulated_features_to_replace: int

    rng: PRNGKeyArray  # = random.PRNGKey(0)
    step_size: float = 0.001
    replacement_rate: float = 0.5
    decay_rate: float = 0.9
    maturity_threshold: int = 10
    accumulate: bool = False
    logs: FrozenDict = FrozenDict({"avg_age": 0, "nodes_reset": 0})


# -------------- Overall optimizer TrainState ---------------
class ResettingTrainState(TrainState):
    reset_method: Callable = struct.field(pytree_node=False)
    cbp_state: optax.OptState = struct.field(pytree_node=True)

    # TODO: Investigate if we can OVERWRITE_WITH_GRADIENT and GradientTransformationExtraArgs
    @classmethod
    def create(cls, *, apply_fn, params, tx, reset_method, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        # params_with_opt = (
        #   params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        # )
        opt_state = tx.init(params)
        cbp_state = reset_method().init(params, **kwargs)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            cbp_state=cbp_state,
            reset_method=reset_method
        )

    def apply_gradients(self, *, grads, features, **kwargs):
        """TrainState that gives intermediates to optimizer and overwrites params with updates directly"""

        # Get updates from optimizer
        tx_updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params
        )  # tx first then reset so we don't change reset params based on old grads
        params_after_tx = optax.apply_updates(self.params, tx_updates)

        # Update with continual backprop
        params_after_cbp, new_cbp_state = self.reset_method().update(
            grads["params"],
            self.cbp_state,
            params_after_tx["params"],
            features=features["intermediates"]["activations"][0],
        )

        # utils.check_tree_shapes(params_after_tx, params_after_cbp)
        # utils.check_tree_shapes(self.params, params_after_cbp)

        return self.replace(
            step=self.step + 1,
            params=params_after_cbp,
            opt_state=new_opt_state,
            cbp_state=new_cbp_state[0],
            **kwargs,
        )


# -------------- Weight reset ---------------
class BaseOptimiser(abc.ABC):
    @staticmethod
    def reset_weights(
        reset_mask: PyTree[Bool[Array, "#neurons"]],
        layer_w: PyTree[Float[Array, "..."]],
        key_tree: PyTree[PRNGKeyArray],
        initial_weights: PyTree[Float[Array, "..."]],
        replacement_rate: Float[Array, ""] = None,
    ):
        layer_names = list(reset_mask.keys())
        logs = {}

        for i in range(len(layer_names) - 1):
            in_layer = layer_names[i]
            out_layer = layer_names[i + 1]

            # Generate random weights for resets
            # random_in_weights = random.uniform(
            #     key_tree[in_layer], layer_w[in_layer].shape, float, -bound, bound
            # )
            zero_out_weights = jnp.zeros(layer_w[out_layer].shape, float)

            assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"

            in_reset_mask = reset_mask[in_layer].flatten()  # [1, out_size]
            _in_layer_w = jnp.where(
                in_reset_mask, initial_weights[in_layer], layer_w[in_layer]
            )

            # out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
            _out_layer_w = jnp.where(in_reset_mask, zero_out_weights, layer_w[out_layer])
            n_reset = reset_mask[in_layer].sum()

            layer_w[in_layer] = _in_layer_w
            layer_w[out_layer] = _out_layer_w

            logs[in_layer] = {"nodes_reset": n_reset}
        logs[out_layer] = {"nodes_reset": 0}
        return layer_w, logs

    @staticmethod
    def reset_method(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, BaseOptimState]:

        @abc.abstractmethod
        def init(params: optax.Params, **kwargs):
            pass

        @abc.abstractmethod
        def update(
            updates: optax.Updates,  # Gradients
            state: CBPOptimState,
            params: optax.Params | None = None,
            features: Array | None = None,
        )->tuple[optax.Updates, BaseOptimState]:
            pass

