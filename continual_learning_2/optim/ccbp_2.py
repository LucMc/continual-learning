from cgitb import reset
from flax import struct
import flax.linen as nn
from flax.core import FrozenDict
from flax.typing import FrozenVariableDict
from jax.random import PRNGKey
from jaxlib.mlir.dialects.sparse_tensor import out
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
from typing import Tuple
from chex import dataclass
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from dataclasses import field

import continual_learning_2.utils.optim as utils
from continual_learning_2.optim.cbp import CBPOptimState

# -------------- CCBP Weight reset ---------------
def continuous_weight_reset(
    reset_mask: PyTree[Float[Array, "#neurons"]], 
    layer_w: PyTree[Float[Array, "..."]],
    initial_weights: PyTree[Float[Array, "..."]],
    utilities: PyTree[Float[Array, "..."]],
    replacement_rate: Float[Array, ""] = 0.001,
):
    layer_names = list(reset_mask.keys())
    logs = {}

    for i in range(len(layer_names) - 1):
        in_layer = layer_names[i]
        out_layer = layer_names[i + 1]

        zero_out_weights = jnp.zeros(layer_w[out_layer].shape, float)

        in_reset_mask = reset_mask[in_layer].reshape(1, -1)  # [1, out_size]

        stepped_in_weights = (replacement_rate * initial_weights[in_layer]) + (
            (1 - replacement_rate) * layer_w[in_layer]
        )
        stepped_util_in_weights = utilities[in_layer]*layer_w[in_layer] + (1-utilities[in_layer])*stepped_in_weights
        _in_layer_w = jnp.where(in_reset_mask, stepped_util_in_weights, layer_w[in_layer])

        stepped_out_weights = (replacement_rate * zero_out_weights) + (
            (1 - replacement_rate) * layer_w[out_layer]
        )

        out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
        stepped_util_out_weights = utilities[out_layer]*layer_w[out_layer] + (1-utilities[out_layer])*stepped_out_weights

        _out_layer_w = jnp.where(out_reset_mask, stepped_util_out_weights, layer_w[out_layer])

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": 0} # n_reset no longer applicable
    logs[out_layer] = {"nodes_reset": 0}
    return layer_w, logs


def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],
    decay_rate: Float[Array, ""] = 0.9,
):
    # Remove batch dim from some inputs just in case
    updated_utility = (
        (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features) * out_w_mag
    ).flatten()  # Arr[#neurons]
    return nn.softmax(updated_utility)


# -------------- mature only mask ---------------
def get_reset_mask(
    updated_utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
) -> Bool[Array, "#neurons"]:
    maturity_mask = ages > maturity_threshold  # get nodes over maturity threshold Arr[Bool]
    return maturity_mask


# -------------- Main CCBP Optimiser body ---------------
def ccbp2(
    replacement_rate: float = 0.5,  # Update to paper hyperparams
    decay_rate: float = 0.9,
    maturity_threshold: int = 10,
) -> optax.GradientTransformationExtraArgs:
    """ Continuous Continual Backpropergation (CCBP) """

    def init(params: optax.Params, **kwargs):
        weights, bias, _ = utils.process_params(params["params"])

        del params

        return CBPOptimState(
            initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), bias),
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CBPOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
        tx_state: optax.OptState | None = None
    ) -> tuple[optax.Updates, CBPOptimState]:
        def _ccbp2(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            # assert features, "Features must be provided in update"
            # _features = features["activations"][0]

            weights, bias, out_w_mag, excluded = utils.process_params_with_outmag(params["params"])

            # vmap utility calculation over batch
            batched_util_calculation = jax.vmap(
                partial(get_updated_utility, decay_rate=decay_rate),
                in_axes=(None, None, 0),
            )
            # Flatten all the structures (for unmatched features/params pytrees)
            out_w_mag_leaves, _ = jax.tree.flatten(out_w_mag)
            utilities_leaves, utilities_tree_def = jax.tree.flatten(state.utilities)
            features_leaves = [f[0] for f in jax.tree.leaves(features)]

            # Apply the calculation to the leaves
            utility_batch_leaves = [
                batched_util_calculation(w, u, f)
                for w, u, f in zip(out_w_mag_leaves, utilities_leaves, features_leaves)
            ]
            # Reconstruct the tree
            _utility_batch = jax.tree.unflatten(utilities_tree_def, utility_batch_leaves)
            _utility = jax.tree.map(lambda x: x.mean(axis=0), _utility_batch)

            reset_mask = jax.tree.map(
                partial(
                    get_reset_mask,
                    maturity_threshold=maturity_threshold,
                    replacement_rate=replacement_rate,
                ),
                _utility,
                state.ages,
            )

            # reset weights given mask
            _weights, reset_logs = continuous_weight_reset( 
                reset_mask, weights, state.initial_weights, _utility, replacement_rate
            )

            # reset bias given mask
            # Expermiment: reset bias/continuous reset bias/ leave bias alone
            # _bias = jax.tree.map(
            #     lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
            #     reset_mask,
            #     bias,
            # )
            _bias = bias

            # Update ages
            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), jnp.clip(a + 1, max=maturity_threshold+1)
                ),  # Clip to stop huge ages
                state.ages,
                reset_mask,
            )

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: kinda sucks for adding logs

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            avg_util = jax.tree.map(lambda v: v.mean(), _utility)
            std_util = jax.tree.map(lambda v: v.std(), _utility)

            # Logging
            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["avg_util"] += avg_util[layer_name]
                _logs["std_util"] += std_util[layer_name]
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                ages=_ages, logs=FrozenDict(_logs), utilities=_utility
            )
            new_params.update(excluded)

            return {"params": new_params}, new_state, tx_state

        return _ccbp2(updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
