from flax import struct
from flax.core import FrozenDict
from flax.core.lift import C
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
from typing import Tuple
from chex import dataclass
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from dataclasses import field

import continual_learning.utils.optim as utils
from continual_learning.optim.cbp import (
    get_out_weights_mag,
    process_params,
    CBPOptimState,
)


# -------------- CCBP Weight reset ---------------
def reset_weights(
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    key_tree: PyTree[PRNGKeyArray],
    initial_weights: PyTree[Float[Array, "..."]],
    replacement_rate: Float[Array, ""] = 0.01,
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

        # TODO: Check this is resetting the correct row and columns
        in_reset_mask = reset_mask[in_layer].reshape(1, -1)  # [1, out_size]

        # _in_layer_w = jnp.where(in_reset_mask, random_in_weights, layer_w[in_layer])
        stepped_in_weights = (replacement_rate * initial_weights[in_layer]) + (
            (1 - replacement_rate) * layer_w[in_layer]
        )
        _in_layer_w = jnp.where(in_reset_mask, stepped_in_weights, layer_w[in_layer])

        stepped_out_weights = (replacement_rate * zero_out_weights) + (
            (1 - replacement_rate) * layer_w[out_layer]
        )
        out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
        _out_layer_w = jnp.where(out_reset_mask, zero_out_weights, layer_w[out_layer])
        n_reset = reset_mask[in_layer].sum()

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": n_reset}
    logs[out_layer] = {"nodes_reset": 0}
    return layer_w, logs


# -------------- mature only mask ---------------
def get_reset_mask(
    ages: Float[Array, "#neurons"],
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
) -> Bool[Array, "#neurons"]:
    maturity_mask = ages > maturity_threshold  # get nodes over maturity threshold Arr[Bool]
    # n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int
    # k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)  # bool
    # return k_masked_utility
    return maturity_mask


# -------------- Main CCBP Optimiser body ---------------
def ccbp(
) -> optax.GradientTransformation:
    def init(params: optax.Params, **kwargs):
        weights, bias, _, _ = process_params(params["params"])

        del params  # Delete params?

        return CBPOptimState(
            initial_weights=weights,
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), bias),
            accumulated_features_to_replace=0,
            # rng=random.PRNGKey(0), # Seed passed in through kwargs?
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CBPOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
    ) -> tuple[optax.Updates, CBPOptimState]:
        def _ccbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            weights, bias, out_w_mag, excluded = process_params(params)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            reset_mask = jax.tree.map(
                partial(
                    get_reset_mask,
                    maturity_threshold=state.maturity_threshold,
                    replacement_rate=state.replacement_rate,
                ),
                state.ages,
            )

            # reset weights given mask
            _weights, reset_logs = reset_weights(
                reset_mask, weights, key_tree, state.initial_weights, state.replacement_rate
            )
            _weights = weights

            # reset bias given mask
            _bias = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
                reset_mask,
                bias,
            )
            _bias = bias

            # Update ages
            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), a + 1
                ),  # Clip to stop huge ages unnessesarily
                state.ages,
                reset_mask,
            )

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: Improve logging

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)

            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                ages=_ages, rng=new_rng, logs=FrozenDict(_logs)
            )
            new_params.update(excluded)  # TODO

            return {"params": new_params}, (new_state,)

        return _ccbp(updates)

    return optax.GradientTransformation(init=init, update=update)
