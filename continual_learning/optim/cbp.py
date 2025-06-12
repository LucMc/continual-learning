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

"""
This is an implementation of continual back propergation (CBP): https://www.nature.com/articles/s41586-024-07711-7

TODO:
 * Clip ages
 * Reset adam/optim state for reset nodes
 * fix logging

Count = 15

:: Testing ::
  * See testing list in test_reset.py

:: Implementation ::
  * Implement accumulated nodes to reset for inexact division by replacement rate

:: Errors ::
  * Assert statements throughout, check mask is always false when replacement rate is 0 and n_to_replace is also always zero etc same with maturity_threshold
  * Is utility a good measure/ do we outperform random weight reinitialisation?
"""


@dataclass
class CBPOptimState:
    initial_weights: PyTree[Float[Array, "..."]]
    utilities: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, ""]
    ages: Array
    accumulated_features_to_replace: int
    rng: PRNGKeyArray

    step_size: float = 0.001
    replacement_rate: float = 0.5
    decay_rate: float = 0.9
    maturity_threshold: int = 10
    accumulate: bool = False
    logs: FrozenDict = FrozenDict({"avg_age": 0, "nodes_reset": 0})

# Define parameter partitioning functions
def is_kernel(path, value):
    return path[-1].key == 'kernel'

def is_bias(path, value):
    return path[-1].key == 'bias'

def reset_optim_params(tx_state, reset_mask):
    # Adam: tuple -> scale by adam @ tx_state[0]
    # Adam: PartitionState -> scale by adam @ tx_state[0]

    def composite(tx_state):
        new_state = {}
        for name, txs in tx_state.inner_states.items():
            new_state[name] = reset_optim_params(txs.inner_state, reset_mask)

        return new_state
    
    def reset_params(tx_state):
        
        if hasattr(tx_state, "mu"):
            mu = tx_state.mu
            # reset_transform = optax.multi_transform({"kernel": lambda k: k, "bias": lambda b: b})
            # optax.multi_transform
            # breakpoint()
            # reset_mu = jax.tree.map(
            #     lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
            #     {"params": reset_mask},
            #     mu)

        if hasattr(tx_state, "nu"):
            nu = tx_state.nu
            # reset_nu = jax.tree.map(
            #     lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
            #     {"params": reset_mask},
            #     nu)
            
        return tx_state
    
    # Check if it has the inner_states attribute
    is_partition_state = hasattr(tx_state, 'inner_states')
    
    if is_partition_state:
        return composite(tx_state)
    else:
        return reset_params(tx_state[0])


# -------------- CBP Weight reset ---------------
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
    return updated_utility


# -------------- lowest utility mask ---------------
def get_reset_mask(
    updated_utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
) -> Bool[Array, "#neurons"]:
    maturity_mask = ages > maturity_threshold  # get nodes over maturity threshold Arr[Bool]
    n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int
    k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)  # bool

    return k_masked_utility


@jax.jit
def get_out_weights_mag(weights):
    w_mags = jax.tree.map(
        lambda layer_w: jnp.abs(layer_w).mean(axis=1), weights
    )  # [2, 10] -> [2,1] mag over w coming out of neuron - LOP does axis 0 of out_layer but should be eqivalent

    keys = list(w_mags.keys())
    return {keys[i]: w_mags[keys[i + 1]] for i in range(len(keys) - 1)}


def process_params(params: PyTree):
    # TODO: Make out_w_mag optional so can be used by redo too
    out_layer_name = "out_layer"

    excluded = {
        out_layer_name: params[out_layer_name]
    }  # TODO: pass excluded layer names as inputs to cp optim/final by default
    bias = {}
    weights = {}

    for layer_name in params.keys():
        # For layer norm etc
        if type(params[layer_name]) != dict:
            excluded.update({layer_name: params[layer_name]})
            continue

        elif not ("kernel" in params[layer_name].keys()):
            excluded.update({layer_name: params[layer_name]})
            continue

        bias[layer_name] = params[layer_name]["bias"]
        weights[layer_name] = params[layer_name]["kernel"]

    out_w_mag = get_out_weights_mag(weights)

    # Remove output layer
    weights.pop(out_layer_name)
    bias.pop(out_layer_name)

    return weights, bias, out_w_mag, excluded


# -------------- Main CBP Optimiser body ---------------
def cbp(
    replacement_rate: float = 0.5,  # Update to paper hyperparams
    decay_rate: float = 0.9,
    maturity_threshold: int = 10,
    rng: Array = random.PRNGKey(0),
) -> optax.GradientTransformationExtraArgs:
    def init(params: optax.Params, **kwargs):
        weights, bias, _, _ = process_params(params["params"])

        del params  # Delete params?

        return CBPOptimState(
            initial_weights=weights,
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), bias),
            accumulated_features_to_replace=0,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            rng=rng,
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates, # Gradients
        state: CBPOptimState,
        params: optax.Params,
        features: Array,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, CBPOptimState]:
        def _cbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            assert features, "Features must be provided in update"
            _features = features["intermediates"]["activations"][0]

            weights, bias, out_w_mag, excluded = process_params(params["params"])

            new_rng, util_key = random.split(rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            # vmap utility calculation over batch
            batched_util_calculation = jax.vmap(
                partial(get_updated_utility, decay_rate=decay_rate),
                in_axes=(None, None, 0),
            )
            _utility_batch = jax.tree.map(
                batched_util_calculation, out_w_mag, state.utilities, _features
            )
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
            _weights, reset_logs = utils.reset_weights(
                reset_mask, weights, key_tree, state.initial_weights
            )

            # reset bias given mask
            _bias = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
                reset_mask,
                bias,
            )

            # Update ages
            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), a + 1
                ),  # Clip to stop huge ages unnessesarily
                state.ages,
                reset_mask,
            )

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: kinda sucks for adding logs

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)

            # Reset optim, o.e. Adamw params
            reset_optim_params(tx_state, reset_mask)

            # Logging
            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                ages=_ages, rng=new_rng, logs=FrozenDict(_logs), utilities=_utility
            )
            new_params.update(excluded)  # TODO

            return {"params": new_params}, new_state, tx_state

        return _cbp(updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
