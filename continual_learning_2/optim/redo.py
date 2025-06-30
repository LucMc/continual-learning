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

import continual_learning_2.utils.optim as utils

"""
This file implements ReDo: 
https://github.com/google/dopamine/blob/master/dopamine/labs/redo/weight_recyclers.py
as an optax optimizer
"""

# TODO: Features are bundled with params and don't need to passed in explicitly!

@dataclass
class RedoOptimState:
    initial_weights: PyTree[Float[Array, "..."]]
    utilities: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, ""]

    # rng: PRNGKeyArray  # = random.PRNGKey(0)
    time_step: int = 0
    # scale: float = 1.0
    # step_size: float = 0.001
    # replacement_rate: float = 0.01
    # decay_rate: float = 0.9
    # update_frequency: int = 10
    logs: FrozenDict = FrozenDict({"nodes_reset": 0})


# -------------- Redo Weight reset ---------------
def get_score(  # averages over a batch
    features: Float[Array, "#batch #neurons"],
) -> Float[Array, "#neurons"]:
    # Avg over batches
    mean_act_per_neuron = jnp.mean(jnp.abs(features), axis=0)  # Arr[#neurons]
    score = mean_act_per_neuron / (
        jnp.mean(mean_act_per_neuron) + 1e-8
    )  # Arr[#neurons] / Scalar
    return score


# -------------- Main Redo Optimiser body ---------------
def redo(
    replacement_rate: float = 0.5,  # Update to paper hyperparams
    update_frequency: int = 100,
    score_threshold: float = 0.1,
) -> optax.GradientTransformationExtraArgs:
    def init(params: optax.Params, **kwargs):
        weights, bias, _ = utils.process_params(params["params"])

        del params  # Delete params?

        return RedoOptimState(
            initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            **kwargs,
        )

    def get_reset_mask(
        scores: Float[Array, "#neurons"],
    ) -> Bool[Array, "#neurons"]:
        score_mask = scores <= score_threshold  # get nodes over maturity threshold Arr[Bool]
        return score_mask

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: RedoOptimState,
        params: optax.Params,
        features: Array,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, RedoOptimState]:
        def no_update(updates):
            new_state = state.replace(time_step=state.time_step + 1)
            return params, new_state, tx_state

        def _redo(updates: optax.Updates,) -> Tuple[optax.Updates, RedoOptimState]:  # fmt: skip
            weights, bias, excluded = utils.process_params(params["params"])

            scores = {
                key: get_score(feature_tuple[0])
                for key, feature_tuple in zip(bias.keys(), features.values())
                # for key, feature_tuple in features.items()
            }
            # scores = jax.tree.map(
            #     lambda f: get_score(f[0]), features
            # )  # Scores, avged over batches: PyTree[#neurons]

            reset_mask = jax.tree.map(get_reset_mask, scores)

            # reset weights given mask
            _weights, reset_logs = utils.reset_weights(
                reset_mask, weights, state.initial_weights
            )

            # Update bias
            _bias = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b), reset_mask, bias
            )
            # bias_leaves, bias_tree_def = jax.tree.flatten(bias)
            # mask_leaves = jax.tree.leaves(reset_mask)
            # masked_leaves = [jnp.where(m, jnp.zeros_like(b, dtype=float), b)
            #                  for m, b in zip(mask_leaves, bias_leaves)]
            # _bias = jax.tree.unflatten(bias_tree_def, masked_leaves)

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: Could be improved

            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(logs=FrozenDict(_logs), time_step=state.time_step + 1)
            new_params.update(excluded)  # TODO

            # Reset optim, i.e. Adamw params
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)

            return {"params": new_params}, new_state, _tx_state

        return jax.lax.cond(state.time_step % update_frequency == 0, _redo, no_update, updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
