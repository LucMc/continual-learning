import flax
from chex import dataclass
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
from typing import Tuple, Callable
from chex import dataclass
from numpy import mean
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from dataclasses import field

import continual_learning_2.utils.optim as utils
from continual_learning_2.optim.cbp import CbpOptimState


@dataclass
class CcbpOptimState(CbpOptimState):
    time_step: int = 0
    logs: FrozenDict = FrozenDict(
        {"std_util": 0.0, "nodes_reset": 0.0, "low_utility": 0, "mean_utils": 0.0}
    )


# -------------- CCBP Weight reset ---------------
# def get_updated_utility(  # Add batch dim
#     out_w_mag: Float[Array, "#weights"],
#     utility: Float[Array, "#neurons"],
#     features: Float[Array, "#batch #neurons"],
#     decay_rate: Float[Array, ""] = 0.9,
# ) -> Float[Array, "#neurons"]:
#     # TODO: Mean activations etc over the whole network instead of per layer
#     # Remove batch dim from some inputs just in case
#     reduce_axis = tuple(range(features.ndim - 1))
#     mean_act_per_neuron = jnp.abs(features).mean(axis=reduce_axis)
#
#     # Running stats normalising both out and in utils
#     updated_utility = (
#         (decay_rate * utility)
#         + (1 - decay_rate)
#         * 0.5
#         * (
#             (mean_act_per_neuron / (jnp.mean(mean_act_per_neuron) + 1e-8))  # Inbound stat
#             + (out_w_mag / (jnp.mean(out_w_mag) + 1e-8))  # Outbound stat
#         )
#     ).flatten()  # Arr[#neurons]
#     # avg neuron is arround 1 utility, using relu means min act of 0
#
#     return updated_utility

def get_updated_utility(
    grads: Float[Array, "#batch #inweights #neurons"],
    utility: Float[Array, "#neurons"],
    decay_rate: Float[Array, ""] = 0.9 # 0 means no running stats
) -> Float[Array, "#neurons"]:
    # Avg over other dims

    reduce_axes = tuple(range(grads.ndim - 1))
    mean_grad_per_neuron = jnp.mean(jnp.abs(grads), axis=reduce_axes)  # Arr[#neurons]
    score = mean_grad_per_neuron / (
        jnp.mean(mean_grad_per_neuron) + 1e-8
    )  # Arr[#neurons] / Scalar

    updated_utility = (decay_rate * utility) + (1-decay_rate) * score
    return updated_utility

# -------------- weight reset ---------------
def continuous_reset_weights(
    key_tree: PRNGKeyArray,
    weights: PyTree[Float[Array, "..."]],
    utilities: PyTree[Float[Array, "..."]],
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    replacement_rate: Float[Array, ""] = 0.001,
):
    all_layer_names = list(weights.keys())
    logs = {}

    assert all_layer_names[-1] == "output", "Last layer should be Dense with name 'output'"

    for idx, layer_name in enumerate(all_layer_names[:-1]):
        # Reset incoming weights
        init_weights = weight_init_fn(key_tree[layer_name], weights[layer_name].shape)
        threshold = 0.95 # Where is the hard threshold
        steepness = 16 # How soft before that?

        # transformed_utilities = jax.tree.map(lambda x: 1 - jnp.exp(-steepness * x), utilities) # Sigmoid
        # transformed_utilities = jax.tree.map(
        #     lambda x: 1 - jnp.exp(-steepness * x + (steepness - 1)), utilities
        # )  # Sigmoid

        transformed_utilities = jax.tree.map( lambda x: 1 - jnp.exp(-steepness * x + steepness*threshold), utilities)

        # Softer reset to consider experimenting with!
        # threshold = 0.8 # Where is the hard threshold
        # steepness = 3.3 # How soft before that?
        # replacement_rate /=3 (at least)
        # transformed_utilities = jax.tree.map(
        #     lambda x: 2 - 2jnp.exp(-steepness * (x-threshold)), utilities
        # )  # Sigmoid

        # transformed_utilities = jax.tree.map(lambda x: x, utilities) # Linear
        transformed_utilities = jax.tree.map(lambda x: jnp.clip(x, 0, 1), transformed_utilities)

        # Proportion reset
        reset_prop = replacement_rate * (1 - transformed_utilities[layer_name])
        keep_prop = 1 - reset_prop

        weights[layer_name] = (keep_prop * weights[layer_name]) + (reset_prop * init_weights)

        # Reset outgoing weights
        if idx + 1 < len(all_layer_names):
            next_layer = all_layer_names[idx + 1]
            out_weight_shape = weights[next_layer].shape

            # Handle shape transitions
            if len(out_weight_shape) == 2:  # Dense layer
                if len(weights[layer_name].shape) == 4:  # Previous layer was conv
                    spatial_size = out_weight_shape[0] // transformed_utilities[layer_name].size
                    out_utilities_1d = jnp.repeat(
                        transformed_utilities[layer_name], spatial_size
                    )

                else:  # Dense -> Dense
                    out_utilities_1d = transformed_utilities[layer_name]

            elif len(out_weight_shape) == 4:  # Conv layer
                out_utilities_1d = transformed_utilities[layer_name]

            expanded_utils = utils.expand_mask_for_weights(
                out_utilities_1d, weights[next_layer].shape, mask_type="outgoing"
            )

            out_reset_prop = replacement_rate * (1 - expanded_utils)
            out_keep_prop = 1 - out_reset_prop
            weights[next_layer] = (
                out_keep_prop * weights[next_layer]
            )  # + (out_reset_prop * out_init_weights) # Decay towards zero

        effective_reset = replacement_rate * (1 - transformed_utilities[layer_name]).mean()

        logs[layer_name] = {
            "nodes_reset": effective_reset,
            "low_utility": jnp.sum(utilities[layer_name] < 0.95),
            "mean_utils": jnp.mean(utilities[layer_name]),
        }

    logs[all_layer_names[-1]] = {"nodes_reset": 0.0, "low_utility": 0, "mean_utils": 0.0}

    return weights, logs


# -------------- Main CCBP Optimiser body ---------------
def ccbp(
    seed: int,
    replacement_rate: float = 0.5,
    decay_rate: float = 0.9,
    update_frequency: int = 100,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str = "output",
    sharpness: float = 0.0,
    threshold: float = 0.0
) -> optax.GradientTransformationExtraArgs:
    """Continuous Continual Backpropergation (CCBP)"""

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])
        biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
        biases.pop(out_layer_name)

        del params

        return CcbpOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            mean_feature_act=jax.tree.map(
                lambda layer: jnp.zeros_like(layer), biases
            ),  # TODO: Remove
            rng=jax.random.PRNGKey(seed),
            time_step=0,
            # update_frequency=update_frequency, # TODO: Change to update_frequency
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CcbpOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
        tx_state: optax.OptState | None = None,
    ) -> tuple[optax.Updates, CcbpOptimState]:
        def no_update(updates):
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            weights = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
            flat_updates = flax.traverse_util.flatten_dict(updates["params"])
            weight_grads = {k[-2]: v for k, v in flat_updates.items() if k[-1] == 'kernel'}

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            weight_grads.pop(out_layer_name)
            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                weight_grads,
                state.utilities,
            )
            all_utils = jnp.concatenate([u.flatten() for u in jax.tree.leaves(_utility)])

            _logs = {
                "std_util": all_utils.std(),
                "nodes_reset": 0.0,  # state.logs['nodes_reset'],
                "low_utility": jnp.sum(all_utils < 0.95),
                "mean_utils": all_utils.mean(),
            }

            new_state = state.replace(
                time_step=state.time_step + 1, logs=FrozenDict(_logs), utilities=_utility
            )

            return params, new_state, tx_state

        def _ccbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CcbpOptimState]:
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            weights = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
            # out_w_mag = utils.get_out_weights_mag(weights)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            flat_updates = flax.traverse_util.flatten_dict(updates["params"])
            weight_grads = {k[-2]: v for k, v in flat_updates.items() if k[-1] == 'kernel'}

            weight_grads.pop(out_layer_name)
            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                weight_grads,
                state.utilities,
            )

            # reset weights given mask
            _weights, reset_logs = continuous_reset_weights(
                key_tree,
                # reset_mask,  # No out_layer
                weights,  # Yes out_layer
                _utility,
                weight_init_fn,
                replacement_rate,
            )

            # Expermiment: reset bias/continuous reset bias/ leave bias alone/ bias correction
            _biases = biases

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: kinda sucks for adding logs

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            avg_util = jax.tree.map(lambda v: v.mean(), _utility)
            std_util = jax.tree.map(lambda v: v.std(), _utility)

            # Logging TODO: Add stats from continuous resetweights
            for layer_name in weights.keys():  # Exclude output layer
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _biases[layer_name],
                }

            for layer_name in _utility.keys():
                # _logs["avg_age"] += avg_ages[layer_name]
                # _logs["avg_util"] += avg_util[layer_name]
                _logs["std_util"] += std_util[layer_name]
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]
                _logs["low_utility"] += reset_logs[layer_name]["low_utility"]
                _logs["mean_utils"] += reset_logs[layer_name]["mean_utils"]

            _logs["mean_utils"] /= len(reset_logs.keys())

            # We reset running utilities once used for an update
            # Reset to 1 as this should be the mean of the utility distribution given norm
            new_state = state.replace(
                # ages=_ages,
                logs=FrozenDict(_logs),
                rng=new_rng,
                utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), _utility),
                # utilities=_utility, # Try with and without keeping the runing average
                time_step=state.time_step + 1,
            )
            flat_new_params, _ = jax.tree.flatten(new_params)
            # TODO: Update bias and tx_state

            return (
                jax.tree.unflatten(jax.tree.structure(params), flat_new_params),
                new_state,
                tx_state,
            )

        return jax.lax.cond(
            state.time_step % update_frequency == 0, _ccbp, no_update, updates
        )

    return optax.GradientTransformationExtraArgs(init=init, update=update)

