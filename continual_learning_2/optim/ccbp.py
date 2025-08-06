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
from continual_learning_2.optim.cbp import CBPOptimState


@dataclass
class CCBPOptimState(CBPOptimState):
    time_step: int = 0
    logs: FrozenDict = FrozenDict(
        {"std_util": 0.0,
         "nodes_reset": 0.0,
         "clipped_utils": 0}
    )


# -------------- CCBP Weight reset ---------------
def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],
    decay_rate: Float[Array, ""] = 0.9,
) -> Float[Array, "#neurons"]:
    # TODO: Mean activations etc over the whole network instead of per layer
    # Remove batch dim from some inputs just in case
    reduce_axis = tuple(range(features.ndim - 1))
    mean_act_per_neuron = jnp.abs(features).mean(axis=reduce_axis)

    # Running stats normalising both out and in utils
    updated_utility = (
        (decay_rate * utility) 
        + (1 - decay_rate) * (
                    (mean_act_per_neuron / (jnp.mean(mean_act_per_neuron) + 1e-8)) # Inbound stat
                    + (out_w_mag / (jnp.mean(out_w_mag) + 1e-8)) # Outbound stat *to+
            )
    ).flatten()  # Arr[#neurons]
    # avg neuron is arround 1 utility, using relu means min act of 0

    steepness = 10
    squish = lambda x: -jnp.e**(-steepness*x)+2 # +1 because updated_utility centers ~1 and out_w_mag
    return squish(updated_utility) # -1 to recenter around 0

# Replacement rate is a linear factor, steepness is how linear/exponential do we want the tradeoff to be
# Not a great name for it, as we still do the weight decay thing when replacement_rate is 0
# It works more mathematically as like a threshold

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

        # Clip so that we don't move beyond target weights, shouldn't be clipped anyway
        reset_prob = replacement_rate * (1 - utilities[layer_name])
        keep_prob = 1 - reset_prob

        weights[layer_name] = (keep_prob * weights[layer_name]) + (reset_prob * init_weights)

        
        # Reset outgoing weights
        if idx + 1 < len(all_layer_names):
            next_layer = all_layer_names[idx + 1]
            out_weight_shape = weights[next_layer].shape

            # Handle shape transitions
            if len(out_weight_shape) == 2:  # Dense layer
                if len(weights[layer_name].shape) == 4:  # Previous layer was conv
                    spatial_size = out_weight_shape[0] // in_mask_1d.size
                    out_utilities_1d = jnp.repeat(utilities[layer_name], spatial_size)

                else:  # Dense -> Dense
                    out_utilities_1d = utilities[layer_name]

            elif len(out_weight_shape) == 4:  # Conv layer
                out_utilities_1d = utilities[layer_name]

            expanded_utils = utils.expand_mask_for_weights(
                out_utilities_1d, weights[next_layer].shape, mask_type="outgoing"
            )

            # resetting to zero is aggressive, there is a reason we use random weights not zeros
            # this needs to be verified if I want to add it to papers claims tho
            out_init_weights = weight_init_fn(key_tree[next_layer], weights[next_layer].shape)
            out_reset_prob = replacement_rate * (1 - expanded_utils)
            out_keep_prob = 1 - out_reset_prob
            weights[next_layer] = (out_keep_prob * weights[next_layer]) + (out_reset_prob * out_init_weights)

        effective_reset = replacement_rate * (1 - utilities[layer_name]).mean()

        logs[layer_name] = {
            "nodes_reset": effective_reset,
            "clipped_utils": jnp.sum(utilities[layer_name] > 1)
        }

    logs[all_layer_names[-1]] = {
        "nodes_reset": 0.0,
        "clipped_utils": 0,
    }

    return weights, logs
# -------------- Main CCBP Optimiser body ---------------
def ccbp(
    seed: int,
    replacement_rate: float = 0.5,
    decay_rate: float = 0.9,
    maturity_threshold: int = 100,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str = "output",
) -> optax.GradientTransformationExtraArgs:
    """Continuous Continual Backpropergation (CCBP)"""

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])
        biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
        biases.pop(out_layer_name)

        del params

        return CCBPOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            mean_feature_act=jax.tree.map(
                lambda layer: jnp.zeros_like(layer), biases
            ),  # TODO: Remove
            rng=jax.random.PRNGKey(seed),
            time_step=0,
            # update_frequency=maturity_threshold, # TODO: Change to update_frequency
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CCBPOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
        tx_state: optax.OptState | None = None,
    ) -> tuple[optax.Updates, CCBPOptimState]:
        def no_update(updates):
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            weights = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
            out_w_mag = utils.get_out_weights_mag(weights)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            # Features arrive as tuple so we have to restructure
            w_mag_tdef = jax.tree.structure(out_w_mag)

            # Don't need out_layer feats and normalises layer names
            _features = jax.tree.unflatten(w_mag_tdef, flat_feats[:-1])

            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                out_w_mag,
                state.utilities,
                _features,
            )
            _logs = {'std_util': jax.tree.reduce(jnp.add, jax.tree.map(lambda v: v.std(), _utility)),
                     'nodes_reset': state.logs['nodes_reset'],
                     'clipped_utils': state.logs['clipped_utils']}

            new_state = state.replace(time_step=state.time_step + 1, logs=FrozenDict(_logs))
         
            return params, new_state, tx_state

        def _ccbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CCBPOptimState]:
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            weights = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
            out_w_mag = utils.get_out_weights_mag(weights)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            # Features arrive as tuple so we have to restructure
            w_mag_tdef = jax.tree.structure(out_w_mag)

            # Don't need out_layer feats and normalises layer names
            _features = jax.tree.unflatten(w_mag_tdef, flat_feats[:-1])

            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                out_w_mag,
                state.utilities,
                # _mean_feature_act,
                _features,
            )

            # reset_mask = jax.tree.map(
            #     partial(
            #         get_reset_mask,
            #         maturity_threshold=maturity_threshold,
            #         replacement_rate=replacement_rate,
            #     ),
            #     _utility,
            #     state.ages,
            # )
            #
            # reset weights given mask
            _weights, reset_logs = continuous_reset_weights(
                key_tree,
                # reset_mask,  # No out_layer
                weights,  # Yes out_layer
                _utility,
                weight_init_fn,
                replacement_rate,
            )

            # reset bias given mask
            # Expermiment: reset bias/continuous reset bias/ leave bias alone/ bias correction
            _biases = biases

            # Update ages (CLIPPED HERE)
            # _ages = jax.tree.map(
            #     lambda a, m: jnp.where(
            #         m, jnp.zeros_like(a), jnp.clip(a + 1, max=maturity_threshold + 1)
            #     ),  # Clip to stop huge ages
            #     state.ages,
            #     reset_mask,
            # )

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
                _logs["clipped_utils"] += reset_logs[layer_name]["clipped_utils"]

            # We reset running utilities once used for an update
            # Reset to 1 as this should be the mean of the utility distribution given norm
            new_state = state.replace(
                # ages=_ages,
                logs=FrozenDict(_logs),
                rng=new_rng,
                utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), _utility),
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
            state.time_step % maturity_threshold == 0, _ccbp, no_update, updates
        )

    return optax.GradientTransformationExtraArgs(init=init, update=update)
