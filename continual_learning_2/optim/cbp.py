import flax
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

import continual_learning_2.utils.optim as utils

"""
This is an implementation of continual back propergation (CBP): """

@dataclass
class CBPOptimState:
    utilities: Float[Array, "#n_layers"]
    ages: Array
    rng: PRNGKeyArray
    logs: FrozenDict = FrozenDict({"avg_age": 0, "nodes_reset": 0, "avg_util": 0, "std_util": 0})


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

    masked_utility = jnp.where(maturity_mask, updated_utility, jnp.zeros_like(updated_utility))
    k_masked_utility = utils.get_bottom_k_mask(masked_utility, n_to_replace)  # bool

    return k_masked_utility


# -------------- Main CBP Optimiser body ---------------
def cbp(
    seed: int,
    replacement_rate: float = 0.5,  # Update to paper hyperparams
    decay_rate: float = 0.9,
    maturity_threshold: int = 10,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    # out_layer_name: str = 
) -> optax.GradientTransformationExtraArgs:
    """ Continual Backpropergation (CBP): [Sokar et al.](https://www.nature.com/articles/s41586-024-07711-7) """

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])
        biases = {k[0]: v for k, v in flat_params.items() if k[-1] == 'bias'}

        del params

        return CBPOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            rng=jax.random.PRNGKey(seed),
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CBPOptimState,
        params: optax.Params,
        features: Array,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, CBPOptimState]:
        def _cbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            # assert features, "Features must be provided in update"
            # _features = features["activations"][0]
            out_layer_name = "output"

            # Exclude final layer as it has no out weight mag
            excluded = params["params"].pop(out_layer_name)
            features.pop(out_layer_name+"_act")

            flat_params = flax.traverse_util.flatten_dict(params["params"])

            weights = {k[0]: v for k, v in flat_params.items() if k[-1] == 'kernel'}
            biases = {k[0]: v for k, v in flat_params.items() if k[-1] == 'bias'}

            out_w_mag = utils.get_out_weights_mag(weights)

            # weights, bias, out_w_mag, excluded = utils.process_params_with_outmag(params["params"])

            # new_rng, util_key = random.split(rng)
            key_tree = utils.gen_key_tree(state.rng, weights)

            # vmap utility calculation over batch
            batched_util_calculation = jax.vmap(
                partial(get_updated_utility, decay_rate=decay_rate),
                in_axes=(None, None, 0),
            )

            # Flatten all the structures (for unmatched trees)
            out_w_mag_leaves, _ = jax.tree.flatten(out_w_mag)
            utilities_leaves, utilities_tree_def = jax.tree.flatten(state.utilities)
            feature_leaves = [f[0] for f in jax.tree.leaves(features)]

            # Apply the calculation to the leaves
            utility_batch_leaves = [
                batched_util_calculation(w, u, f)
                for w, u, f in zip(out_w_mag_leaves, utilities_leaves, feature_leaves)
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
            _weights, reset_logs = utils.new_reset_weights(
                key_tree, reset_mask, weights, weight_init_fn # state.initial_weights
            )

            # reset bias given mask
            _bias = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
                reset_mask,
                bias
            )

            # Update ages
            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), jnp.clip(a + 1, max=maturity_threshold+1)
                ),  # Clip to stop huge ages
                state.ages,
                reset_mask,
            )

            new_params = {}
            _logs = {k: 0 for k in state.logs}

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

            # Reset optim, i.e. Adamw params
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)
            return {"params": new_params}, new_state, _tx_state

        return _cbp(updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
