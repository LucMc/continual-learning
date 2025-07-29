import flax
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
def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],
    decay_rate: Float[Array, ""] = 0.9,
):
    # Remove batch dim from some inputs just in case
    updated_utility = (
        (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features).mean(axis=tuple(range(features.ndim-1))) * out_w_mag
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
def ccbp(
    seed: int,
    replacement_rate: float = 0.5,
    decay_rate: float = 0.9,
    maturity_threshold: int = 10,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str = "output"
) -> optax.GradientTransformationExtraArgs:
    """ Continuous Continual Backpropergation (CCBP) """

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])
        biases = {k[0]: v for k, v in flat_params.items() if k[-1] == 'bias'}
        biases.pop(out_layer_name)


        del params

        return CBPOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            mean_feature_act=jax.tree.map(lambda layer: jnp.zeros_like(layer), biases), # TODO: Remove
            rng=jax.random.PRNGKey(seed),
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
        def _ccbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            weights = {k[0]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[0]: v for k, v in flat_params.items() if k[-1] == "bias"}
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
            _weights, reset_logs = utils.continuous_reset_weights(
                key_tree,
                reset_mask,  # No out_layer
                weights,  # Yes out_layer
                _utility,
                weight_init_fn,
                replacement_rate
            )

            # reset bias given mask
            # Expermiment: reset bias/continuous reset bias/ leave bias alone/ bias correction
            _biases = biases

            # Update ages (CLIPPED HERE)
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

            # Logging TODO: Add stats from continuous resetweights
            for layer_name in weights.keys():  # Exclude output layer
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _biases[layer_name],
                }

            for layer_name in reset_mask.keys():
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["avg_util"] += avg_util[layer_name]
                _logs["std_util"] += std_util[layer_name]

                # _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                ages=_ages, logs=FrozenDict(_logs), rng=new_rng, utilities=_utility
            )

            return {"params": new_params}, new_state, tx_state

        return _ccbp(updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
