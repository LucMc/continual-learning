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
from numpy import mean
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from dataclasses import field

import continual_learning_2.utils.optim as utils


@dataclass
class CbpOptimState:
    utilities: Float[Array, "#n_layers"]
    ages: Float[Array, "#n_layers"]
    rng: PRNGKeyArray
    remainder: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, "#n_layers"] | None = None
    logs: FrozenDict = FrozenDict(
        {"avg_age": 0, "nodes_reset": 0, "avg_util": 0, "std_util": 0}
    )


# -------------- CBP Weight reset ---------------
def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],  # Vmap over batches
    decay_rate: Float[Array, ""] = 0.9,
):
    updated_utility = (
        ((1-decay_rate) * utility) + (decay_rate * jnp.abs(features)).mean(axis=tuple(range(features.ndim-1))) * out_w_mag
    ).flatten()  # Arr[#neurons]

    return updated_utility


def get_updated_mfa(mean_feature_act, layer_feats, decay_rate):
    return mean_feature_act * decay_rate + (1 - decay_rate) * layer_feats.mean(axis=0)

def bias_correction(
    weights,
    biases,
    mean_feature_act,
    ages,
    reset_mask,
    decay_rate
):
    all_layers = list(weights.keys())
    corrected_biases = {}
    corrected_biases[all_layers[0]] = biases[all_layers[0]] # No outbound correction for 1st layer
    
    for layer_id in range(len(weights.keys()) - 1):
        layer_name = all_layers[layer_id]
        next_layer_name = all_layers[layer_id + 1]

        next_bias = biases[next_layer_name]
        next_weights = weights[next_layer_name]
        mask = reset_mask[layer_name]
        
        bias_correction_factor = 1 - decay_rate ** ages[layer_name]
        
        correction_term = jnp.where(
            mask,
            mean_feature_act[layer_name] / jnp.maximum(bias_correction_factor, 1e-8),
            0.0
        )
        
        expanded_correction_term = utils.expand_mask_for_weights(correction_term, next_weights.shape, mask_type='outgoing')
        bias_correction_delta = (next_weights * expanded_correction_term).sum(axis=tuple(range(len(next_weights.shape) - 1)))
        
        corrected_biases[next_layer_name] = next_bias + bias_correction_delta
    
    # Skip last layer
    output_layer_name = all_layers[-1]
    corrected_biases[output_layer_name] = biases[output_layer_name]
    
    return corrected_biases

# -------------- lowest utility mask ---------------
def get_reset_mask(
    updated_utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    remainder: Float[Array, "#neurons"],
    key: PRNGKey,
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
    accumulate: Bool[Array, ""] = False,
) -> Bool[Array, "#neurons"]:
    # ages+1 because cbp updates ages first, whereas we do it together with resetting
    maturity_mask = (
        ages + 1 > maturity_threshold
    )  # get nodes over maturity threshold Arr[Bool]

    ideally_replace = (jnp.sum(maturity_mask) * replacement_rate) + remainder # float
    n_to_replace = jnp.floor(ideally_replace).astype(jnp.int32) # int
    remainder = ideally_replace - n_to_replace # float

    if not accumulate:
        top_up = jnp.where(
            random.uniform(key, shape=()) < remainder,
            1.0,
            0.0
        )
        

    # Just mask*updated_utility?
    masked_utility = jnp.where(maturity_mask, updated_utility, jnp.inf) # Immature nodes are inf to avoid replacing
    k_masked_utility = utils.get_bottom_k_mask(masked_utility, n_to_replace + top_up)  # bool

    return k_masked_utility, remainder


# -------------- Main CBP Optimiser body ---------------
def cbp(
    seed: int,
    replacement_rate: float = 1e-4,
    decay_rate: float = 0.99,
    maturity_threshold: int = 1000,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str | None = "output",
    accumulate: bool = False
) -> optax.GradientTransformationExtraArgs:
    """Continual Backpropergation (CBP): [Sokar et al.](https://www.nature.com/articles/s41586-024-07711-7)"""

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])
        biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
        biases.pop(out_layer_name)

        del params

        return CbpOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            mean_feature_act=jax.tree.map(lambda layer: jnp.zeros_like(layer), biases),
            rng=jax.random.PRNGKey(seed),
            remainder=jax.tree.map(lambda x: 0.0, biases),
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CbpOptimState,
        params: optax.Params,
        features: Array,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, CbpOptimState]:
        def _cbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CbpOptimState]:

            # Separate weights and biases
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            weights = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
            out_w_mag = utils.get_out_weights_mag(weights)

            new_rng, util_key, acc_key = random.split(state.rng, 3)
            key_tree = utils.gen_key_tree(util_key, weights)
            acc_tree = utils.gen_key_tree(util_key, weights)
            acc_tree.pop(out_layer_name)  # Add this line to remove the output layer


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
            bias_corrected_utility = jax.tree.map(
                lambda u, a: u / jnp.maximum(1.0 - (decay_rate ** (a + 1)), 1e-8),
                _utility, state.ages
            )


            reset_info = jax.tree.map(
                partial(
                    get_reset_mask,
                    maturity_threshold=maturity_threshold,
                    replacement_rate=replacement_rate,
                    accumulate=accumulate
                ),
                bias_corrected_utility,
                state.ages,
                state.remainder,
                acc_tree
            )
            reset_mask = jax.tree.map(lambda x: x[0], reset_info, is_leaf=lambda x: isinstance(x, tuple))
            remainder = jax.tree.map(lambda x: x[1], reset_info, is_leaf=lambda x: isinstance(x, tuple))

            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), a+1
                ),
                state.ages,
                reset_mask,
            )

            # reset weights given mask
            _weights, reset_logs = utils.reset_weights(
                key_tree,
                reset_mask,  # No out_layer
                weights,  # Yes out_layer
                weight_init_fn,  # state.initial_weights
            )

            _mean_feature_act = jax.tree.map(
                partial(get_updated_mfa, decay_rate=decay_rate), state.mean_feature_act, _features
            )

            # zero biases of reset nodes
            zeroed_biases = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=b.dtype), b),
                reset_mask | {out_layer_name: jnp.zeros_like(biases[out_layer_name])},
                biases,
            )

            # Bias correction
            corrected_biases = partial(bias_correction, decay_rate=decay_rate)(
                weights,  # Uses original weights
                zeroed_biases,
                _mean_feature_act,
                _ages,  # 2
                reset_mask
            )

            new_params = {}
            _logs = {k: 0 for k in state.logs}

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            avg_util = jax.tree.map(lambda v: v.mean(), bias_corrected_utility)
            std_util = jax.tree.map(lambda v: v.std(), bias_corrected_utility)

            # Logging
            for layer_name in weights.keys():  # Exclude output layer
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": corrected_biases[layer_name],
                }

            for layer_name in reset_mask.keys():
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["avg_util"] += avg_util[layer_name]
                _logs["std_util"] += std_util[layer_name]

                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                ages=_ages,
                logs=FrozenDict(_logs),
                rng=new_rng,
                utilities=_utility,
                mean_feature_act=_mean_feature_act,
                remainder=remainder
            )
            # Reset optim, i.e. Adamw params
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)
            flat_new_params, _ = jax.tree.flatten(new_params)

            return jax.tree.unflatten(jax.tree.structure(params), flat_new_params), new_state, _tx_state

        return _cbp(updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
