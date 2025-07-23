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
class CBPOptimState:
    utilities: Float[Array, "#n_layers"]
    ages: Float[Array, "#n_layers"]
    rng: PRNGKeyArray
    mean_feature_act: Float[Array, "#n_layers"] | None = None
    logs: FrozenDict = FrozenDict(
        {"avg_age": 0, "nodes_reset": 0, "avg_util": 0, "std_util": 0}
    )


# -------------- CBP Weight reset ---------------
def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    # mean_feature_act: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],  # Vmap over batches
    decay_rate: Float[Array, ""] = 0.9,
):
    updated_utility = (
        (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features).mean(axis=tuple(range(features.ndim-1))) * out_w_mag
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
    """
    Gradient explodes for some reason, needs debugging
    """
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
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
) -> Bool[Array, "#neurons"]:
    # ages+1 because cbp updates ages first, whereas we do it together with resetting
    maturity_mask = (
        ages + 1 > maturity_threshold
    )  # get nodes over maturity threshold Arr[Bool]
    n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int

    # Just mask*updated_utility?
    masked_utility = jnp.where(maturity_mask, updated_utility, jnp.zeros_like(updated_utility))
    k_masked_utility = utils.get_bottom_k_mask(masked_utility, n_to_replace)  # bool

    return k_masked_utility


# -------------- Main CBP Optimiser body ---------------
def cbp(
    seed: int,
    replacement_rate: float = 1e-4,  # Update to paper hyperparams
    decay_rate: float = 0.99,
    maturity_threshold: int = 20,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str | None = "output",
) -> optax.GradientTransformationExtraArgs:
    """Continual Backpropergation (CBP): [Sokar et al.](https://www.nature.com/articles/s41586-024-07711-7)"""

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])

        # if any(fp[0].startswith("conv") for fp in flat_params.keys()):
        #     ordered_params = sorted(flat_params.items(), key=lambda x: x[0][0].split('_')[-1])
        # else:
        #     ordered_params = flat_params

        biases = {k[0]: v for k, v in flat_params.items() if k[-1] == "bias"}
        biases.pop(out_layer_name)

        del params

        return CBPOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            mean_feature_act=jax.tree.map(lambda layer: jnp.zeros_like(layer), biases),
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

            # Separate weights and biases
            flat_params = flax.traverse_util.flatten_dict(params["params"])
            flat_feats, _ = jax.tree.flatten(features)

            # HACK: Added id's to layers in CNN as they arrive alphabetically ordered
            # if any(fp[0].startswith("conv") for fp in flat_params.keys()):
            #     ordered_params = sorted(flat_params.items(), key=lambda x: x[0][0].split('_')[-1])
            #     ordered_feats = sorted(flat_feats.items(), key=lambda x: x[0][0].split('_')[-1])
            # else:
            #     ordered_params = flat_params
            #     ordered_feats = flat_feats

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
            _weights, reset_logs = utils.reset_weights(
                key_tree,
                reset_mask,  # No out_layer
                weights,  # Yes out_layer
                weight_init_fn,  # state.initial_weights
            )

            _mean_feature_act = jax.tree.map(
                partial(get_updated_mfa, decay_rate=decay_rate), state.mean_feature_act, _features
            )

            # Update ages (clip to prevent huge values)
            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), a+1
                ),
                state.ages,
                reset_mask,
            )

            # zero
            zeroed_biases = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
                reset_mask | {out_layer_name: jnp.zeros_like(biases[out_layer_name])},
                biases,
            )

            # Bias correction
            # corrected_biases = partial(bias_correction, decay_rate=decay_rate)(
            #     weights,  # Uses original weights
            #     zeroed_biases,
            #     _mean_feature_act,
            #     _ages,  # 2
            #     reset_mask
            # )
            corrected_biases = biases # TODO: DISABLED FOR NOW

            new_params = {}
            _logs = {k: 0 for k in state.logs}

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            avg_util = jax.tree.map(lambda v: v.mean(), _utility)
            std_util = jax.tree.map(lambda v: v.std(), _utility)

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
            )
            # Reset optim, i.e. Adamw params
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)
            return {"params": new_params}, new_state, _tx_state

        return _cbp(updates)

    return optax.GradientTransformationExtraArgs(init=init, update=update)
