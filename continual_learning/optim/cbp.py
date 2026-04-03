import flax
from flax import struct
import flax.traverse_util
from flax.core import FrozenDict
from jaxtyping import (
    Array,
    Float,
    Bool,
    PRNGKeyArray,
    PyTree,
)
from typing import Tuple, Callable
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from functools import partial

from continual_learning.types import GradientTransformationExtraArgsReset
import continual_learning.utils.optim as utils


class CbpOptimState(struct.PyTreeNode):
    utilities: Float[Array, "#n_layers"]
    ages: Float[Array, "#n_layers"]
    rng: PRNGKeyArray
    remainder: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, "#n_layers"] | None = None
    logs: FrozenDict = FrozenDict(
        {"avg_age": 0, "nodes_reset": 0, "avg_util": 0, "std_util": 0}
    )


def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],  # Vmap over batches
    decay_rate: float = 0.9,
):
    updated_utility = (
        (decay_rate * utility)
        + ((1 - decay_rate) * jnp.abs(features)).mean(axis=tuple(range(features.ndim - 1)))
        * out_w_mag
    ).flatten()  # Arr[#neurons]

    return updated_utility


def get_updated_mfa(mean_feature_act, layer_feats, decay_rate):
    return mean_feature_act * decay_rate + (1 - decay_rate) * layer_feats.mean(axis=0)


def bias_correction(weights, biases, mean_feature_act, ages, reset_mask, decay_rate):
    all_layers = list(weights.keys())
    corrected_biases = {}
    corrected_biases[all_layers[0]] = biases[
        all_layers[0]
    ]  # No outbound correction for 1st layer

    for layer_id in range(len(weights.keys()) - 1):
        layer_name = all_layers[layer_id]
        next_layer_name = all_layers[layer_id + 1]

        next_bias = biases[next_layer_name]
        next_weights = weights[next_layer_name]
        mask = reset_mask[layer_name]

        bias_correction_factor = 1 - decay_rate ** (ages[layer_name] + 1)

        correction_term = jnp.where(
            mask, mean_feature_act[layer_name] / jnp.maximum(bias_correction_factor, 1e-6), 0.0
        )

        # Tile for conv→dense transition (spatial flattening)
        curr_weights = weights[layer_name]
        if len(curr_weights.shape) == 4 and len(next_weights.shape) == 2:
            spatial_size = next_weights.shape[0] // correction_term.size
            correction_term = jnp.tile(correction_term, spatial_size)

        expanded_correction_term = utils.expand_mask_for_weights(
            correction_term, next_weights.shape, mask_type="outgoing"
        )
        bias_correction_delta = (next_weights * expanded_correction_term).sum(
            axis=tuple(range(len(next_weights.shape) - 1))
        )

        corrected_biases[next_layer_name] = next_bias + bias_correction_delta

    # Skip last layer
    # output_layer_name = all_layers[-1]
    # corrected_biases[output_layer_name] = biases[output_layer_name]

    return corrected_biases


def get_reset_mask(
    updated_utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    remainder: Float[Array, "#neurons"],
    key: PRNGKeyArray,
    maturity_threshold: int = 100,
    replacement_rate: float = 0.01,
    accumulate: bool = False,
) -> tuple[Bool[Array, "#neurons"], int | Float[Array, "#neurons"]]:
    # ages+1 because cbp updates ages first, whereas we do it together with resetting
    maturity_mask = (
        ages + 1 > maturity_threshold
    )  # get nodes over maturity threshold Arr[Bool]

    ideally_replace = (jnp.sum(maturity_mask) * replacement_rate) + remainder  # float
    n_to_replace = jnp.floor(ideally_replace).astype(jnp.int32)  # int
    remainder = ideally_replace - n_to_replace  # float

    if not accumulate:
        top_up = jnp.where(random.uniform(key, shape=()) < remainder, 1.0, 0.0)
        remainder = jnp.zeros_like(remainder)
    else:
        n_mature = jnp.sum(maturity_mask)
        ideally = n_mature * replacement_rate
        n_to_replace = jnp.floor(ideally).astype(jnp.int32)
        frac = ideally - n_to_replace
        top_up = (random.uniform(key, ()) < frac).astype(jnp.int32)
        remainder = jnp.zeros_like(remainder)

    masked_utility = jnp.where(
        maturity_mask, updated_utility, jnp.inf
    )  # Immature nodes are inf to avoid replacing
    k_masked_utility = utils.get_bottom_k_mask(masked_utility, n_to_replace + top_up)  # bool

    return k_masked_utility, remainder


def cbp(
    seed: int,
    replacement_rate: float = 1e-4,
    decay_rate: float = 0.99,
    maturity_threshold: int = 1000,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str | None = "output",
    accumulate: bool = False,
) -> GradientTransformationExtraArgsReset:
    """Continual Backpropergation (CBP): [Sokar et al.](https://www.nature.com/articles/s41586-024-07711-7)"""

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]
        biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}
        biases.pop(out_layer_name)

        return CbpOptimState(
            utilities=jax.tree.map(lambda layer: jnp.zeros_like(layer), biases),
            # utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            mean_feature_act=jax.tree.map(lambda layer: jnp.zeros_like(layer), biases),
            rng=jax.random.PRNGKey(seed),
            remainder=jax.tree.map(lambda _: 0.0, biases),
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CbpOptimState,
        params: optax.Params,
        features: PyTree,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, CbpOptimState, optax.OptState]:
        def _cbp(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CbpOptimState, optax.OptState]:
            del updates

            flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]
            flat_feats, _ = jax.tree.flatten(features)

            # String-keyed dicts for scoring (q2 overwrites q1 for twin-Q)
            weights_str = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            out_w_mag = utils.get_out_weights_mag(weights_str)

            new_rng, util_key, acc_key = random.split(state.rng, 3)

            # Build acc_tree matching state utilities (string-keyed, no output)
            util_keys = {k: v for k, v in weights_str.items() if k != out_layer_name}
            acc_tree = utils.gen_key_tree(acc_key, util_keys)

            # Features arrive as tuple so we have to restructure
            w_mag_tdef = jax.tree.structure(out_w_mag)
            _features = jax.tree.unflatten(w_mag_tdef, flat_feats[:-1])

            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                out_w_mag,
                state.utilities,
                _features,
            )
            bias_corrected_utility = jax.tree.map(
                lambda u, a: u / jnp.maximum(1.0 - (decay_rate ** (a + 1)), 1e-8),
                _utility,
                state.ages,
            )

            reset_info = jax.tree.map(
                partial(
                    get_reset_mask,
                    maturity_threshold=maturity_threshold,
                    replacement_rate=replacement_rate,
                    accumulate=accumulate,
                ),
                bias_corrected_utility,
                state.ages,
                state.remainder,
                acc_tree,
            )
            reset_mask = jax.tree.map(
                lambda x: x[0], reset_info, is_leaf=lambda x: isinstance(x, tuple)
            )
            remainder = jax.tree.map(
                lambda x: x[1], reset_info, is_leaf=lambda x: isinstance(x, tuple)
            )

            _ages = jax.tree.map(
                lambda a, m: jnp.where(m, jnp.zeros_like(a), a + 1),
                state.ages,
                reset_mask,
            )

            _mean_feature_act = jax.tree.map(
                partial(get_updated_mfa, decay_rate=decay_rate),
                state.mean_feature_act,
                _features,
            )

            # Apply resets per sub-network chain
            weights_full = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases_full = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}
            weight_chains = utils.split_by_chain(weights_full)
            bias_chains = utils.split_by_chain(biases_full)

            _logs = {k: 0 for k in state.logs}
            _rng = util_key

            for prefix in sorted(weight_chains.keys()):
                chain_w = weight_chains[prefix]
                chain_b = bias_chains[prefix]

                _rng, key = random.split(_rng)
                key_tree = utils.gen_key_tree(key, chain_w)

                # reset_weights modifies chain_w in place
                chain_w, reset_logs = utils.reset_weights(
                    key_tree, reset_mask, chain_w, weight_init_fn,
                )

                # Zero biases of reset nodes
                mask_with_output = dict(reset_mask)
                mask_with_output[out_layer_name] = jnp.zeros_like(
                    chain_b[out_layer_name], dtype=bool
                )
                zeroed_chain_b = jax.tree.map(
                    lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=b.dtype), b),
                    mask_with_output, chain_b,
                )

                # Bias correction (uses post-reset chain weights)
                corrected_chain_b = bias_correction(
                    chain_w, zeroed_chain_b, _mean_feature_act,
                    state.ages, reset_mask, decay_rate,
                )

                weight_chains[prefix] = chain_w
                bias_chains[prefix] = corrected_chain_b
                for layer_name in reset_logs:
                    _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            # Reset utility and MFA for reset neurons
            _utility = jax.tree.map(lambda u, m: jnp.where(m, 0.0, u), _utility, reset_mask)
            _mean_feature_act = jax.tree.map(
                lambda mfa, m: jnp.where(m, 0.0, mfa), _mean_feature_act, reset_mask
            )

            # Logging
            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            avg_util = jax.tree.map(lambda v: v.mean(), bias_corrected_utility)
            std_util = jax.tree.map(lambda v: v.std(), bias_corrected_utility)

            for layer_name in reset_mask.keys():
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["avg_util"] += avg_util[layer_name]
                _logs["std_util"] += std_util[layer_name]

            new_state = state.replace(
                ages=_ages,
                logs=FrozenDict(_logs),
                rng=new_rng,
                utilities=_utility,
                mean_feature_act=_mean_feature_act,
                remainder=remainder,
            )

            new_params = utils.reconstruct_params(params, weight_chains, bias_chains)
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)

            return new_params, new_state, _tx_state

        return _cbp(updates)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
