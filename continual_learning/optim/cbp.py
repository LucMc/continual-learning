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
        # Accumulate mode: use fractional remainder from previous step
        n_mature = jnp.sum(maturity_mask)
        ideally = n_mature * replacement_rate + remainder
        n_to_replace = jnp.floor(ideally).astype(jnp.int32)
        frac = ideally - n_to_replace
        top_up = (random.uniform(key, ()) < frac).astype(jnp.int32)
        remainder = frac  # Carry over fractional part

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

    def _is_output_layer(path: tuple) -> bool:
        """Check if this is an output layer that shouldn't be reset."""
        return path[-1] == "output" or path[-1].endswith("_output") or (out_layer_name is not None and path[-1] == out_layer_name)

    def _feature_key_to_path(key: str) -> tuple:
        """Convert flattened feature key to path tuple.

        e.g., "BroNet_0/Dense_0_act" -> ("BroNet_0", "Dense_0")
        """
        # Remove _act or _pre suffix
        if key.endswith("_act"):
            key = key[:-4]
        elif key.endswith("_pre"):
            key = key[:-4]
        return tuple(key.split("/"))

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]
        # Use kernels (not biases) to determine tracked layers - avoids LayerNorm mismatch
        # LayerNorm has bias but no kernel (only scale), which would cause tree structure mismatch
        kernels = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
        # Remove output layers
        kernels = {k: v for k, v in kernels.items() if not _is_output_layer(k)}

        return CbpOptimState(
            # Utilities shape = [num_neurons] = kernel output dim (last axis)
            utilities=jax.tree.map(lambda kernel: jnp.zeros(kernel.shape[-1]), kernels),
            ages=jax.tree.map(lambda kernel: jnp.zeros(kernel.shape[-1]), kernels),
            mean_feature_act=jax.tree.map(lambda kernel: jnp.zeros(kernel.shape[-1]), kernels),
            rng=jax.random.PRNGKey(seed),
            remainder=jax.tree.map(lambda _: 0.0, kernels),
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

            # Separate weights and biases using full paths to avoid collisions
            flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]

            # Use full path (except last element) as key to avoid collisions with nested networks
            weights = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}

            # Filter out output layers for processing
            weights_no_output = {k: v for k, v in weights.items() if not _is_output_layer(k)}
            # Only include biases for layers that have kernels (excludes LayerNorm which has scale, not kernel)
            biases_no_output = {k: v for k, v in biases.items() if not _is_output_layer(k) and k in weights_no_output}

            out_w_mag = utils.get_out_weights_mag(weights_no_output)

            new_rng, util_key, acc_key = random.split(state.rng, 3)
            key_tree = utils.gen_key_tree(util_key, weights_no_output)
            acc_tree = utils.gen_key_tree(acc_key, weights_no_output)

            # Flatten nested feature dicts (e.g., {'main': {'layer_0_act': ...}})
            # into flat dict with "/" separators (e.g., {'main/layer_0_act': ...})
            flat_features = utils.flatten_features(features)

            # Map features to layer paths using explicit key conversion
            # This handles nested architectures (BRO, twin networks) correctly
            _features = {}
            for key, feat in flat_features.items():
                path = _feature_key_to_path(key)
                if path in weights_no_output:
                    # Handle tuple wrapping from sow() - take first element if tuple
                    _features[path] = feat[0] if isinstance(feat, tuple) else feat

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

            # reset weights given mask
            _weights, reset_logs = utils.reset_weights(
                key_tree,
                reset_mask,
                weights_no_output,
                weight_init_fn,
            )

            _mean_feature_act = jax.tree.map(
                partial(get_updated_mfa, decay_rate=decay_rate),
                state.mean_feature_act,
                _features,
            )

            _utility = jax.tree.map(lambda u, m: jnp.where(m, 0.0, u), _utility, reset_mask)
            _mean_feature_act = jax.tree.map(lambda mfa, m: jnp.where(m, 0.0, mfa), _mean_feature_act, reset_mask)

            # Zero biases of reset nodes
            zeroed_biases = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=b.dtype), b),
                reset_mask,
                biases_no_output,
            )

            # Skip bias correction for nested networks (complex sequential logic)
            # Just use zeroed biases
            corrected_biases = zeroed_biases

            _logs = {k: 0 for k in state.logs}
            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            avg_util = jax.tree.map(lambda v: v.mean(), bias_corrected_utility)
            std_util = jax.tree.map(lambda v: v.std(), bias_corrected_utility)

            # Build new flat params dict with updated weights/biases
            new_flat_params = {}
            counted_layers = set()

            for path, value in flat_params.items():
                layer_path = path[:-1]
                param_type = path[-1]

                if layer_path in _weights and param_type == "kernel":
                    new_flat_params[path] = _weights[layer_path]
                    if layer_path in reset_logs and layer_path not in counted_layers:
                        _logs["nodes_reset"] += reset_logs[layer_path]["nodes_reset"]
                        counted_layers.add(layer_path)
                elif layer_path in corrected_biases and param_type == "bias":
                    new_flat_params[path] = corrected_biases[layer_path]
                else:
                    new_flat_params[path] = value

            # Aggregate logs from state
            for layer_path in reset_mask.keys():
                _logs["avg_age"] += avg_ages[layer_path]
                _logs["avg_util"] += avg_util[layer_path]
                _logs["std_util"] += std_util[layer_path]

            new_state = state.replace(
                ages=_ages,
                logs=FrozenDict(_logs),
                rng=new_rng,
                utilities=_utility,
                mean_feature_act=_mean_feature_act,
                remainder=remainder,
            )

            # Reconstruct params tree
            new_params_dict = flax.traverse_util.unflatten_dict(new_flat_params)
            new_params = {"params": new_params_dict}
            for key in params:
                if key not in new_params:
                    new_params[key] = params[key]

            # Reset optim params
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)

            return (new_params, new_state, _tx_state)

        return _cbp(updates)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
