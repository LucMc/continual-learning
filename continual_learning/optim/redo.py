from flax import struct
import flax
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

from continual_learning.types import GradientTransformationExtraArgsReset
import continual_learning.utils.optim as utils


class RedoOptimState(struct.PyTreeNode):
    rng: PRNGKeyArray
    time_step: int = 0
    logs: FrozenDict = FrozenDict({"nodes_reset": 0})


def get_score(
    features: Float[Array, "#batch #neurons"],
) -> Float[Array, "#neurons"]:
    # Avg over other dims
    reduce_axes = tuple(range(features.ndim - 1))
    mean_act_per_neuron = jnp.mean(jnp.abs(features), axis=reduce_axes)  # Arr[#neurons]
    score = mean_act_per_neuron / (
        jnp.mean(mean_act_per_neuron) + 1e-8
    )  # Arr[#neurons] / Scalar
    return score


def redo(
    seed: int,
    update_frequency: int = 1000,
    score_threshold: float = 0.0095,
    max_reset_frac: float | None = None,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
) -> GradientTransformationExtraArgsReset:
    """Recycle Dormant Neurons (ReDo): [Sokar et al.](https://arxiv.org/pdf/2302.12902)"""

    def init(params: optax.Params, **kwargs):
        del params

        return RedoOptimState(
            rng=jax.random.PRNGKey(seed),
            **kwargs,
        )

    def get_reset_mask(
        scores: Float[Array, "#neurons"],
    ) -> Bool[Array, "#neurons"]:
        threshold_mask = (
            scores <= score_threshold
        )  # get nodes over maturity threshold Arr[Bool]

        if (max_reset_frac is None) or (max_reset_frac <= 0.0):
            return threshold_mask

        size = scores.shape[-1]
        k_max = jnp.asarray(jnp.floor(max_reset_frac * size), dtype=jnp.int32)
        n_in = jnp.asarray(jnp.sum(threshold_mask), dtype=jnp.int32)
        k_eff = jnp.minimum(k_max, n_in)
        gated_scores = jnp.where(threshold_mask, scores, jnp.inf)
        return utils.get_bottom_k_mask(gated_scores, k_eff)

    def _feature_key_to_path(key: str) -> tuple:
        """Convert flattened feature key to path tuple.

        e.g., "q1/main/layer_0_act" -> ("q1", "main", "layer_0")
        """
        # Remove _act or _pre suffix
        if key.endswith("_act"):
            key = key[:-4]
        elif key.endswith("_pre"):
            key = key[:-4]
        return tuple(key.split("/"))

    def _is_output_layer(path: tuple) -> bool:
        """Check if this is an output layer that shouldn't be reset."""
        return path[-1] == "output" or path[-1].endswith("_output")

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: RedoOptimState,
        params: optax.Params,
        features: PyTree,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, RedoOptimState, optax.OptState]:
        def no_update(updates):
            del updates
            new_state = state.replace(time_step=state.time_step + 1)
            return params, new_state, tx_state

        def _redo(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, RedoOptimState, optax.OptState]:
            del updates

            flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]

            # Use full path (except last element) as key to avoid collisions with nested networks
            # e.g., ("q1", "main", "layer_0", "kernel") -> ("q1", "main", "layer_0")
            weights = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}

            # Flatten nested feature dicts (e.g., {'main': {'layer_0_act': ...}})
            # into flat dict with "/" separators (e.g., {'main/layer_0_act': ...})
            flat_features = utils.flatten_features(features)

            # Compute dormancy scores from activations
            # Features are flattened like "q1/main/layer_0_act" -> convert to path tuple
            scores = {}
            for key, feature_tuple in flat_features.items():
                path = _feature_key_to_path(key)
                # feature_tuple is a tuple of (activation_array,)
                scores[path] = get_score(feature_tuple[0] if isinstance(feature_tuple, tuple) else feature_tuple)

            reset_mask = jax.tree.map(get_reset_mask, scores)

            # Exclude output layers from resets
            for path in list(reset_mask.keys()):
                if _is_output_layer(path):
                    reset_mask[path] = jnp.zeros_like(reset_mask[path], dtype=bool)

            _rng, key = random.split(state.rng)

            # Filter to only layers that exist in both weights and reset_mask
            layers_to_reset = [k for k in weights.keys() if k in reset_mask]
            weights_to_reset = {k: weights[k] for k in layers_to_reset}
            key_tree_filtered = utils.gen_key_tree(key, weights_to_reset)

            # Reset weights
            _weights, reset_logs = utils.reset_weights(
                key_tree_filtered,
                reset_mask,
                weights_to_reset,
                weight_init_fn,
            )

            # Reset biases for layers that were reset
            _biases = {}
            for layer_path in layers_to_reset:
                if layer_path in biases and layer_path in reset_mask:
                    mask = reset_mask[layer_path]
                    bias = biases[layer_path]
                    _biases[layer_path] = jnp.where(mask, jnp.zeros_like(bias, dtype=bias.dtype), bias)

            # Build new flat params by updating original with reset values
            new_flat_params = {}
            _logs = {k: 0 for k in state.logs}
            counted_layers = set()

            for path, value in flat_params.items():
                layer_path = path[:-1]  # Full path without 'kernel'/'bias'
                param_type = path[-1]  # "kernel", "bias", "scale", etc.

                if layer_path in _weights and param_type == "kernel":
                    new_flat_params[path] = _weights[layer_path]
                    if layer_path in reset_logs and layer_path not in counted_layers:
                        _logs["nodes_reset"] += reset_logs[layer_path]["nodes_reset"]
                        counted_layers.add(layer_path)
                elif layer_path in _biases and param_type == "bias":
                    new_flat_params[path] = _biases[layer_path]
                else:
                    # Keep original value for non-reset params
                    new_flat_params[path] = value

            # Reconstruct params tree, preserving any extra keys from original params
            new_params_dict = flax.traverse_util.unflatten_dict(new_flat_params)
            new_params = {"params": new_params_dict}
            # Copy over any extra keys (e.g., activations, preactivations)
            for key in params:
                if key not in new_params:
                    new_params[key] = params[key]

            new_state = state.replace(
                logs=FrozenDict(_logs), time_step=state.time_step + 1, rng=_rng
            )

            # Reset optimizer momentum
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)

            return (new_params, new_state, _tx_state)

        condition = jnp.logical_and(
            state.time_step > 0, (state.time_step % update_frequency == 0)
        )
        return jax.lax.cond(condition, _redo, no_update, updates)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
