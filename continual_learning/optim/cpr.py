from functools import partial
from typing import Callable, Tuple, Literal

import flax
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.core import FrozenDict
import flax.traverse_util
from jaxtyping import (
    Array,
    Float,
    PRNGKeyArray,
    PyTree,
)

from continual_learning.types import GradientTransformationExtraArgsReset
import continual_learning.utils.optim as utils
from continual_learning.optim.cbp import CbpOptimState


class CprOptimState(CbpOptimState):
    time_step: int = 0
    logs: FrozenDict = FrozenDict(
        {"std_util": 0.0, "nodes_reset": 0.0, "low_utility": 0, "mean_utils": 0.0}
    )


def get_updated_utility(
    grads: Float[Array, "#batch #inweights #neurons"],
    utility: Float[Array, "#neurons"],
    decay_rate: float = 0.9,  # 0 means no running stats
) -> Float[Array, "#neurons"]:
    # Avg over other dims

    reduce_axes = tuple(range(grads.ndim - 1))
    mean_grad_per_neuron = jnp.mean(jnp.abs(grads), axis=reduce_axes)  # Arr[#neurons]
    score = mean_grad_per_neuron / (
        jnp.mean(mean_grad_per_neuron) + 1e-8
    )  # Arr[#neurons] / Scalar

    updated_utility = (decay_rate * utility) + (1 - decay_rate) * score
    return updated_utility


def calibrated_reset_weights(
    key_tree: PRNGKeyArray,
    weights: PyTree[Float[Array, "..."]],
    utilities: PyTree[Float[Array, "..."]],
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    replacement_rate: float = 0.012,
    sharpness: float = 16,
    threshold: float = 0.95,
    transform_type: Literal["exp", "sigmoid", "softplus", "linear"] = "exp",
    out_layer_name: str = "output",
):
    """Partially reset low-utility neurons.

    Keys are full-path tuples (e.g. ('q1','main','0_conv_16_0')) to support
    nested sub-networks like twin Q-critics.  Outgoing-weight resets only
    happen between consecutive layers in the same sub-network chain
    (same key prefix).
    """
    all_keys = list(weights.keys())
    logs = {}

    # Pre-compute next layer within the same sub-network chain
    next_in_chain = {}
    for i, key in enumerate(all_keys):
        if i + 1 < len(all_keys):
            next_key = all_keys[i + 1]
            if key[:-1] == next_key[:-1]:  # Same prefix → same chain
                next_in_chain[key] = next_key

    hidden_keys = [k for k in all_keys if k[-1] != out_layer_name]

    for layer_key in hidden_keys:
        # Reset incoming weights
        init_weights = weight_init_fn(key_tree[layer_key], weights[layer_key].shape)

        match transform_type:
            case "exp": transform = lambda x: jnp.minimum(jnp.exp(-sharpness * (x - threshold)), 1.0)
            case "sigmoid": transform = lambda x: jnp.minimum(2.0 * jax.nn.sigmoid(-sharpness * (x - threshold)), 1.0)
            case "softplus": transform = lambda x: jnp.minimum(jax.nn.softplus(sharpness * (threshold - x)) / jnp.log(2.0), 1.0)
            case "linear": transform = lambda x: jnp.clip(1.0 - sharpness * (x - threshold), 0.0, 1.0)

        transformed_utilities = jax.tree.map(transform, utilities)

        reset_prop = replacement_rate * transformed_utilities[layer_key]

        keep_prop = 1 - reset_prop

        weights[layer_key] = (keep_prop * weights[layer_key]) + (reset_prop * init_weights)

        # Reset outgoing weights (only within same sub-network)
        if layer_key in next_in_chain:
            next_key = next_in_chain[layer_key]
            out_weight_shape = weights[next_key].shape

            # Handle shape transitions
            if len(out_weight_shape) == 2:  # Dense layer
                if len(weights[layer_key].shape) == 4:  # Conv -> Dense
                    spatial_size = (
                        out_weight_shape[0] // transformed_utilities[layer_key].size
                    )
                    out_utilities_1d = jnp.tile(
                        transformed_utilities[layer_key], spatial_size
                    )

                else:  # Dense -> Dense
                    out_utilities_1d = transformed_utilities[layer_key]

            elif len(out_weight_shape) == 4:  # Conv layer
                out_utilities_1d = transformed_utilities[layer_key]
            else:
                raise ValueError(f"Unexpected shape {out_weight_shape}")

            expanded_utils = utils.expand_mask_for_weights(
                out_utilities_1d, weights[next_key].shape, mask_type="outgoing"
            )

            out_reset_prop = replacement_rate * expanded_utils
            out_keep_prop = 1 - out_reset_prop

            weights[next_key] = (
                out_keep_prop * weights[next_key]
            )  # + (out_reset_prop * out_init_weights) # Decay towards zero

        logs[layer_key] = {
            "nodes_reset": reset_prop.mean(),
            "low_utility": jnp.sum(utilities[layer_key] < threshold),
            "mean_utils": jnp.mean(utilities[layer_key]),
        }

    for k in all_keys:
        if k[-1] == out_layer_name:
            logs[k] = {"nodes_reset": 0.0, "low_utility": 0, "mean_utils": 0.0}

    return weights, logs


def cpr(
    seed: int,
    replacement_rate: float = 0.012,
    sharpness: float = 16,
    threshold: float = 0.95,
    decay_rate: float = 0.99,
    update_frequency: int = 1000,
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    out_layer_name: str = "output",
    transform_type: Literal["exp", "sigmoid", "softplus", "linear"] = "exp",
) -> GradientTransformationExtraArgsReset:
    """Calibrated Partial Resets (CPR)."""

    def init(params: optax.Params, **kwargs):
        flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]
        biases = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}
        biases = {k: v for k, v in biases.items() if k[-1] != out_layer_name}

        return CprOptimState(
            # initial_weights=deepcopy(weights),
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), biases),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), biases),
            remainder=jax.tree.map(lambda _: 0.0, biases),
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
        state: CprOptimState,
        params: optax.Params,
        features: PyTree,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, CprOptimState, optax.OptState | None]:
        del features

        def no_update(updates):
            flat_updates = flax.traverse_util.flatten_dict(updates["params"])
            weight_grads = {k[:-1]: v for k, v in flat_updates.items() if k[-1] == "kernel"}
            weight_grads = {k: v for k, v in weight_grads.items() if k[-1] != out_layer_name}
            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                weight_grads,
                state.utilities,
            )
            all_utils = jnp.concatenate([u.flatten() for u in jax.tree.leaves(_utility)])

            _logs = {
                "std_util": all_utils.std(),
                "nodes_reset": 0.0,  # state.logs['nodes_reset'],
                "low_utility": jnp.sum(all_utils < threshold),
                "mean_utils": all_utils.mean(),
            }

            new_state = state.replace(
                time_step=state.time_step + 1, logs=FrozenDict(_logs), utilities=_utility
            )

            return params, new_state, tx_state

        def _cpr(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CprOptimState, optax.OptState | None]:
            flat_params = flax.traverse_util.flatten_dict(params["params"])  # pyright: ignore[reportIndexIssue]

            weights = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}
            # out_w_mag = utils.get_out_weights_mag(weights)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            flat_updates = flax.traverse_util.flatten_dict(updates["params"])  # pyright: ignore[reportIndexIssue]
            weight_grads = {k[:-1]: v for k, v in flat_updates.items() if k[-1] == "kernel"}
            weight_grads = {k: v for k, v in weight_grads.items() if k[-1] != out_layer_name}

            _utility = jax.tree.map(
                partial(get_updated_utility, decay_rate=decay_rate),
                weight_grads,
                state.utilities,
            )

            # reset weights given mask
            _weights, reset_logs = calibrated_reset_weights(
                key_tree,
                weights,  # Yes out_layer
                _utility,
                weight_init_fn,
                replacement_rate,
                sharpness,
                threshold,
                transform_type,
                out_layer_name,
            )

            # Keep biases unchanged for partial resets.
            _biases = biases

            _logs = {k: 0 for k in state.logs}  # TODO: Slow operation

            # avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)
            # avg_util = jax.tree.map(lambda v: v.mean(), _utility)
            std_util = jax.tree.map(lambda v: v.std(), _utility)

            for layer_key in _utility.keys():
                # _logs["avg_age"] += avg_ages[layer_key]
                # _logs["avg_util"] += avg_util[layer_key]
                _logs["std_util"] += std_util[layer_key]
                _logs["nodes_reset"] += reset_logs[layer_key]["nodes_reset"]
                _logs["low_utility"] += reset_logs[layer_key]["low_utility"]
                _logs["mean_utils"] += reset_logs[layer_key]["mean_utils"]

            _logs["mean_utils"] /= len(reset_logs.keys())  # pyright: ignore[reportArgumentType]

            # We reset running utilities once used for an update
            # Reset to 1 as this should be the mean of the utility distribution given norm
            new_state = state.replace(
                # ages=_ages,
                logs=FrozenDict(_logs),
                rng=new_rng,
                utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), _utility),
                # utilities=_utility, # Try with and without keeping the running average
                time_step=state.time_step + 1,
            )
            new_params = utils.reconstruct_params(
                params,
                utils.split_by_chain(_weights),
                utils.split_by_chain(_biases),
            )

            return (
                new_params,
                new_state,
                tx_state,
            )

        condition = jnp.logical_and(
            state.time_step > 0, (state.time_step % update_frequency == 0)
        )
        return jax.lax.cond(condition, _cpr, no_update, updates)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
