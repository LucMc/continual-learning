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

            weights = {k[-2]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases = {k[-2]: v for k, v in flat_params.items() if k[-1] == "bias"}

            scores = {
                key.split("_act")[0]: get_score(feature_tuple[0])
                for key, feature_tuple in zip(features.keys(), features.values())
            }
            reset_mask = jax.tree.map(get_reset_mask, scores)
            reset_mask["output"] = jnp.zeros_like(reset_mask["output"], dtype=bool)
            _rng, key = random.split(state.rng)
            key_tree = utils.gen_key_tree(key, weights)

            # reset weights given mask
            _weights, reset_logs = utils.reset_weights(
                key_tree,
                reset_mask,
                weights,
                weight_init_fn,  # state.initial_weights
            )

            # Update bias
            _biases = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=b.dtype), b),
                reset_mask,
                biases,
            )

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: Could be improved

            for layer_name in biases.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _biases[layer_name],
                }
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                logs=FrozenDict(_logs), time_step=state.time_step + 1, rng=_rng
            )
            # new_params.update(excluded)

            # Reset optim, i.e. Adamw params
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)
            flat_new_params, _ = jax.tree.flatten(new_params)

            return (
                jax.tree.unflatten(jax.tree.structure(params), flat_new_params),
                new_state,
                _tx_state,
            )

        condition = jnp.logical_and(
            state.time_step > 0, (state.time_step % update_frequency == 0)
        )
        return jax.lax.cond(condition, _redo, no_update, updates)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
