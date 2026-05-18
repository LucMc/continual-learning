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
    out_layer_name: str = "output",
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

            features_by_layer = utils.flatten_activation_tree(features, strip_suffix=True)
            scores = {
                key: get_score(feature_tuple[0])
                for key, feature_tuple in features_by_layer.items()
                if key[-1] != out_layer_name
            }
            reset_mask = jax.tree.map(get_reset_mask, scores)

            # Tuple-keyed dicts preserve sub-network structure (q1/q2)
            weights_full = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases_full = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}
            weight_chains = utils.split_by_chain(weights_full)
            bias_chains = utils.split_by_chain(biases_full)

            _rng = state.rng
            _logs = {k: 0 for k in state.logs}

            for prefix in sorted(weight_chains.keys()):
                chain_w = weight_chains[prefix]
                chain_b = bias_chains[prefix]

                _rng, key = random.split(_rng)
                key_tree = utils.gen_key_tree(key, chain_w)
                chain_reset_mask = utils.split_reset_mask(reset_mask, prefix)

                chain_w, reset_logs = utils.reset_weights(
                    key_tree, chain_reset_mask, chain_w, weight_init_fn,
                )
                chain_reset_mask = utils.add_output_mask(
                    chain_reset_mask, chain_b[out_layer_name], out_layer_name
                )
                chain_b = jax.tree.map(
                    lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=b.dtype), b),
                    chain_reset_mask, chain_b,
                )

                weight_chains[prefix] = chain_w
                bias_chains[prefix] = chain_b
                for layer_name in reset_logs:
                    _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                logs=FrozenDict(_logs), time_step=state.time_step + 1, rng=_rng
            )
            new_params = utils.reconstruct_params(params, weight_chains, bias_chains)
            _tx_state = utils.reset_optim_params(tx_state, reset_mask)

            return new_params, new_state, _tx_state

        condition = jnp.logical_and(
            state.time_step > 0, (state.time_step % update_frequency == 0)
        )
        return jax.lax.cond(condition, _redo, no_update, updates)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
