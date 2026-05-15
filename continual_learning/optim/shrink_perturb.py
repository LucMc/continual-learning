from typing import Callable

import chex
import flax
import jax
import jax.numpy as jnp
import jax.random as random
from flax import struct
import flax.traverse_util

from continual_learning.types import GradientTransformationExtraArgsReset
import continual_learning.utils.optim as utils


class ShrinkPerturbOptimState(struct.PyTreeNode):
    count: chex.Array
    rng: chex.PRNGKey
    logs: dict


def shrink_perturb(
    param_noise_fn: Callable,
    seed: int = 42,
    shrink: float = 0.8,
    perturb: float = 0.01,
    every_n: int = 1,
) -> GradientTransformationExtraArgsReset:
    """Shrink and perturb: [Ash & Adams, 2020](https://arxiv.org/abs/1910.08475)"""

    def init(params):
        del params
        return ShrinkPerturbOptimState(
            count=jnp.zeros([], jnp.int32), rng=random.PRNGKey(seed), logs={}
        )

    def update(updates, state, params=None, features=None, tx_state=None):
        del updates, features

        if params is None:
            raise ValueError("Update requires params argument")

        def no_shrink_perturb(params):
            new_state = state.replace(count=state.count + 1)
            return params, new_state, tx_state

        def apply_shrink_perturb(params):
            flat_params = flax.traverse_util.flatten_dict(params["params"])

            # Tuple-keyed dicts preserve sub-network structure (q1/q2)
            weights_full = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "kernel"}
            biases_full = {k[:-1]: v for k, v in flat_params.items() if k[-1] == "bias"}

            new_rng, noise_rng = random.split(state.rng, num=2)
            noise_key_tree = utils.gen_key_tree(noise_rng, weights_full)

            new_weights = jax.tree.map(
                lambda w, k: w * shrink + param_noise_fn(k, shape=w.shape) * perturb,
                weights_full,
                noise_key_tree,
            )

            weight_chains = utils.split_by_chain(new_weights)
            bias_chains = utils.split_by_chain(biases_full)

            new_state = state.replace(count=(state.count + 1) % every_n, rng=new_rng)
            new_params = utils.reconstruct_params(params, weight_chains, bias_chains)

            return new_params, new_state, tx_state

        should_apply = (state.count % every_n == 0) & (state.count > 0)

        return jax.lax.cond(should_apply, apply_shrink_perturb, no_shrink_perturb, params)

    return GradientTransformationExtraArgsReset(init=init, update=update)  # pyright: ignore[reportArgumentType]
