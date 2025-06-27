from typing import Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from chex import dataclass
import continual_learning_2.utils.optim as utils

@dataclass
class ShrinkPerturbOptimState:
    count: chex.Array
    rng: chex.PRNGKey


def shrink_perturb(
    param_noise_fn: Callable,
    seed: int = 42,
    shrink: float = 0.8,
    perturb: float = 0.01,
    every_n: int = 1,
) -> optax.GradientTransformationExtraArgs:
    """Shrink and perturb: [Ash & Adams, 2020](https://arxiv.org/abs/1910.08475) """

    def init(params):
        del params
        return ShrinkPerturbOptimState(count=jnp.zeros([], jnp.int32), rng=random.PRNGKey(seed))

    def update(updates, state, params=None, features=None, tx_state=None):
        if params is None:
            raise ValueError("Update requires params argument")

        def no_shrink_perturb(params):
            new_state = state.replace(count=state.count + 1)
            return params, new_state, tx_state

        def apply_shrink_perturb(params):
            weights, bias, excluded = utils.process_params(params["params"])
            new_rng, noise_rng = random.split(state.rng, num=2)
            noise_key_tree = utils.gen_key_tree(noise_rng, weights)

            new_params = jax.tree_map(
                lambda w, b, k: {
                    "kernel": w * shrink + param_noise_fn(k, shape=w.shape) * perturb,
                    "bias": b
                },
                weights,
                bias,
                noise_key_tree
            )
            new_params.update(excluded)

            new_state = state.replace(count=(state.count + 1) % every_n, rng=new_rng)
            return {"params": new_params}, new_state, tx_state

        should_apply = (state.count % every_n == 0) & (state.count > 0)
        
        return jax.lax.cond(
            should_apply,
            apply_shrink_perturb,
            no_shrink_perturb,
            params
        )

    return optax.GradientTransformationExtraArgs(init=init, update=update)
