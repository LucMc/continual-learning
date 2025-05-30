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
from beartype import beartype as typechecker
from flax.training.train_state import TrainState
from typing import Tuple
from chex import dataclass
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from dataclasses import field

import continual_learning.optim.utils as utils
from continual_learning.optim.continual_backprop import get_out_weights_mag, process_params, get_reset_mask

@dataclass
# @jaxtyped(typechecker=typechecker)
class CCBPOptimState:
    initial_weights: PyTree[Float[Array, "..."]]
    utilities: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, ""]
    ages: Array
    util_type_id: int
    accumulated_features_to_replace: int

    rng: PRNGKeyArray  # = random.PRNGKey(0)
    step_size: float = 0.001
    replacement_rate: float = 0.01
    decay_rate: float = 0.9
    maturity_threshold: int = 100
    accumulate: bool = False
    logs: FrozenDict = FrozenDict({"avg_age": 0, "nodes_reset": 0})



# -------------- Overall optimizer TrainState ---------------
class CCBPTrainState(TrainState):
    cbp_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        # params_with_opt = (
        #   params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        # )
        opt_state = tx.init(params)
        cbp_state = continuous_continual_backprop().init(params, **kwargs)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            cbp_state=cbp_state,
        )

    def apply_gradients(self, *, grads, features, **kwargs):
        """TrainState that gives intermediates to optimizer and overwrites params with updates directly"""

        # Get updates from optimizer
        tx_updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params
        )  # tx first then reset so we don't change reset params based on old grads
        params_after_tx = optax.apply_updates(self.params, tx_updates)

        # Update with continual backprop
        params_after_cbp, new_cbp_state = continuous_continual_backprop().update(
            grads["params"],
            self.cbp_state,
            params_after_tx["params"],
            features=features["intermediates"]["activations"][0],
        )

        # utils.check_tree_shapes(params_after_tx, params_after_cbp)
        # utils.check_tree_shapes(self.params, params_after_cbp)

        return self.replace(
            step=self.step + 1,
            params=params_after_cbp,
            opt_state=new_opt_state,
            cbp_state=new_cbp_state[0],
            **kwargs,
        )


# -------------- CCBP Weight reset ---------------
# @jaxtyped(typechecker=typechecker)
def reset_weights(
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    key_tree: PyTree[PRNGKeyArray],
    initial_weights: PyTree[Float[Array, "..."]],
    bound: Float[Array, ""] = 0.01,
):
    layer_names = list(reset_mask.keys())
    logs = {}

    for i in range(len(layer_names) - 1):
        in_layer = layer_names[i]
        out_layer = layer_names[i + 1]

        # Generate random weights for resets
        # random_in_weights = random.uniform(
        #     key_tree[in_layer], layer_w[in_layer].shape, float, -bound, bound
        # )
        zero_out_weights = jnp.zeros(layer_w[out_layer].shape, float)

        assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"

        # TODO: Check this is resetting the correct row and columns
        in_reset_mask = reset_mask[in_layer].reshape(1, -1)  # [1, out_size]
        # _in_layer_w = jnp.where(in_reset_mask, random_in_weights, layer_w[in_layer])
        _in_layer_w = jnp.where(in_reset_mask, initial_weights[in_layer], layer_w[in_layer])

        out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
        _out_layer_w = jnp.where(out_reset_mask, zero_out_weights, layer_w[out_layer])
        n_reset = reset_mask[in_layer].sum()

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": n_reset}
    logs[out_layer] = {"nodes_reset": 0}
    return layer_w, logs


# @jaxtyped(typechecker=typechecker)
def get_updated_utility(  # Add batch dim
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],
    decay_rate: Float[Array, ""] = 0.9,
):
    # Remove batch dim from some inputs just in case
    updated_utility = (
        (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features) * out_w_mag
    ).flatten()  # Arr[#neurons]
    return updated_utility


# -------------- mature only mask ---------------
# @jaxtyped(typechecker=typechecker)
def get_reset_mask(
    updated_utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
) -> Bool[Array, "#neurons"]:
    maturity_mask = ages > maturity_threshold  # get nodes over maturity threshold Arr[Bool]
    # n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int
    # k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)  # bool
    # return k_masked_utility
    return maturity_mask


# -------------- Main CCBP Optimiser body ---------------
# @jaxtyped(typechecker=typechecker)
def continuous_continual_backprop(
    util_type: str = "contribution", **kwargs
) -> optax.GradientTransformation:
    def init(params: optax.Params, **kwargs):
        assert util_type in utils.UTIL_TYPES, ValueError(
            f"Invalid util type, select from ({'|'.join(utils.UTIL_TYPES)})"
        )
        weights, bias, _, _ = process_params(params["params"])

        del params  # Delete params?

        return CCBPOptimState(
            initial_weights=weights,
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), bias),
            util_type_id=utils.UTIL_TYPES.index(
                util_type
            ),  # Replace with util function directly?
            accumulated_features_to_replace=0,
            # rng=random.PRNGKey(0), # Seed passed in through kwargs?
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: CCBPOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
    ) -> tuple[optax.Updates, CCBPOptimState]:
        def _continuous_continual_backprop(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CCBPOptimState]:
            weights, bias, out_w_mag, excluded = process_params(params)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            # vmap utility calculation over batch
            batched_util_calculation = jax.vmap(
                partial(get_updated_utility, decay_rate=state.decay_rate),
                in_axes=(None, None, 0),
            )
            _utility_batch = jax.tree.map(
                batched_util_calculation, out_w_mag, state.utilities, features
            )
            _utility = jax.tree.map(lambda x: x.mean(axis=0), _utility_batch)

            reset_mask = jax.tree.map(
                partial(
                    get_reset_mask,
                    maturity_threshold=state.maturity_threshold,
                    replacement_rate=state.replacement_rate,
                ),
                _utility,
                state.ages,
            )

            # reset weights given mask
            _weights, reset_logs = reset_weights(
                reset_mask, weights, key_tree, state.initial_weights
            )

            # reset bias given mask
            _bias = jax.tree.map(
                lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
                reset_mask,
                bias,
            )

            # Update ages
            _ages = jax.tree.map(
                lambda a, m: jnp.where(
                    m, jnp.zeros_like(a), a + 1
                ),  # Clip to stop huge ages unnessesarily
                state.ages,
                reset_mask,
            )

            new_params = {}
            _logs = { k: 0 for k in state.logs} # TODO: kinda sucks for adding logs

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)

            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs["avg_age"] += avg_ages[layer_name]
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(ages=_ages, rng=new_rng, logs=FrozenDict(_logs), utilities=_utility)
            new_params.update(excluded)  # TODO

            return {"params": new_params}, (new_state,)

        return _continuous_continual_backprop(updates)

    return optax.GradientTransformation(init=init, update=update)

