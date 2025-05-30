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

"""
TODO:
 * Clip ages
 * Reset adam/optim state for reset nodes
 * FIX LOGGING

Count = 15

:: Testing ::
  * See testing list in test_reset.py

:: Experiments ::
  * Implement SGDW as for some reason it's not in https://optax.readthedocs.io/en/latest/api/optimizers.html
  * Make continual task sequence sequential i.e. [1,2,3] instead of random [1,3,2] (SlipperyAnt and sine exp)
  * Test the ppo continual env more/ guage performance of base agent in comparison to lop results
  * Make cont sine regression graphs better and log to wandb, seperate out methods and add sgd

:: Implementation ::
  * Implement accumulated nodes to reset for inexact division by replacement rate
  * Additional logging
  * Link with continual time-delays

:: Errors ::
  * Replacement rate of 0 gives worse loss than adam, should be equal
  * Assert statements throughout, check mask is always false when replacement rate is 0 and n_to_replace is also always zero etc same with maturity_threshold

:: Errors ::
 * Is utility a good measure/ do we outperform random weight reinitialisation?
"""


@dataclass
# @jaxtyped(typechecker=typechecker)
class CBPOptimState:
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
class CBPTrainState(TrainState):
    cbp_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        # params_with_opt = (
        #   params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        # )
        opt_state = tx.init(params)
        cbp_state = continual_backprop().init(params, **kwargs)
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
        params_after_cbp, new_cbp_state = continual_backprop().update(
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


# -------------- CBP Weight reset ---------------
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


# -------------- lowest utility mask ---------------
# @jaxtyped(typechecker=typechecker)
def get_reset_mask(
    updated_utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    maturity_threshold: Int[Array, ""] = 100,
    replacement_rate: Float[Array, ""] = 0.01,
) -> Bool[Array, "#neurons"]:
    maturity_mask = ages > maturity_threshold  # get nodes over maturity threshold Arr[Bool]
    n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int
    k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)  # bool

    return k_masked_utility


# @jaxtyped(typechecker=typechecker)
@jax.jit
def get_out_weights_mag(weights):
    """TODO: Make this not hardcoded"""
    w_mags = jax.tree.map(
        lambda layer_w: jnp.abs(layer_w).mean(axis=1), weights
    )  # [2, 10] -> [2,1] mag over w coming out of neuron - LOP does axis 0 of out_layer but should be eqivalent

    keys = list(w_mags.keys())
    return {keys[i]: w_mags[keys[i + 1]] for i in range(len(keys) - 1)}


def process_params(params: PyTree):
    out_layer_name = "out_layer"
    # Removed deep copy of params however be careful as changes to `weights` and `bias` are

    excluded = {
        out_layer_name: params[out_layer_name]
    }  # TODO: pass excluded layer names as inputs to cp optim/final by default
    bias = {}
    weights = {}

    for layer_name in params.keys():
        # For layer norm etc
        if type(params[layer_name]) != dict:
            excluded.update({layer_name: params[layer_name]})
            continue

        elif not ("kernel" in params[layer_name].keys()):
            excluded.update({layer_name: params[layer_name]})
            continue

        bias[layer_name] = params[layer_name]["bias"]
        weights[layer_name] = params[layer_name]["kernel"]

    out_w_mag = get_out_weights_mag(weights)

    # Remove output layer
    # out_w_mag.pop(out_layer_name) # Removes nan for output layer as no out weights
    weights.pop(out_layer_name)
    bias.pop(out_layer_name)

    return weights, bias, out_w_mag, excluded


# -------------- Main CBP Optimiser body ---------------
# @jaxtyped(typechecker=typechecker)
def continual_backprop(
    util_type: str = "contribution", **kwargs
) -> optax.GradientTransformation:
    def init(params: optax.Params, **kwargs):
        assert util_type in utils.UTIL_TYPES, ValueError(
            f"Invalid util type, select from ({'|'.join(utils.UTIL_TYPES)})"
        )
        weights, bias, _, _ = process_params(params["params"])

        del params  # Delete params?

        return CBPOptimState(
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
        state: CBPOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
    ) -> tuple[optax.Updates, CBPOptimState]:
        def _continual_backprop(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
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

        return _continual_backprop(updates)

    return optax.GradientTransformation(init=init, update=update)


""" old code snippets:
    
    # Why are exactly half the same?? How can I manage multiple utilities with the same value?
    # zeros = jnp.zeros_like(k_masked_utility)
    # twos = jnp.ones_like(k_masked_utility) * 2
    # new_mask = jnp.where(k_masked_utility, twos, zeros) # 2=reset inp,1=reset out,0=Nothing

            # reflect_mask(reset_mask, weights)
            # next_layer_mask = jnp.roll(reset_mask, shift=1)

            # Ensure this optimiser is running
            # cbp_update = reset_params(reset_mask, weights, bias, state.ages, features, key_tree)
            # cbp_update = jax.tree.map(
            #     reset_params,
            #     reset_mask,
            #     weights,
            #     bias,
            #     state.ages,
            #     features,
            #     key_tree,
            # )  # This is the mask for incoming weights, we need to reflect it for outgoing weights too
            # , next_layer_weight_sum) # Instead of applying to each neuron we split it into layers now
            # age_split = jax.vmap(lambda x: x, in_axes=(0,))(cbp_update)

    # utilities=_utilities,
    # mean_feature_act=_mean_feature_act,
    # util_type_id=_util_type_id,
    # accumulated_features_to_replace=_accumulated_features_to_replace,
    # step_size=_step_size,
    # replacement_rate=_replacement_rate,
    # decay_rate=_decay_rate,
    # maturity_threshold=_maturity_threshold,
    # accumulate=_accumulate,

            # TODO: Replace with vmap/treemap
            # def split_data(layer):
            #     # ages = layer.pop("ages")
            #     logs = layer.pop("logs")
            #     return layer, logs

            # for key, value in cbp_update.items():
            #     new_params[key], new_ages[key], new_logs[key] = split_data(value)

            # IDEA: Would generating a rondom number and if it is bellow the threshold then rplace if elegible be better as it introduces more randomness?
            # num_new_features_to_replace = state.replacement_rate * eligable_features_to_replace
            # new_accumulated_features_to_replace += features_to_replace
            # I actually think it's this layers weight sum since this layer weights connect to next
            # See ""calculate feature utility"" because it looks a little different, certainly need features
            # bias_correction = 1 - state.decay_rate ** self.ages
            # bias_correction = jax.tree.map(
            #     lambda a: 1 - state.decay_rate**a, state.ages
            # )

            # layerwise_utility = jax.vmap(utility, in_axes=(0, None))(params) # Expect (layer_n, params) and map over layer_n
            # next_layer_weight_sum = jax.tree.map(lambda layer: layer.sum(), params) # Instead of applying to each neuron we split it into layers now
            # updated_utility = (state.decay_rate * layer) + (1-state.decay_rate) * jnp.abs(features) * next_layer_weight_sum
            # idx_nodes_to_reset = lax.top_k_idx(, )

            k_masked_utility = jax.lax.cond(
                n_to_replace == 0,
                lambda _: jnp.full_like(sorted_utility, fill_value=False, dtype=bool),
                lambda _: jnp.asarray(sorted_utility < sorted_utility[n_to_replace - 1], dtype=bool),
                operand=None
            )

            # resetting outbound connections [128] per node
            # if len(idx_nodes_to_reset) > 0:
            #     _layer_w = layer_w.at[idx_nodes_to_reset].set(
            #         random.uniform(key, layer_w.shape[1], float, -bound, bound)
            #     )
            #     _ages = ages.at[idx_nodes_to_reset].set(0.0)
            #     _layer_b = layer_b.at[idx_nodes_to_reset].set(0.0)
            # else:
            #     _layer_w = layer_w
            #     _layer_b = layer_b
            #     _ages = ages + 1

            # mature_utils = updated_utility[:, maturity_mask]
            #
            # idx_nodes_to_reset = mature_utils[
            #     jnp.argsort(mature_utils)[-n_to_replace:][::-1]
            # ]
        # util_functions = [
        #     lambda x: output_weight_mag, # weight
        #     lambda x: weight_mag * features.abs().mean(dim=1), # contribution
        # lambda x: x, # adaptation
        # lambda x: x, # zero_contribution
        # lambda x: x, # adaptable_contribution
        # lambda x: x, # feature_by_input
        # ]
        # Calculate new_util based on util_type
        # util_function =

        # new_params_after_cbp = new_params_from_tx # THIS MAKES IT EQUAL ADAM THEREFORE NEWPARAMS ARNT THE SAME?

        ## debug -- Add to testing, only with --no-jit
        # if self.cbp_state.replacement_rate == 0:
        # equal_leaves = jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), params_after_tx, params_after_cbp)
        # flat, _ = jax.tree_flatten(equal_leaves)
        # assert jnp.all(jnp.array(flat)), f"Tree has changed: {breakpoint()}"

        # assert jax.tree_util.tree_structure(params_after_tx) == jax.tree_util.tree_structure(params_after_cbp)
        # assert jax.tree.map(lambda p1, p2: jnp.all(p1==p2), new_params_after_cbp, new_params_from_tx), f"old params != new params: \nOld Params['dense_1']:\n{params_for_cbp['dense_1']}\nNew Params['dense_1']:\n{new_params['dense_1']}"
        #
        # elif self.cbp_state.maturity_threshold == 0:
        #     assert jax.tree.map(lambda p1, p2: not jnp.all(p1==p2), new_params_after_cbp, new_params_from_tx), f"old params != new params: \nOld Params['dense_1']:\n{params_for_cbp['dense_1']}\nNew Params['dense_1']:\n{new_params['dense_1']}"

    # out_tree = {
    #     "dense1": w_mags["dense2"],  # [128,]
    #     "dense2": w_mags["dense3"],  # [128,]
    #     "dense3": w_mags["out_layer"],
    # }  # [128,]
    # for k in w_mags.keys()[-1:]:
    # _, unravel_fn = jax.flatten_util.ravel_pytree(w_mags)
    # first_layer = w_mags.pop(w_mags.keys()[0]) # Pop first layer
    # return unravel_fn(jnp.concatenate((flat_ws[1:], jnp.array([jnp.nan])))) # Offset and nan last layer as no weights out of output layer

# @jaxtyped(typechecker=typechecker)
def process_params_old(params: PyTree):
    out_layer_name = "out_layer"

    _params = deepcopy(params)  # ["params"]

    excluded = {
        out_layer_name: params[out_layer_name]
    }  # TODO: pass excluded layer names as inputs to cp optim/final by default
    bias = {}
    weights = {}

    for layer_name in _params.keys():
        # For layer norm etc
        if not ("kernel" in _params[layer_name].keys()):
            excluded.update({layer_name: _params[layer_name]})
            continue

        bias[layer_name] = _params[layer_name].pop("bias")
        weights[layer_name] = _params[layer_name].pop("kernel")

    out_w_mag = get_out_weights_mag(weights)

    # Remove output layer
    # out_w_mag.pop(out_layer_name) # Removes nan for output layer as no out weights
    weights.pop(out_layer_name)
    bias.pop(out_layer_name)

    return weights, bias, out_w_mag, excluded


"""
