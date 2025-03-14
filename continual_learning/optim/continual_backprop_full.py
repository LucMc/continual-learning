from flax import struct
from flax.core import FrozenDict
from flax.typing import FrozenVariableDict
from jax.random import PRNGKey
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
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
TODO Friday:
It's all about checking this works as expected basically, I know the plasticity looks nice,
but now is the time to do some testing and experiments before working on my own implementation stuff.

 - Check performance plots and run with reasonable hyperparams, eg age at 100

Notes:
Okay, so currently it's a bit of a mess and doesn't really do anything. But the weights are in theory getting the double reset.
Next steps are:
* Update the ages and bias naively [Thursday]
* Compare against half and write tests to ensure everything is as expected [Thursday]
* Jit everything [Friday]

[x] Update the params to see the weight resets impact on performance. [Wednesday]

Plan:
> Reproduce the half implementation but in 2 tree_map stages [x]
> for loop through reset mask to relect the masking
> Test full implementation vs half vs None
> Figure out smarter way, avoiding for loop
> Write tests for full implementation
> Swap to cleanRL PPO w/ custom network optims
> Results for: Sine regression, SlipperyAnt, ContinualDelays
> Done
"""


# Overall optimizer TrainState
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
        # Extract the params we want to optimize
        grads = grads["params"]
        params_for_cbp = self.params["params"]

        # Update with continual backprop
        new_params, new_cbp_state = continual_backprop().update(
            grads,
            self.cbp_state,
            params_for_cbp,
            features=features["intermediates"]["activations"][0],
        )

        utils.check_tree_shapes(params_for_cbp, new_params)

        # Prepare for optax optimizer
        params_for_opt = {"params": new_params}
        grad_for_opt = {"params": grads}

        # Get updates from optimizer
        tx_updates, new_opt_state = self.tx.update(
            grad_for_opt, self.opt_state, params_for_opt
        )
        new_params_with_opt = optax.apply_updates(params_for_opt, tx_updates)

        # Extract the updated parameters from the nested structure
        final_params = new_params_with_opt["params"]

        return self.replace(
            step=self.step + 1,
            params={
                "params": final_params
            },  # Make sure to maintain the 'params' structure
            opt_state=new_opt_state,
            cbp_state=new_cbp_state[0],
            **kwargs,
        )


# CBP Optimizer TrainState (input to overall optim cls above)
@dataclass
class CBPOptimState:
    # Things you shouldn't really mess with
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
    logs: dict = field(default_factory=dict)


def reset_weights(
    reset_mask: Float[Array, "#neurons"],
    layer_w: Float[Array, "#weights"],
    key_tree: PyTree,
    bound: float = 0.01,
):
    layer_names = list(reset_mask.keys())
    logs = {}

    for i in range(len(layer_names) - 1):
        in_layer = layer_names[i]
        out_layer = layer_names[i + 1]

        # Generate random weights for resets
        random_in_weights = random.uniform(
            key_tree[in_layer], layer_w[in_layer].shape, float, -bound, bound
        )
        random_out_weights = random.uniform(
            key_tree[out_layer], layer_w[out_layer].shape, float, -bound, bound
        )

        assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"

        # TODO: Check this is resetting the correct row and columns
        in_reset_mask = reset_mask[in_layer].reshape(1, -1)  # [1, out_size]
        _in_layer_w = jnp.where(in_reset_mask, random_in_weights, layer_w[in_layer])

        out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
        _out_layer_w = jnp.where(
            out_reset_mask,
            random_out_weights,  # Reuse the same random weights or generate new ones if needed
            layer_w[out_layer],
        )
        n_reset = reset_mask[in_layer].sum()

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": n_reset}
    logs[out_layer] =  {"nodes_reset": 0}
    return layer_w, logs


def get_reset_mask(
    layer_w: Float[Array, "#weights"],
    layer_b: Float[Array, "#neurons"],
    utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    features: Float[Array, "#neurons"],
    key: PRNGKey,
    bound: float = 0.01,
    decay_rate: float = 0.9,
    maturity_threshold: float = 100,
    replacement_rate=0.01, 
):
    # Maybe have a dictionary of the different util func transformations and then call the index in a cond
    new_param = layer_w * decay_rate

    updated_utility = (
        (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features) * jnp.sum(layer_w)
    ).flatten()  # Arr[#neurons]

    # get nodes over maturity threshold
    maturity_mask = ages > maturity_threshold  ##
    n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int

    k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)  # bool
    assert k_masked_utility.dtype == bool, "Mask type isn't bool"

    return k_masked_utility


# Give this the ContinualBackpropState params
def continual_backprop(
    util_type: str = "contribution", **kwargs
) -> optax.GradientTransformation:
    def process_params(params: FrozenDict):
        # seperates bias from params
        _params = deepcopy(params)  # ["params"]
        # excluded = {
        #     "out_layer": _params.pop("out_layer")
        # }  # TODO: pass excluded layer names as inputs to cp optim/final by default
        excluded = {} # TODO: Not workin in current full setup

        bias = {}
        weights = {}

        for layer_name in _params.keys():
            bias[layer_name] = _params[layer_name].pop("bias")
            weights[layer_name] = _params[layer_name].pop("kernel")

        return weights, bias, excluded

    def init(params: optax.Params, **kwargs):
        assert util_type in utils.UTIL_TYPES, ValueError(
            f"Invalid util type, select from ({'|'.join(utils.UTIL_TYPES)})"
        )
        weights, bias, _ = process_params(params["params"])

        del params  # Delete params?

        return CBPOptimState(
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree_map(lambda x: jnp.zeros_like(x), bias),
            util_type_id=utils.UTIL_TYPES.index(
                util_type
            ),  # Replace with util function directly?
            accumulated_features_to_replace=0,
            rng=random.PRNGKey(0),
            **kwargs,
        )

    # @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: optax.OptState,
        params: optax.Params | None = None,
        features: PyTree | None = None,
    ) -> tuple[optax.Updates, CBPOptimState]:
        def _continual_backprop(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            weights, bias, excluded = process_params(params)

            # because we need the next layers weight magnitude
            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            reset_mask = jax.tree.map(
                get_reset_mask,
                weights,
                bias,
                state.utilities,
                state.ages,
                features,
                key_tree,
            )

            # reset weights given mask
            _weights, reset_logs = reset_weights(reset_mask, weights, key_tree)

            # reset bias given mask
            _bias = jax.tree.map(lambda b,m: jnp.where(m, jnp.zeros_like(b, dtype=float), b), reset_mask, bias)

            # Update ages
            _ages = jax.tree.map(
                lambda a, m: jnp.where(m, jnp.zeros_like(a), a + 1),
                state.ages,
                reset_mask,
            )

            new_params = {}
            _logs = {k: {} for k in bias.keys()} 

            avg_ages = jax.tree.map(lambda a: a.mean(), state.ages)

            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs[layer_name]["avg_age"] = avg_ages[layer_name]
                _logs[layer_name]["nodes_reset"] = reset_logs[layer_name]["nodes_reset"]


            new_state = state.replace(ages=_ages, rng=new_rng, logs=_logs)
            # new_params.update(excluded) # TODO

            return new_params, (new_state,)  # For now

        return _continual_backprop(updates)  # updates, ContinualBackpropState()

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
"""
