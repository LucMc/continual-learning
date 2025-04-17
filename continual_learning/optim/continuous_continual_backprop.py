from flax import struct
from flax.core import FrozenDict
from flax.typing import FrozenVariableDict
from jax.random import PRNGKey
from jaxtyping import Array, Float, Bool, PRNGKeyArray, PyTree
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
  * Make out_w_mag general
  * Additional logging
  * Implement online norm: https://github.com/Cerebras/online-normalization/blob/master/online-norm/numpy_on/online_norm_1d.py
  * Implement layer norm
  * Implement Dtanh (meta recent paper)
  * Link with continual time-delays
  * Implement my own PPO to show it's not some random trick in SBX

:: Errors ::
  * Replacement rate of 0 gives worse loss than adam, should be equal
  * Assert statements throughout, check mask is always false when replacement rate is 0 and n_to_replace is also always zero etc same with maturity_threshold

"""


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
        ) # tx first then reset so we don't change reset params based on old grads
        params_after_tx = optax.apply_updates(self.params, tx_updates)

        # Update with continual backprop
        params_after_cbp, new_cbp_state = continual_backprop().update(
            grads["params"],
            self.cbp_state,
            params_after_tx["params"],
            features=features["intermediates"]["activations"][0],
        )
        utils.check_tree_shapes(params_after_tx, params_after_cbp)
        utils.check_tree_shapes(self.params, params_after_cbp)

        return self.replace(
            step=self.step + 1,
            params=params_after_cbp,
            opt_state=new_opt_state,
            cbp_state=new_cbp_state[0],
            **kwargs,
        )


# -------------- CBP Weight reset ---------------
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
        zero_out_weights = jnp.zeros(layer_w[out_layer].shape, float)

        assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"

        # TODO: Check this is resetting the correct row and columns
        in_reset_mask = reset_mask[in_layer].reshape(1, -1)  # [1, out_size]
        _in_layer_w = jnp.where(in_reset_mask, random_in_weights, layer_w[in_layer])

        out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
        _out_layer_w = jnp.where(
            out_reset_mask,
            zero_out_weights,  # Reuse the same random weights or generate new ones if needed
            layer_w[out_layer],
        )
        n_reset = reset_mask[in_layer].sum()

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": n_reset}
    logs[out_layer] = {"nodes_reset": 0}
    return layer_w, logs


# -------------- lowest utility mask ---------------
def get_reset_mask(
    out_w_mag: Float[Array, "#weights"],
    utility: Float[Array, "#neurons"],
    ages: Float[Array, "#neurons"],
    features: Float[Array, "#neurons"],
    decay_rate: float = 0.9,
    maturity_threshold: float = 100,
    replacement_rate=0.01,
) -> Bool[Array, "#neurons"]:
    # TODO: Remove batch dim from some inputs just in case
    updated_utility = (
        (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features) * out_w_mag
    ).flatten()  # Arr[#neurons]

    maturity_mask = (
        ages > maturity_threshold
    )  # get nodes over maturity threshold Arr[Bool]
    n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int
    k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)  # bool

    return k_masked_utility


def get_out_weights_mag(weights):
    w_mags = jax.tree.map(
        lambda layer_w: jnp.abs(layer_w).mean(axis=1), weights
    )  # [2, 10] -> [2,1] mag over w coming out of neuron - LOP does axis 0 of ou_layer but should be eqivalent
    out_tree = {
        "dense1": w_mags["dense2"],  # [128,]
        "dense2": w_mags["dense3"],  # [128,]
        "dense3": w_mags["out_layer"],
    }  # [128,]

    return out_tree


def process_params(params: FrozenDict):
    out_layer_name = "out_layer"

    _params = deepcopy(params)  # ["params"]

    excluded = {
        out_layer_name: params[out_layer_name]
    }  # TODO: pass excluded layer names as inputs to cp optim/final by default

    bias = {}
    weights = {}

    for layer_name in _params.keys():
        bias[layer_name] = _params[layer_name].pop("bias")
        weights[layer_name] = _params[layer_name].pop("kernel")

    out_w_mag = get_out_weights_mag(weights)

    # Remove output layer
    # out_w_mag.pop(out_layer_name) # Removes nan for output layer as no out weights
    weights.pop(out_layer_name)
    bias.pop(out_layer_name)

    return weights, bias, out_w_mag, excluded


# -------------- Main CBP Optimiser body ---------------
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
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree.map(lambda x: jnp.zeros_like(x), bias),
            util_type_id=utils.UTIL_TYPES.index(
                util_type
            ),
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
            weights, bias, out_w_mag, excluded = process_params(params)

            # because we need the next layers weight magnitude
            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            reset_mask = jax.tree.map(
                partial(
                    get_reset_mask,
                    decay_rate=state.decay_rate,
                    maturity_threshold=state.maturity_threshold,
                    replacement_rate=state.replacement_rate
                ),
                out_w_mag,
                state.utilities,
                state.ages,
                features,
            )

            # reset weights given mask
            _weights, reset_logs = reset_weights(reset_mask, weights, key_tree)

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
            new_params.update(excluded)  # TODO

            return {"params": new_params}, (new_state,)  # For now

        return _continual_backprop(updates)  # updates, ContinualBackpropState()

    return optax.GradientTransformation(init=init, update=update)


