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

import continual_learning.utils.optim as utils


@dataclass
class RedoOptimState:
    initial_weights: PyTree[Float[Array, "..."]]
    utilities: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, ""]
    accumulated_features_to_replace: int

    rng: PRNGKeyArray  # = random.PRNGKey(0)
    time_step: int = 0
    step_size: float = 0.001
    replacement_rate: float = 0.01
    decay_rate: float = 0.9
    update_frequency: int = 10
    accumulate: bool = False
    logs: FrozenDict = FrozenDict({"nodes_reset": 0})


# -------------- Overall optimizer TrainState ---------------
class RedoTrainState(TrainState):
    redo_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        # params_with_opt = (
        #   params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        # )
        opt_state = tx.init(params)
        redo_state = redo().init(params, **kwargs)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            redo_state=redo_state,
        )

    def apply_gradients(self, *, grads, features, **kwargs):
        """TrainState that gives intermediates to optimizer and overwrites params with updates directly"""

        # Get updates from optimizer
        tx_updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params
        )  # tx first then reset so we don't change reset params based on old grads
        params_after_tx = optax.apply_updates(self.params, tx_updates)

        # Update with continual backprop
        params_after_redo, new_redo_state = redo().update(
            grads["params"],
            self.redo_state,
            params_after_tx["params"],
            features=features["intermediates"]["activations"][0],
        )

        return self.replace(
            step=self.step + 1,
            params=params_after_redo,
            opt_state=new_opt_state,
            redo_state=new_redo_state[0],
            **kwargs,
        )


# -------------- Redo Weight reset ---------------
def reset_weights(
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    key_tree: PyTree[PRNGKeyArray],
    initial_weights: PyTree[Float[Array, "..."]],
    replacement_rate: Float[Array, ""] = None,
):
    layer_names = list(reset_mask.keys())
    logs = {}

    for i in range(len(layer_names) - 1):
        in_layer = layer_names[i]
        out_layer = layer_names[i + 1]

        assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"

        in_reset_mask = reset_mask[in_layer].reshape(1, -1)  # [1, out_size]
        _in_layer_w = jnp.where(in_reset_mask, initial_weights[in_layer], layer_w[in_layer])

        out_reset_mask = reset_mask[in_layer].reshape(-1, 1)  # [in_size, 1]
        _out_layer_w = jnp.where(
            out_reset_mask, jnp.zeros_like(layer_w[out_layer]), layer_w[out_layer]
        )
        n_reset = reset_mask[in_layer].sum()

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": n_reset}

    logs[out_layer] = {"nodes_reset": 0}

    return layer_w, logs


def get_score(  # averages over a batch
    # out_w_mag: Float[Array, "#weights"],
    # utility: Float[Array, "#neurons"],
    features: Float[Array, "#batch #neurons"],
    # decay_rate: Float[Array, ""] = 0.9,
) -> Float[Array, "#neurons"]:
    # Remove batch dim from some inputs just in case
    mean_act_per_neuron = jnp.mean(jnp.abs(features), axis=0)  # Arr[#neurons]
    score = mean_act_per_neuron / (
        jnp.mean(mean_act_per_neuron) + 1e-8
    )  # Arr[#neurons] / Scalar
    return score


# -------------- lowest utility mask ---------------
def get_reset_mask(
    scores: Float[Array, "#neurons"],
    score_threshold: Int[Array, ""] = 0.2,
) -> Bool[Array, "#neurons"]:
    score_mask = scores <= score_threshold  # get nodes over maturity threshold Arr[Bool]
    return score_mask


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

    # out_w_mag = get_out_weights_mag(weights)

    # Remove output layer
    # out_w_mag.pop(out_layer_name) # Removes nan for output layer as no out weights
    weights.pop(out_layer_name)
    bias.pop(out_layer_name)

    return weights, bias, excluded


# -------------- Main Redo Optimiser body ---------------
def redo(**kwargs) -> optax.GradientTransformation:
    def init(params: optax.Params, **kwargs):
        weights, bias, _ = process_params(params["params"])

        del params  # Delete params?

        return RedoOptimState(
            initial_weights=weights,
            utilities=jax.tree.map(lambda layer: jnp.ones_like(layer), bias),
            mean_feature_act=jnp.zeros(0),
            accumulated_features_to_replace=0,
            # rng=random.PRNGKey(0), # Seed passed in through kwargs?
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: RedoOptimState,
        params: optax.Params | None = None,
        features: Array | None = None,
    ) -> tuple[optax.Updates, RedoOptimState]:
        def no_update(updates):
            new_state = state.replace(time_step=state.time_step + 1)
            return {"params": params}, (new_state,)

        def _redo(updates: optax.Updates,) -> Tuple[optax.Updates, RedoOptimState]:  # fmt: skip
            weights, bias, excluded = process_params(params)

            new_rng, util_key = random.split(state.rng)
            key_tree = utils.gen_key_tree(util_key, weights)

            # vmap score calculation over batch
            batched_score_calculation = jax.vmap(
                get_score,
                in_axes=0,
            )
            scores_batch = jax.tree.map(
                batched_score_calculation, features
            )  # Get each batch scores
            scores = jax.tree.map(
                lambda x: x.mean(axis=0), scores_batch
            )  # Avg scores into single batch: Arr[#neurons]

            reset_mask = jax.tree.map(get_reset_mask, scores)

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

            new_params = {}
            _logs = {k: 0 for k in state.logs}  # TODO: kinda sucks for adding logs

            for layer_name in bias.keys():
                new_params[layer_name] = {
                    "kernel": _weights[layer_name],
                    "bias": _bias[layer_name],
                }
                _logs["nodes_reset"] += reset_logs[layer_name]["nodes_reset"]

            new_state = state.replace(
                rng=new_rng, logs=FrozenDict(_logs), time_step=state.time_step + 1
            )
            new_params.update(excluded)  # TODO

            return {"params": new_params}, (new_state,)

        return jax.lax.cond(
            state.time_step % state.update_frequency == 0, _redo, no_update, updates
        )

    return optax.GradientTransformation(init=init, update=update)
