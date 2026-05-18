import jax

jax.config.update("jax_disable_jit", True)

import jax.numpy as jnp
import optax

from continual_learning.optim.cbp import cbp
from continual_learning.optim.redo import redo
from continual_learning.optim.regrama import regrama


def _twin_dense_params():
    return {
        "params": {
            "q1": {
                "layer_0": {
                    "kernel": jnp.ones((2, 3)),
                    "bias": jnp.ones((3,)),
                },
                "output": {
                    "kernel": jnp.ones((3, 1)),
                    "bias": jnp.ones((1,)),
                },
            },
            "q2": {
                "layer_0": {
                    "kernel": jnp.ones((2, 3)),
                    "bias": jnp.ones((3,)),
                },
                "output": {
                    "kernel": jnp.ones((3, 1)),
                    "bias": jnp.ones((1,)),
                },
            },
        }
    }


def _twin_dense_grads():
    return {
        "params": {
            "q1": {
                "layer_0": {
                    "kernel": jnp.zeros((2, 3)),
                    "bias": jnp.zeros((3,)),
                },
                "output": {
                    "kernel": jnp.zeros((3, 1)),
                    "bias": jnp.zeros((1,)),
                },
            },
            "q2": {
                "layer_0": {
                    "kernel": jnp.ones((2, 3)),
                    "bias": jnp.zeros((3,)),
                },
                "output": {
                    "kernel": jnp.zeros((3, 1)),
                    "bias": jnp.zeros((1,)),
                },
            },
        }
    }


def test_regrama_uses_full_tuple_keys_for_twin_networks():
    params = _twin_dense_params()
    method = regrama(seed=0, update_frequency=1, score_threshold=0.5)
    state = method.init(params).replace(time_step=1)

    new_params, _, _ = method.update(
        _twin_dense_grads(),
        state,
        params,
        features={},
        tx_state=optax.EmptyState(),
    )

    assert not jnp.allclose(
        new_params["params"]["q1"]["layer_0"]["kernel"],
        params["params"]["q1"]["layer_0"]["kernel"],
    )
    assert jnp.allclose(
        new_params["params"]["q2"]["layer_0"]["kernel"],
        params["params"]["q2"]["layer_0"]["kernel"],
    )
    assert jnp.allclose(
        new_params["params"]["q2"]["output"]["kernel"],
        params["params"]["q2"]["output"]["kernel"],
    )


def test_redo_uses_full_tuple_keys_for_twin_networks():
    params = _twin_dense_params()
    features = {
        "q1": {"layer_0_act": (jnp.zeros((4, 3)),)},
        "q2": {"layer_0_act": (jnp.ones((4, 3)),)},
    }

    method = redo(seed=0, update_frequency=1, score_threshold=0.5)
    state = method.init(params).replace(time_step=1)

    new_params, _, _ = method.update(
        _twin_dense_grads(),
        state,
        params,
        features=features,
        tx_state=optax.EmptyState(),
    )

    assert not jnp.allclose(
        new_params["params"]["q1"]["layer_0"]["kernel"],
        params["params"]["q1"]["layer_0"]["kernel"],
    )
    assert jnp.allclose(
        new_params["params"]["q2"]["layer_0"]["kernel"],
        params["params"]["q2"]["layer_0"]["kernel"],
    )
    assert jnp.allclose(
        new_params["params"]["q2"]["output"]["kernel"],
        params["params"]["q2"]["output"]["kernel"],
    )


def test_cbp_keeps_twin_network_utilities_separate():
    params = _twin_dense_params()
    features = {
        "q1": {"layer_0_act": (jnp.zeros((4, 3)),)},
        "q2": {"layer_0_act": (jnp.ones((4, 3)),)},
    }

    method = cbp(seed=0, replacement_rate=0.0, decay_rate=0.0, maturity_threshold=100)
    state = method.init(params)

    _, new_state, _ = method.update(
        _twin_dense_grads(),
        state,
        params,
        features=features,
        tx_state=optax.EmptyState(),
    )

    assert ("q1", "layer_0") in new_state.utilities
    assert ("q2", "layer_0") in new_state.utilities
    assert jnp.allclose(new_state.utilities[("q1", "layer_0")], 0.0)
    assert jnp.all(new_state.utilities[("q2", "layer_0")] > 0.0)
