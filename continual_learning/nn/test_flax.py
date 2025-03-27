import continual_learning.nn.online_norm as on
import continual_learning.nn.online_norm_np as onnp

import jax
import jax.numpy as jnp
import numpy as np
import pytest

@pytest.fixture
def base_setup():
    inps = jnp.ones((64, 128))
    varstream = inps[0]
    eps = 1e-5
    afwd = 0.99

    mstream = jnp.ones_like(varstream)

    return (inps, varstream, eps, afwd, mstream)

@pytest.fixture
def setup_w_fw():
    inps = jnp.ones((64, 128))
    varstream = inps[0]
    eps = 1e-5
    afwd = 0.99
    abkw = 0.99

    mstream = jnp.ones_like(varstream)
    data = on.norm_forward(inps, mstream, varstream, afwd, eps)

    return (inps, varstream, eps, afwd, abkw, mstream, data)


def test_norm_fw(base_setup):
    inps, varstream, eps, afwd, mstream = base_setup

    data = on.norm_forward(inps, mstream, varstream, afwd, eps)
    np_data = onnp.norm_forward(inps, mstream, varstream, afwd, eps)

    outputs, mstream_final, varstream_final, cache = data
    np_outputs, np_mstream_final, np_varstream_final, np_cache = np_data

    assert jnp.all(outputs == np_outputs), "outputs mismatach"
    assert jnp.all(mstream_final == np_mstream_final), "mstream mismatach"
    assert jnp.all(varstream_final == np_varstream_final), "varstream mismatch"
    assert jnp.all(cache[0] == np_cache[0]), "cache[0] mismatch"
    assert jnp.all(cache[1] == np_cache[1]), "cache[1] mismatch"
    return data

def test_norm_bw(setup_w_fw):
    inps, varstream, eps, afwd, abkw, mstream, data = setup_w_fw
    outputs, mstream_final, varstream_final, cache = data
    # np_outputs, np_mstream_final, np_varstream_final, np_cache = onnp.norm_forward(inps, mstream, varstream, afwd, eps)

    grad_out = jnp.zeros_like(inps)
    u = jnp.ones_like(varstream) # means
    v = jnp.ones_like(varstream) # variance

    grad_in, ustream, vstream, x = onnp.norm_backward(grad_out,  u, v, abkw, cache)
    grad_in, ustream, vstream, x = on.norm_backward(grad_out,  u, v, abkw, cache)

