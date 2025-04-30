import jax
import jax.tree_util as tu
import jax.random as random
import jax.numpy as jnp

import flax.linen as nn
from functools import partial


import pytest
import continual_learning.optim.continual_backprop as cbp
import continual_learning.optim.utils as utils
from continual_learning.nn import TestNet
from copy import deepcopy
import optax


def pt(pytree, indent=0):
    prefix = "  " * indent

    if isinstance(pytree, dict):
        for k, v in pytree.items():
            print(f"{prefix}  {k}:")
            pt(v, indent + 2)

    elif isinstance(pytree, (list, tuple)):
        container = "list" if isinstance(pytree, list) else "tuple"
        print(f"{prefix}{container}:")
        for i, v in enumerate(pytree):
            print(f"{prefix}  {i}:")
            pt(v, indent + 2)

    elif hasattr(pytree, "shape") and hasattr(pytree, "dtype"):
        # Handle JAX arrays/DeviceArrays
        shape_str = "×".join(str(dim) for dim in pytree.shape)
        print(f"{prefix}Array({shape_str}, {pytree.dtype})")

    else:
        # Handle primitive values
        print(f"{prefix}{type(pytree).__name__}: {pytree}")



@pytest.fixture
def full_setup():
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    net = TestNet()
    dummy_input = jnp.zeros((1, 1))
    params = net.init(init_key, dummy_input)
    cbp_kwargs = {"replacement_rate": 0.5}  # replace exacly one

    cbp_adam_tx = optax.adam(learning_rate=1e-3)
    cbp_state = cbp.CBPTrainState.create(
        apply_fn=net.predict, params=params, tx=cbp_adam_tx, **cbp_kwargs
    )
    return cbp_state, net


def test_process_params(full_setup):
    """
    To Test:
      Mask:
       - Replacement rate [x]
       - Ages [x]
       - features [x]
       - utilities (getting correct bottom n)
       - out_w_mag values are correct
      Reset:
       - Bias (corresponding to mask)
       - Weights (corresponding to mask, i.e. col/row correct)
    """

    cbp_outer_state, net = full_setup
    cbp_params = cbp_outer_state.params["params"]
    cbp_state = cbp_outer_state.cbp_state

    inputs = jnp.ones((1, 1))
    predictions, features = net.apply(
        cbp_outer_state.params, inputs, mutable="intermediates"
    )
    features = features["intermediates"]["activations"][0]

    weights, bias, out_w_mag, excluded = cbp.process_params(cbp_params)

    print(">> ages:\n")
    pt(cbp_state.ages)
    print("\n")

    reset_mask = jax.tree.map(
        cbp.get_reset_mask,
        out_w_mag,
        cbp_state.utilities,
        cbp_state.ages,
        features,
    )

    # Test ages and mask update correctly
    _ages = jax.tree.map(
        lambda a, m: jnp.where(
            m, jnp.zeros_like(a), a + 777
        ),  # Clip to stop huge ages unnessesarily
        cbp_state.ages,
        reset_mask,
    )

    def get_mask(
        replacement_rate=1, ages=cbp_state.maturity_threshold, maturity_threshold=-1
    ):
        reset_mask = jax.tree.map(
            partial(
                cbp.get_reset_mask,
                decay_rate=cbp_state.decay_rate,
                maturity_threshold=maturity_threshold,
                replacement_rate=replacement_rate,
            ),
            out_w_mag,
            cbp_state.utilities,
            ages,
            features,
        )
        return reset_mask  # tu.tree_all(

    ## Test mask based on replacement rate
    rr_mask = partial(get_mask, ages=_ages, maturity_threshold=1)  # All mature ages
    rm_0, rm_025, rm_050, rm_075, rm_1 = (
        rr_mask(x) for x in [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    assert tu.tree_all(tu.tree_map(lambda m: jnp.all(m == False), rm_0)), (
        f"Replace none mask failed, mask: {reset_mask}"
    )
    assert tu.tree_all(tu.tree_map(lambda m: jnp.all(m == True), rm_1)), (
        f"Replace all mask failed, mask: {reset_mask}"
    )
    assert tu.tree_all(tu.tree_map(lambda m: jnp.mean(m) == 0.5, rm_050)), (
        f"Replace half mask failed, mask: {reset_mask}"
    )
    assert tu.tree_all(tu.tree_map(lambda m: jnp.mean(m) == 0.25, rm_025)), (
        f"Replace quater mask failed, mask: {reset_mask}"
    )
    assert tu.tree_all(tu.tree_map(lambda m: jnp.mean(m) == 0.75, rm_075)), (
        f"Replace quater mask failed, mask: {reset_mask}"
    )

    def set_age(value, reset_mask):
        ages = jax.tree.map(
            lambda a, m: jnp.where(m, jnp.zeros_like(a), value),
            cbp_state.ages,
            reset_mask,
        )
        return ages

    ## Test mask based on ages
    def test_ages():
        ages_mask = partial(get_mask, maturity_threshold=10)  # Replace all
        a_0, a_m = (set_age(x, rm_0) for x in [0, 11])  # None staged for resetting
        am_0, am_m = (ages_mask(ages=ages, replacement_rate=1) for ages in [a_0, a_m])
        ah_0, ah_m = (ages_mask(ages=ages, replacement_rate=0.5) for ages in [a_0, a_m])

        assert tu.tree_all(tu.tree_map(lambda a: jnp.all(a == 0), a_0)), (
            f"Not all ages equal 0: {a_0}"
        )
        assert tu.tree_all(tu.tree_map(lambda a: jnp.all(a == 11), a_m)), (
            f"Not all ages equal 11: {a_m}"
        )
        assert tu.tree_all(tu.tree_map(lambda m: jnp.all(m == False), am_0)), (
            f"Mask not false when ages 0: {am_0}"
        )
        assert tu.tree_all(tu.tree_map(lambda m: jnp.all(m == True), am_m)), (
            f"Mask not ture when ages over maturity threshold: {am_m}"
        )
        assert tu.tree_all(tu.tree_map(lambda m: jnp.mean(m) == 0, ah_0)), (
            f"Half mask not false when ages 0: {ah_0}"
        )
        assert tu.tree_all(tu.tree_map(lambda m: jnp.mean(m) == 0.5, ah_m)), (
            f"Half mask not ture when ages over maturity threshold: {ah_m}"
        )

    ## Test features
    def test_features():
        # feature: relu( inp x W + b )
        # arch: 1->4x4x4x1
        ones_weights = jax.tree.map(lambda x: jnp.ones_like(x), weights)
        ones_bias = jax.tree.map(lambda x: jnp.ones_like(x), bias)
        ones_params = jax.tree.map(lambda x: jnp.ones_like(x), cbp_outer_state.params)
        pred, feats = net.apply(ones_params, jnp.ones((1, 1)), mutable="intermediates")
        # 1x1+1 = 2
        assert jnp.all(feats["intermediates"]["activations"][0]["dense1"] == 2.0), "First layer activations check"
        # 4x2+1
        assert jnp.all(feats["intermediates"]["activations"][0]["dense2"] == 9.0), "Second layer activations check"
        # 4x9+1
        assert jnp.all(feats["intermediates"]["activations"][0]["dense3"] == 37.0), "Third layer activations check"
        # 4x36+1=149
        assert pred == 149.0, "pred check"

    def test_reset_weights():
        pass # TODO by you, Gemini




    # test_rr not in function
    test_ages()
    test_features()

    """
    # decay_rate = cbp_state.decay_rate
    # utility = cbp_state.utilities["dense1"]
    # features = features["dense1"]
    # ages = _ages["dense1"]
    # out_w_mag = out_w_mag["dense1"]
    # maturity_threshold = cbp_state.maturity_threshold
    # replacement_rate = 1# cbp_state.replacement_rate
    #
    # updated_utility = (
    #     (decay_rate * utility) + (1 - decay_rate) * jnp.abs(features) * out_w_mag
    # ).flatten()  # Arr[#neurons]
    # maturity_mask = ages > maturity_threshold  # # get nodes over maturity threshold Arr[Bool]
    # n_to_replace = jnp.round(jnp.sum(maturity_mask) * replacement_rate)  # int
    # k_masked_utility = utils.get_bottom_k_mask(updated_utility, n_to_replace)
    # assert sum(k_masked_utility) == 4, f"Replace all mask failed (should be all True): {k_masked_utility}"
    # print("k_masked_utility:\n", k_masked_utility)

    # @pytest.fixture
    # def small_setup():
    #     key = random.PRNGKey(0)
    #     reset_all_mask = {"dense_1": jnp.ones((128,)) * 2, "dense_2": jnp.ones((128,)) * 2}
    #     layer_w = {"dense_1": jnp.ones((784, 128)), "dense_2": jnp.ones((128, 128))}
    #     key_tree = utils.gen_key_tree(key, layer_w)
    #     return reset_all_mask, layer_w, key_tree
def test_full_reset(setup):
    reset_mask, layer_w, key_tree = setup
    initial_weights = deepcopy(layer_w)
    new_weights = cbp.reset_params(reset_mask, layer_w, key_tree)
    assert utils.are_pytrees_equal(layer_w, layer_w)
    assert not utils.are_pytrees_equal(initial_weights, new_weights), (
        f"Equal weights after update:\nlayer_w: {layer_w}\n\nnew weights: {new_weights}"
    )

    # print("new_weights:\n", new_weights)
    # print("initial_weights:\n", initial_weights)
    layer_names = list(reset_mask.keys())
    out_dict = deepcopy(layer_w) # just for shape for now, probs more efficient way
    bound = 0.01

    for i in range(len(layer_names)-1):
        in_layer = layer_names[i]
        out_layer = layer_names[i+1]
        
        # Generate random weights for resets
        random_in_weights = random.uniform(
            key_tree[in_layer], layer_w[in_layer].shape, float, -bound, bound
        )
        random_out_weights = random.uniform(
            key_tree[out_layer], layer_w[out_layer].shape, float, -bound, bound
        )
        
        # the whole mask == 2 thing is now useless, do same with bias too
        
        in_reset_mask = (reset_mask[in_layer] == 2).reshape(1, -1)  # [1, out_size]
        _in_layer_w = jnp.where(
            in_reset_mask,
            random_in_weights, 
            layer_w[in_layer]
        )
        
        out_reset_mask = (reset_mask[in_layer] == 2).reshape(-1, 1)  # [in_size, 1]
        _out_layer_w = jnp.where(
            out_reset_mask,
            random_out_weights,  # Reuse the same random weights or generate new ones if needed
            layer_w[out_layer]
        )
        
        breakpoint()
        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w
        """
