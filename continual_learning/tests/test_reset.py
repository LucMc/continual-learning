import jax
import jax.random as random
import jax.numpy as jnp
import pytest
import continual_learning.optim.continual_backprop_full as cbp
import continual_learning.optim.utils as utils
from copy import deepcopy

@pytest.fixture
def setup():
    key = random.PRNGKey(0)
    reset_all_mask = {
        "dense_1": jnp.ones((128,))*2, 
        "dense_2": jnp.ones((128,))*2
    }
    layer_w = {
        "dense_1": jnp.ones((784,128)),
        "dense_2": jnp.ones((128,128))
    }
    key_tree = utils.gen_key_tree(key, layer_w)
    return reset_all_mask, layer_w, key_tree

    
def test_full_reset(setup):
    reset_mask, layer_w, key_tree = setup
    initial_weights = deepcopy(layer_w)
    new_weights = cbp.reset_params(reset_mask, layer_w, key_tree)
    assert utils.are_pytrees_equal(layer_w, layer_w)
    assert not utils.are_pytrees_equal(initial_weights, new_weights), f"Equal weights after update:\nlayer_w: {layer_w}\n\nnew weights: {new_weights}"

    # print("new_weights:\n", new_weights)
    # print("initial_weights:\n", initial_weights)

    """
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
