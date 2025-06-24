import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from jaxtyping import (
    Bool,
)
import flax
from flax.training.train_state import TrainState
import optax

import continual_learning.optim as optim


def reset_weights(
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    initial_weights: PyTree[Float[Array, "..."]],
    replacement_rate: Float[Array, ""] = None,
):
    weight_layer_names = list(layer_w.keys())
    activation_layer_names = list(reset_mask.keys())
    logs = {}

    for i in range(len(weight_layer_names) - 1):
        w_in_layer = weight_layer_names[i]
        w_out_layer = weight_layer_names[i + 1]
        m_in_layer = activation_layer_names[i]
        m_out_layer = activation_layer_names[i + 1]

        assert reset_mask[m_in_layer].dtype == bool, "Mask type isn't bool"
        assert len(reset_mask[m_in_layer].flatten()) == layer_w[w_out_layer].shape[0], (  # ?
            f"Reset mask shape incorrect: {len(reset_mask[m_in_layer].flatten())} should be {layer_w[w_out_layer].shape[0]}"
        )

        in_reset_mask = reset_mask[m_in_layer].reshape(-1)  # [1, out_size]
        _in_layer_w = jnp.where(
            in_reset_mask, initial_weights[w_in_layer], layer_w[w_in_layer]
        )

        _out_layer_w = jnp.where(
            in_reset_mask, jnp.zeros_like(layer_w[w_out_layer]), layer_w[w_out_layer]
        )
        n_reset = reset_mask[m_in_layer].sum()

        layer_w[w_in_layer] = _in_layer_w
        layer_w[w_out_layer] = _out_layer_w

        logs[w_in_layer] = {"nodes_reset": n_reset}

    logs[w_out_layer] = {"nodes_reset": 0}

    return layer_w, logs


# VMAPPED VERSION TODO
def reset_weights_vmap(
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    initial_weights: PyTree[Float[Array, "..."]],
    replacement_rate: Float[Array, ""] = None,
):
    layer_names = list(reset_mask.keys())
    logs = {}

    for i in range(len(layer_names) - 1):
        in_layer = layer_names[i]
        out_layer = layer_names[i + 1]

        assert reset_mask[in_layer].dtype == bool, "Mask type isn't bool"
        assert len(reset_mask[in_layer].flatten()) == layer_w[out_layer].shape[0], (
            f"Reset mask shape incorrect: {len(reset_mask[in_layer].flatten())} should be {layer_w[out_layer].shape[0]}"
        )

        in_reset_mask = reset_mask[in_layer].reshape(-1)  # [1, out_size]
        _in_layer_w = jnp.where(in_reset_mask, initial_weights[in_layer], layer_w[in_layer])

        _out_layer_w = jnp.where(
            in_reset_mask, jnp.zeros_like(layer_w[out_layer]), layer_w[out_layer]
        )
        n_reset = reset_mask[in_layer].sum()

        layer_w[in_layer] = _in_layer_w
        layer_w[out_layer] = _out_layer_w

        logs[in_layer] = {"nodes_reset": n_reset}

    logs[out_layer] = {"nodes_reset": 0}

    return layer_w, logs


def reset_optim_params(tx_state, reset_mask):
    """Reset optimizer params using reset_mask"""

    def composite(tx_state):
        # handle optax chains etc
        new_inner_states = {}
        for name, txs in tx_state.inner_states.items():
            new_inner_states[name] = reset_optim_params(txs.inner_state, reset_mask)

        return tx_state._replace(inner_states=new_inner_states)

    def reset_params(tx_state):
        def map_fn(path, value):
            # resets weights and biases similar to in cbp, but to zero
            if "kernel" in path:
                layer_name = path[0]
                if layer_name in reset_mask:
                    mask = reset_mask[layer_name]
                    mask_expanded = mask[None, :]  # Array: [1, out_features]
                    return jnp.where(mask_expanded, 0.0, value)

            elif "bias" in path:
                layer_name = path[0]  # .key if dict
                if layer_name in reset_mask:
                    mask = reset_mask[layer_name]
                    return jnp.where(mask, 0.0, value)
            return value

        new_state_dict = {}
        if type(tx_state) == dict:
            if "reset_method" in tx_state.keys():
                return {
                    "reset_method": tx_state["reset_method"],
                    "tx": (reset_params(tx_state["tx"][0]),) + tx_state["tx"][1:],
                }
            else:
                raise "Unknown reset method"
                breakpoint()

        if hasattr(tx_state, "mu"):
            new_state_dict["mu"] = {
                "params": flax.traverse_util.path_aware_map(map_fn, tx_state.mu["params"])
            }

        if hasattr(tx_state, "nu"):
            new_state_dict["nu"] = {
                "params": flax.traverse_util.path_aware_map(map_fn, tx_state.nu["params"])
            }

        if isinstance(tx_state, tuple) and len(tx_state)==2: # Make more generic by checking specific type instead
            return (reset_params(tx_state[0]),) + tx_state[1:]

        # copy other attributes
        for attr in tx_state._fields:
            if attr not in ["mu", "nu"] and hasattr(tx_state, attr):
                new_state_dict[attr] = getattr(tx_state, attr)

        return type(tx_state)(**new_state_dict)

    # if hasattr(tx_state, "inner_states"):
    #     return composite(tx_state)
    # else:
    #     if isinstance(tx_state, tuple):
    #         return (reset_params(tx_state[0]),) + tx_state[1:]
    #     else:
    return reset_params(tx_state)


def gen_key_tree(key: PRNGKeyArray, tree: PyTree):
    """
    Creates a PyTree of random keys such that is can be traversed in the tree map and have
    a new key for each leaf.
    """
    leaves, treedef = jax.tree.flatten(tree)
    subkeys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, subkeys)


def get_bottom_k_mask(values, n_to_replace):
    """Get mask for bottom k elements, JIT-compatible with no dynamic indexing."""
    # Get array size
    size = values.shape[-1]

    # Create positions for tie-breaking
    positions = jnp.arange(size)

    # Compute ranks (smaller values → smaller ranks)
    # Double argsort trick to get ranks with tie-breaking
    eps = jnp.finfo(values.dtype).eps * 10.0  # Add small epsilon to avoid equal values
    ranks = jnp.argsort(jnp.argsort(values + positions * eps))

    # Create mask for values with rank < n_to_replace
    mask = ranks < n_to_replace

    return mask


"""
        # copy other attributes
        breakpoint()
        # new_state_dict["reset_method"] = tx_state["reset_method"]
        new_state_dict_tx = {}
        for attr in tx_state["tx"]._fields:
            if attr not in ["mu", "nu"] and hasattr(tx_state["tx"], attr):
                new_state_dict_tx[attr] = getattr(tx_state["tx"], attr)
        
        return type(tx_state)(tx=new_state_dict_tx, res)
    
    if hasattr(tx_state, "inner_states"):
        return composite(tx_state)
    else:
        # for single optimizer states (i.e. just Adam)
        if isinstance(tx_state, tuple):
            # put back into tuple
            return (reset_params(tx_state[0]),) + tx_state[1:]
        else:
            return reset_params(tx_state)
    """
