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
from typing import Callable

#
# def process_params(params: PyTree):
#     out_layer_name = "output"
#     # Removed deep copy of params however be careful as changes to `weights` and `bias` are
#
#     excluded = {
#         out_layer_name: params[out_layer_name]
#     }  # TODO: pass excluded layer names as inputs to cp optim/final by default
#     bias = {}
#     weights = {}
#
#     for layer_name in params.keys():
#         # For layer norm etc
#         if type(params[layer_name]) != dict:
#             excluded.update({layer_name: params[layer_name]})
#             continue
#
#         elif not ("kernel" in params[layer_name].keys()):
#             excluded.update({layer_name: params[layer_name]})
#             continue
#
#         bias[layer_name] = params[layer_name]["bias"]
#         weights[layer_name] = params[layer_name]["kernel"]
#
#     # out_w_mag = get_out_weights_mag(weights)
#
#     # Remove output layer
#     # out_w_mag.pop(out_layer_name) # Removes nan for output layer as no out weights
#     weights.pop(out_layer_name)
#     bias.pop(out_layer_name)
#
#     return weights, bias, excluded


# def extract_weights_biases(params):
#     flat_params = flax.traverse_util.flatten_dict(params)
#     weights = {k[0]: v for k, v in flat_params.items() if k[-1] == 'kernel'}
#     biases = {k[0]: v for k, v in flat_params.items() if k[-1] == 'bias'}
#
def get_out_weights_mag(weights): #TODO: Check/test
    w_mags = jax.tree.map(
        lambda layer_w: jnp.abs(layer_w).mean(axis=1), weights
    )  # [2, 10] -> [2,1] mag over w coming out of neuron - LOP does axis 0 of out_layer but should be eqivalent

    keys = list(w_mags.keys())
    return {keys[i]: w_mags[keys[i + 1]] for i in range(len(keys) - 1)}


# def process_params_with_outmag(params: PyTree):
#     # TODO: Make out_w_mag optional so can be used by redo too
#     out_layer_name = "output"
#
#     excluded = {
#         out_layer_name: params[out_layer_name]
#     }  # TODO: pass excluded layer names as inputs to cp optim/final by default
#     bias = {}
#     weights = {}
#
#     for layer_name in params.keys():
#         # For layer norm etc
#         if type(params[layer_name]) != dict:
#             excluded.update({layer_name: params[layer_name]})
#             continue
#
#         elif not ("kernel" in params[layer_name].keys()):
#             excluded.update({layer_name: params[layer_name]})
#             continue
#
#         bias[layer_name] = params[layer_name]["bias"]
#         weights[layer_name] = params[layer_name]["kernel"]
#
#     out_w_mag = get_out_weights_mag(weights)
#
#     # Remove output layer
#     weights.pop(out_layer_name)
#     bias.pop(out_layer_name)
#
#     return weights, bias, out_w_mag, excluded
#


def attach_reset_method(
    *args: tuple[str, optax.GradientTransformation],
) -> optax.GradientTransformationExtraArgs:
    names = [name for name, _ in args]

    if len(names) != len(set(names)):
        raise ValueError(f"Named transformations must have unique names, but got {names}")

    transforms = [(name, optax.with_extra_args_support(t)) for name, t in args]

    def init_fn(params):
        states = {}
        for name, tx in transforms:
            states[name] = tx.init(params)
        return states

    def update_fn(updates, state, params=None, features=None, **extra_args):
        """Updated named chain update from Optax to enable resetting of base optim running stats"""

        new_state = {}
        assert len(transforms) == 2, "chain the optim with the reset method only"
        assert "tx" == args[0][0], "'tx' is the first part of this chain"
        assert "reset_method" == args[1][0], "'reset_method' is the second part of this chain"

        # TX update
        tx = transforms[0][1]
        reset_method = transforms[1][1]

        updates, new_state["tx"] = tx.update(updates, state["tx"], params, **extra_args)
        new_params_with_opt = optax.apply_updates(params, updates)

        # Reset method
        new_params_with_reset, new_state["reset_method"], new_state["tx"] = (
            reset_method.update(
                updates,
                state["reset_method"],
                new_params_with_opt,
                features=features,
                tx_state=new_state["tx"],
            )
        )

        return new_params_with_reset, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

def new_reset_weights(
    key_tree: PRNGKeyArray,
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
    replacement_rate: Float[Array, ""] = None,
):
    all_layer_names = list(layer_w.keys())
    all_mask_names = list(reset_mask.keys()) # Just to check layer names are the same
    logs = {}

    def expand_mask_for_weights(mask_1d, weight_shape, mask_type='incoming'):
        # seperate masks for in/out similar to the official ReDo
        if len(weight_shape) == 2:  # Dense
            if mask_type == 'incoming':
                return mask_1d[None, :]
            else:  # outgoing
                return mask_1d[:, None]
                
        elif len(weight_shape) == 4:  # Conv
            if mask_type == 'incoming':
                return mask_1d[None, None, None, :]
            else:  # outgoing
                return mask_1d[None, None, :, None]
        else:
            raise ValueError(f"Unsupported weight shape: {weight_shape}")

    # assert all_layer_names[-1] == 'output', "Last layer should be Dense with name 'output'"

    for idx, layer_name in enumerate(all_layer_names[:-1]):
        assert layer_name in all_mask_names, f"Layer names should be identical: {layer_name} not in {all_mask_names}"

        in_mask_1d = reset_mask[layer_name]
        assert in_mask_1d.dtype == bool, f"Mask type isn't bool for {layer_name}"
        
        # Reset incoming weights
        in_weight_mask = expand_mask_for_weights(
            in_mask_1d, layer_w[layer_name].shape, mask_type='incoming'
        )
        
        random_weights = weight_init_fn(key_tree[layer_name], layer_w[layer_name].shape)
        layer_w[layer_name] = jnp.where(
            in_weight_mask, random_weights, layer_w[layer_name]
        )
        
        # Reset outgoing weights
        if idx + 1 < len(all_layer_names):
            next_layer = all_layer_names[idx + 1]
            out_weight_shape = layer_w[next_layer].shape
            
            if len(out_weight_shape) == 2:  # Dense layer
                if len(layer_w[layer_name].shape) == 4: # Check if previous layer was conv
                    spatial_size = out_weight_shape[0] // in_mask_1d.size
                    out_mask_1d = jnp.repeat(in_mask_1d, spatial_size)
                else: # Dense -> Dense
                    out_mask_1d = in_mask_1d
                    
            elif len(out_weight_shape) == 4:  # Conv layer
                # Check if input channels match
                expected_in_channels = out_weight_shape[2]
                out_mask_1d = in_mask_1d
            
            # Apply outgoing mask
            out_weight_mask = expand_mask_for_weights(
                out_mask_1d, out_weight_shape, mask_type='outgoing'
            )
            
            layer_w[next_layer] = jnp.where(
                out_weight_mask, jnp.zeros_like(layer_w[next_layer]), layer_w[next_layer]
            )
        
        # Count reset neurons
        n_reset = in_mask_1d.sum()
        logs[layer_name] = {"nodes_reset": n_reset}

    # Handle last layer seperately as there is no outgoing reset
    last_layer = all_layer_names[-1]
    output_mask_1d = reset_mask[last_layer]
    output_weight_shape = layer_w[last_layer].shape

    if len(out_weight_shape) == 2:  # Dense
        assert out_mask_1d.size == out_weight_shape[0], \
            f"Mask size {out_mask_1d.size} != weight dim {out_weight_shape[0]}"

    # Only reset incoming weights for output layer
    output_weight_mask = expand_mask_for_weights(
        output_mask_1d, output_weight_shape, mask_type='incoming'
    )
    
    random_weights = weight_init_fn(key_tree[last_layer], output_weight_shape)
    layer_w[last_layer] = jnp.where(
        output_weight_mask, random_weights, layer_w[last_layer]
    )
    
    logs[last_layer] = {"nodes_reset": output_mask_1d.sum()}

    return layer_w, logs

def reset_weights(
    key_tree: PRNGKeyArray,
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    layer_w: PyTree[Float[Array, "..."]],
    # initial_weights: PyTree[Float[Array, "..."]],
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
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

        # mask_broadcast = jax.lax.select(len(layer_w[w_in_layer])==4, [None,None,None,1], [])
        assert reset_mask[m_in_layer].dtype == bool, "Mask type isn't bool"
        # assert len(reset_mask[m_in_layer].flatten()) == layer_w[w_out_layer].shape[0], (  # ?
        #     f"Reset mask shape incorrect: {len(reset_mask[m_in_layer].flatten())} should be {layer_w[w_out_layer].shape[0]}"
        # )

        in_reset_mask = reset_mask[m_in_layer].reshape(-1)  # [1, out_size]
        random_weights = weight_init_fn(key_tree[w_in_layer], layer_w[w_in_layer].shape)
        _in_layer_w = jnp.where(
            in_reset_mask, random_weights, layer_w[w_in_layer]
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
