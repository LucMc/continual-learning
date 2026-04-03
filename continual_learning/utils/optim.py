import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from jaxtyping import (
    Bool,
)
import optax

from typing import Callable
from copy import deepcopy

import flax.traverse_util
from flax.core import FrozenDict, freeze

from continual_learning.types import GradientTransformationExtraArgsReset


def split_by_chain(full_dict):
    """Split a dict with tuple keys into per-chain dicts with string keys.

    Example: {('q1', 'conv'): v1, ('q1', 'dense'): v2, ('q2', 'conv'): v3}
    → {('q1',): {'conv': v1, 'dense': v2}, ('q2',): {'conv': v3}}
    """
    chains = {}
    for key, value in full_dict.items():
        prefix = key[:-1]
        layer = key[-1]
        if prefix not in chains:
            chains[prefix] = {}
        chains[prefix][layer] = value
    return chains


def reconstruct_params(params, weight_chains, bias_chains):
    """Rebuild full param tree from per-chain weight/bias dicts."""
    flat_full = flax.traverse_util.flatten_dict(params)
    for prefix, chain_w in weight_chains.items():
        for layer, kernel in chain_w.items():
            flat_full[("params",) + prefix + (layer, "kernel")] = kernel
    for prefix, chain_b in bias_chains.items():
        for layer, bias in chain_b.items():
            flat_full[("params",) + prefix + (layer, "bias")] = bias
    sorted_keys = sorted(flat_full.keys())
    return jax.tree.unflatten(jax.tree.structure(params), [flat_full[k] for k in sorted_keys])


def get_out_weights_mag(weights):
    def calculate_mag(curr_layer_w, next_layer_w, is_conv_to_dense=False):
        if is_conv_to_dense:  # To handle flattening of spacial dims
            num_channels = curr_layer_w.shape[-1]  # Last dim
            flattened_size = next_layer_w.shape[0]
            spatial_positions = flattened_size // num_channels

            reshaped_weights = next_layer_w.reshape((spatial_positions, num_channels, -1))

            return jnp.abs(reshaped_weights).mean(axis=(0, 2))

        elif len(next_layer_w.shape) == 4:  # Conv->Conv
            return jnp.abs(next_layer_w).mean(axis=(0, 1, 3))

        else:  # Dense->Dense
            return jnp.abs(next_layer_w).mean(axis=1)

    keys = list(weights.keys())
    w_mags = {}

    for i in range(len(keys) - 1):
        curr_key = keys[i]
        next_key = keys[i + 1]
        curr_weights = weights[curr_key]
        next_weights = weights[next_key]

        # Check if this is a conv->dense
        is_conv_to_dense = len(curr_weights.shape) == 4 and len(next_weights.shape) == 2

        w_mags[curr_key] = calculate_mag(curr_weights, next_weights, is_conv_to_dense)

    return w_mags


def attach_reset_method(
    tx: optax.GradientTransformation,
    reset_method: GradientTransformationExtraArgsReset,
) -> optax.GradientTransformationExtraArgs:
    tx = optax.with_extra_args_support(tx)

    def init_fn(params):
        return {
            "tx": tx.init(params),
            "reset_method": reset_method.init(params),
        }

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
        **extra_args,
    ):
        """Updated named chain update from Optax to enable resetting of base optim running stats"""

        assert params is not None
        features = extra_args.pop("features")

        new_state = {}
        tx_state, reset_method_state = state["tx"], state["reset_method"]  # pyright: ignore[reportIndexIssue]

        raw_grads = updates
        updates, new_state["tx"] = tx.update(updates, tx_state, params, **extra_args)
        new_params_with_opt = optax.apply_updates(params, updates)

        # Reset method
        new_params_with_reset, new_state["reset_method"], new_state["tx"] = (
            reset_method.update(
                raw_grads,  # Use original grads before optim transform
                reset_method_state,
                new_params_with_opt,
                features=features,
                tx_state=new_state["tx"],
            )
        )

        return new_params_with_reset, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def expand_mask_for_weights(mask_1d, weight_shape, mask_type="incoming"):
    # seperate masks for in/out similar to the official ReDo
    if len(weight_shape) == 2:  # Dense
        if mask_type == "incoming":
            return mask_1d[None, :]
        else:  # outgoing
            return mask_1d[:, None]

    elif len(weight_shape) == 4:  # Conv
        if mask_type == "incoming":
            return mask_1d[None, None, None, :]
        else:  # outgoing
            return mask_1d[None, None, :, None]
    else:
        raise ValueError(f"Unsupported weight shape: {weight_shape}")


def reset_weights(
    key_tree: PRNGKeyArray,
    reset_mask: PyTree[Bool[Array, "#neurons"]],
    weights: PyTree[Float[Array, "..."]],
    weight_init_fn: Callable = jax.nn.initializers.he_uniform(),
):
    all_layer_names = list(weights.keys())
    all_mask_names = list(reset_mask.keys())  # Just to check layer names are the same
    logs = {}

    assert all_layer_names[-1] == "output", "Last layer should be Dense with name 'output'"

    for idx, layer_name in enumerate(all_layer_names[:-1]):
        assert layer_name in all_mask_names, (
            f"Layer names should be identical: {layer_name} not in {all_mask_names}"
        )

        in_mask_1d = reset_mask[layer_name]
        assert in_mask_1d.dtype == bool, f"Mask type isn't bool for {layer_name}"

        # Reset incoming weights
        in_weight_mask = expand_mask_for_weights(
            in_mask_1d, weights[layer_name].shape, mask_type="incoming"
        )

        random_weights = weight_init_fn(key_tree[layer_name], weights[layer_name].shape)
        weights[layer_name] = jnp.where(in_weight_mask, random_weights, weights[layer_name])

        # Reset outgoing weights
        if idx + 1 < len(all_layer_names):
            next_layer = all_layer_names[idx + 1]
            out_weight_shape = weights[next_layer].shape

            if len(out_weight_shape) == 2:  # Dense layer
                if len(weights[layer_name].shape) == 4:  # Check if previous layer was conv
                    spatial_size = out_weight_shape[0] // in_mask_1d.size
                    out_mask_1d = jnp.tile(in_mask_1d, spatial_size)
                else:  # Dense -> Dense
                    out_mask_1d = in_mask_1d

            elif len(out_weight_shape) == 4:  # Conv layer
                # expected_in_channels = out_weight_shape[2]
                out_mask_1d = in_mask_1d
            else:
                raise ValueError(f"Unsupported weight shape: {out_weight_shape}")

            # Apply outgoing mask
            out_weight_mask = expand_mask_for_weights(
                out_mask_1d, out_weight_shape, mask_type="outgoing"
            )

            weights[next_layer] = jnp.where(
                out_weight_mask, jnp.zeros_like(weights[next_layer]), weights[next_layer]
            )

        # Count reset neurons
        n_reset = in_mask_1d.sum()
        logs[layer_name] = {"nodes_reset": n_reset}

    logs[all_layer_names[-1]] = {"nodes_reset": 0}

    return weights, logs


def reset_optim_params(tx_state, reset_mask):
    """Reset optimizer momentum/variance for neurons identified by reset_mask.

    Handles arbitrarily nested param structures (e.g., twin Q-networks)
    by iterating sub-network chains independently.

    Args:
        tx_state: Base optimizer state tuple (e.g., (EmptyState(), ScaleByAdamState(...))).
        reset_mask: Dict with string keys {layer_name: bool_mask}.
    """

    def _process_chain(layers):
        """Reset momentum for a single sub-network chain (string-keyed layer dict)."""
        new_layers = {}
        all_layer_names = list(layers.keys())

        # Copy all layers first
        for layer_name in all_layer_names:
            new_layers[layer_name] = {
                "kernel": layers[layer_name]["kernel"],
                "bias": layers[layer_name]["bias"],
            }

        # Apply incoming + bias + outgoing resets
        for idx, layer_name in enumerate(all_layer_names[:-1]):
            if layer_name not in reset_mask:
                continue
            in_mask_1d = reset_mask[layer_name]

            # Zero incoming kernel momentum
            in_mask = expand_mask_for_weights(
                in_mask_1d, new_layers[layer_name]["kernel"].shape, mask_type="incoming"
            )
            new_layers[layer_name]["kernel"] = jnp.where(
                in_mask, jnp.zeros_like(new_layers[layer_name]["kernel"]),
                new_layers[layer_name]["kernel"],
            )

            # Zero bias momentum
            new_layers[layer_name]["bias"] = jnp.where(
                in_mask_1d, jnp.zeros_like(new_layers[layer_name]["bias"]),
                new_layers[layer_name]["bias"],
            )

            # Zero outgoing kernel momentum (next layer in chain)
            next_layer = all_layer_names[idx + 1]
            out_shape = new_layers[next_layer]["kernel"].shape

            if len(out_shape) == 2:  # Dense
                if len(new_layers[layer_name]["kernel"].shape) == 4:  # Conv → Dense
                    out_mask_1d = jnp.tile(in_mask_1d, out_shape[0] // in_mask_1d.size)
                else:
                    out_mask_1d = in_mask_1d
            elif len(out_shape) == 4:  # Conv
                out_mask_1d = in_mask_1d
            else:
                continue

            out_mask = expand_mask_for_weights(out_mask_1d, out_shape, mask_type="outgoing")
            new_layers[next_layer]["kernel"] = jnp.where(
                out_mask, jnp.zeros_like(new_layers[next_layer]["kernel"]),
                new_layers[next_layer]["kernel"],
            )

        return new_layers

    def _map_fn(momentum_params):
        """Apply reset across all sub-network chains in momentum dict."""
        result = {}
        for subnetwork_name in momentum_params:
            result[subnetwork_name] = _process_chain(momentum_params[subnetwork_name])
        if isinstance(momentum_params, FrozenDict):
            return freeze(result)
        return result

    def _process_optimizer_state(state):
        """Process a single optimizer sub-state (e.g., ScaleByAdamState)."""
        if isinstance(state, optax.EmptyState):
            return state
        if not hasattr(state, "_fields"):
            return state
        new_fields = {}
        for attr in state._fields:
            val = getattr(state, attr)
            if attr in ("mu", "nu") and isinstance(val, (dict, FrozenDict)) and "params" in val:
                new_val = {"params": _map_fn(val["params"])}
                if isinstance(val, FrozenDict):
                    new_val = freeze(new_val)
                new_fields[attr] = new_val
            else:
                new_fields[attr] = val
        return type(state)(**new_fields)

    if isinstance(tx_state, tuple):
        return tuple(_process_optimizer_state(s) for s in tx_state)
    return _process_optimizer_state(tx_state)


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
