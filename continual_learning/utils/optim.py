import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from jaxtyping import (
    Bool,
)
import optax

from typing import Callable
from copy import deepcopy

from continual_learning.types import GradientTransformationExtraArgsReset


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
    """Reset optimizer params using reset_mask"""

    def reset_params(tx_state):
        def map_fn(momentum):
            all_layer_names = list(momentum.keys())
            momentum = deepcopy(momentum)
            for idx, layer_name in enumerate(list(momentum.keys())[:-1]):  # momentum is mu/nu
                in_mask_1d = reset_mask[layer_name]

                in_momentum_mask = expand_mask_for_weights(
                    in_mask_1d, momentum[layer_name]["kernel"].shape, mask_type="incoming"
                )

                # Zero the incoming momentums
                momentum[layer_name]["kernel"] = jnp.where(
                    in_momentum_mask,
                    jnp.zeros_like(momentum[layer_name]["kernel"]),
                    momentum[layer_name]["kernel"],
                )

                # Reset outgoing momentum
                if idx + 1 < len(all_layer_names):
                    next_layer = all_layer_names[idx + 1]
                    out_momentum_shape = momentum[next_layer]["kernel"].shape

                    if len(out_momentum_shape) == 2:  # Dense layer
                        if (
                            len(momentum[layer_name]["kernel"].shape) == 4
                        ):  # Check if previous layer was conv
                            spatial_size = out_momentum_shape[0] // in_mask_1d.size
                            out_mask_1d = jnp.tile(in_mask_1d, spatial_size)
                        else:  # Dense -> Dense
                            out_mask_1d = in_mask_1d

                    elif len(out_momentum_shape) == 4:  # Conv layer
                        out_mask_1d = in_mask_1d
                    else:
                        raise ValueError(f"Unsupported weight shape: {out_momentum_shape}")

                    # Apply outgoing mask
                    out_momentum_mask = expand_mask_for_weights(
                        out_mask_1d, out_momentum_shape, mask_type="outgoing"
                    )

                    momentum[next_layer]["kernel"] = jnp.where(
                        out_momentum_mask,
                        jnp.zeros_like(momentum[next_layer]["kernel"]),
                        momentum[next_layer]["kernel"],
                    )
                    # Reset bias too
                    momentum[layer_name]["bias"] = jnp.where(
                        reset_mask[layer_name][None:],
                        jnp.zeros_like(momentum[layer_name]["bias"]),
                        momentum[layer_name]["bias"],
                    )

            return momentum

        new_state_dict = {}

        if isinstance(tx_state, dict):
            return {
                "reset_method": tx_state["reset_method"],
                "tx": (reset_params(tx_state["tx"][0]),) + tx_state["tx"][1:],
            }

        elif isinstance(tx_state, tuple) and len(tx_state) == 2:
            new_elems = []
            for state in tx_state:
                if isinstance(state, optax.EmptyState):
                    new_elems.append(state)
                else:
                    new_elems.append(reset_params(state))  # transformed namedtuple

            return tuple(new_elems)
            # return (reset_params(tx_state[0]),) + tx_state[1:]

        if hasattr(tx_state, "mu"):
            new_state_dict["mu"] = {
                "params": map_fn(
                    tx_state.mu["params"]  # pyright: ignore[reportAttributeAccessIssue]
                )  # flax.traverse_util.path_aware_map(map_fn, tx_state.mu["params"])
            }

        if hasattr(tx_state, "nu"):
            new_state_dict["nu"] = {
                "params": map_fn(tx_state.nu["params"])  # pyright: ignore[reportAttributeAccessIssue]
                # "params": flax.traverse_util.path_aware_map(map_fn, tx_state.nu["params"])
            }

        # copy other attributes
        for attr in tx_state._fields:  # pyright: ignore[reportAttributeAccessIssue]
            if attr not in ["mu", "nu"] and hasattr(tx_state, attr):
                new_state_dict[attr] = getattr(tx_state, attr)

        return type(tx_state)(**new_state_dict)

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
