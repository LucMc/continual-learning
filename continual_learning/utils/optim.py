import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from jaxtyping import (
    Bool,
)
from flax.training.train_state import TrainState
import optax

import continual_learning.optim as optim


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

