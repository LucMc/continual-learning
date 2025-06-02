import chex
import jax
import jax.numpy as jnp
from jax import tree_util
from jaxtyping import PRNGKeyArray, PyTree

UTIL_TYPES = [
    "weight",
    "contribution",
    "adaptation",
    "zero_contribution",
    "adaptable_contribution",
    "feature_by_input",
]


def are_pytrees_equal(tree1, tree2):
    """Check if two pytrees have the same structure and equal leaves."""
    try:
        # Check if trees have the same structure
        chex.assert_trees_all_equal_shapes(tree1, tree2)

        # Check if all leaves are equal
        leaves1 = tree_util.tree_leaves(tree1)
        leaves2 = tree_util.tree_leaves(tree2)

        if len(leaves1) != len(leaves2):
            return False

        # Compare each pair of leaves
        for leaf1, leaf2 in zip(leaves1, leaves2):
            if not jnp.array_equal(leaf1, leaf2):
                return False

        return True
    except AssertionError:
        return False


def check_tree_shapes(tree1: PyTree, tree2: PyTree):
    ## assert tree shapes havn't changed
    old_tree_structure = jax.tree_util.tree_structure(tree1)
    new_tree_structure = jax.tree_util.tree_structure(tree2)
    assert old_tree_structure == new_tree_structure, (
        f"Tree structure has changed from {old_tree_structure} to {new_tree_structure}"
    )


def get_layer_bound(layer_shape, init="kaiming", gain=1.0):
    """Calculate initialization bounds similar to https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/cbp_linear.py"""
    if len(layer_shape) == 4:  # Conv layer
        in_channels = layer_shape[2]
        kernel_size = layer_shape[0] * layer_shape[1]
        return jnp.sqrt(1.0 / (in_channels * kernel_size))

    else:  # Linear layer
        in_features = layer_shape[0]
        out_features = layer_shape[1]

        if init == "default":
            bound = jnp.sqrt(1.0 / in_features)
        elif init == "xavier":
            bound = gain * jnp.sqrt(6.0 / (in_features + out_features))
        elif init == "lecun":
            bound = jnp.sqrt(3.0 / in_features)
        else:  # kaiming
            bound = gain * jnp.sqrt(3.0 / in_features)
        return bound


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
