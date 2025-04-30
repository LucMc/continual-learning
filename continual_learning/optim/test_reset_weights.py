import jax
import jax.tree_util as tu
import jax.random as random
import jax.numpy as jnp
from flax.core import FrozenDict # Import FrozenDict if needed, though regular dicts work fine for testing inputs here
import pytest
from copy import deepcopy

# Assuming your CBP code is in a file named cbp_module.py
import continual_learning.optim.continual_backprop as cbp
import continual_learning.optim.utils as utils # Assuming utils contains gen_key_tree

def assert_array_equal(l1,l2):
    assert jnp.array_equal(l1,l2)

# Helper function for comparing PyTrees
def assert_pytrees_equal(tree1, tree2):
    tu.tree_map(
        lambda l1, l2: assert_array_equal(l1, l2),
        tree1,
        tree2
    )

def assert_pytrees_not_equal(tree1, tree2):
     tree_eq = tu.tree_map(lambda l1, l2: jnp.array_equal(l1, l2), tree1, tree2)
     leaves, _ = tu.tree_flatten(tree_eq)
     assert not all(leaves), "PyTrees were unexpectedly equal"

# --- Fixtures ---

@pytest.fixture
def simple_weights():
    """Creates a simple 3-layer weight structure (kernels only)."""
    # Input -> Dense1(3 neurons) -> Dense2(4 neurons) -> Dense3(2 neurons)
    weights = {
        "dense1": jnp.arange(1 * 3, dtype=jnp.float32).reshape(1, 3) + 1.0, # Shape (In, Out) = (1, 3)
        "dense2": jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4) + 10.0, # Shape (3, 4)
        "dense3": jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2) + 100.0 # Shape (4, 2)
    }
    return weights

@pytest.fixture
def initial_simple_weights(simple_weights):
    """Creates initial weights, distinct from simple_weights."""
    # Using negative values for easy distinction
    return tu.tree_map(lambda w: -jnp.ones_like(w) * 99.0, simple_weights)
    # return tu.tree_map(lambda w: w * -10.0, simple_weights) # Alternative

@pytest.fixture
def simple_key_tree(simple_weights):
    """Generates a PRNGKey tree matching the simple_weights structure."""
    key = random.PRNGKey(42)
    # Use a helper if available, otherwise create manually matching keys
    # Assuming utils.gen_key_tree works like this:
    try:
        return utils.gen_key_tree(key, simple_weights)
    except AttributeError: # Fallback if gen_key_tree isn't found or has different signature
         keys = {}
         layer_names = list(simple_weights.keys())
         split_keys = random.split(key, len(layer_names))
         for i, name in enumerate(layer_names):
             keys[name] = split_keys[i]
         return keys


# --- Test Cases ---

def test_reset_weights_no_reset(simple_weights, initial_simple_weights, simple_key_tree):
    """
    Test case: reset_mask is all False. Weights should not change.
    Logs should report 0 resets for all layers.
    """
    original_weights = deepcopy(simple_weights)
    num_neurons = {name: w.shape[1] for name, w in simple_weights.items()}
    num_neurons["dense3"] = simple_weights["dense3"].shape[1] # Output layer neurons - though mask isn't used

    # Create a mask where all neurons are marked False for reset
    reset_mask = {
        "dense1": jnp.zeros(num_neurons["dense1"], dtype=bool), # 3 neurons
        "dense2": jnp.zeros(num_neurons["dense2"], dtype=bool), # 4 neurons
        # "dense3": jnp.zeros(num_neurons["dense3"], dtype=bool) # Mask for last layer isn't used by loop
    }

    updated_weights, logs = cbp.reset_weights(
        reset_mask,
        simple_weights, # Pass a mutable copy if needed, but JAX ops return new arrays
        simple_key_tree,
        initial_simple_weights
    )

    # 1. Check weights are unchanged
    assert_pytrees_equal(original_weights, updated_weights)

    # 2. Check logs
    assert logs["dense1"]["nodes_reset"] == 0
    assert logs["dense2"]["nodes_reset"] == 0
    assert logs["dense3"]["nodes_reset"] == 0 # Last layer explicitly set to 0
    assert len(logs) == 3 # Ensure all layers are logged

def test_reset_weights_full_reset_first_layer(simple_weights, initial_simple_weights, simple_key_tree):
    """
    Test case: Fully reset the first hidden layer ('dense1').
    - dense1 weights (incoming) should become initial_weights['dense1'].
    - dense2 weights (outgoing from dense1) columns should become zero.
    - dense3 weights should remain unchanged in this step.
    - Logs should report resets for dense1, 0 for others.
    """
    original_weights = deepcopy(simple_weights)
    num_neurons = {name: w.shape[1] for name, w in simple_weights.items()}

    # Reset all neurons in dense1
    reset_mask = {
        "dense1": jnp.ones(num_neurons["dense1"], dtype=bool), # Reset all 3 neurons
        "dense2": jnp.zeros(num_neurons["dense2"], dtype=bool), # Don't reset dense2
    }

    # Make a mutable copy to pass into the function if it modifies in-place
    # Although JAX operations like jnp.where typically return new arrays,
    # the function structure `layer_w[in_layer] = ...` suggests potential
    # in-place modification semantics if layer_w wasn't immutable (like FrozenDict).
    # Using deepcopy ensures the original is untouched for comparison.
    current_weights = deepcopy(simple_weights)

    updated_weights, logs = cbp.reset_weights(
        reset_mask,
        current_weights, # Pass the mutable copy
        simple_key_tree,
        initial_simple_weights
    )

    # 1. Check weights for dense1 (incoming) - should match initial weights
    assert_array_equal(updated_weights["dense1"], initial_simple_weights["dense1"])

    # 2. Check weights for dense2 (outgoing from dense1) - columns corresponding to dense1 neurons should be zero
    # Since all dense1 neurons are reset, all columns of dense2 should be zero.
    expected_dense2_weights = jnp.zeros_like(original_weights["dense2"])
    assert_array_equal(updated_weights["dense2"], expected_dense2_weights)

    # 3. Check weights for dense3 (unaffected by dense1 reset)
    assert_array_equal(updated_weights["dense3"], original_weights["dense3"])

    # 4. Check logs
    assert logs["dense1"]["nodes_reset"] == num_neurons["dense1"] # Should be 3
    assert logs["dense2"]["nodes_reset"] == 0 # dense2 mask was False
    assert logs["dense3"]["nodes_reset"] == 0 # Last layer always 0
    assert len(logs) == 3

    # 5. Ensure overall structure is different from original (due to reset)
    assert_pytrees_not_equal(original_weights, updated_weights)

def test_reset_weights_partial_reset_second_layer(simple_weights, initial_simple_weights, simple_key_tree):
    """
    Test case: Partially reset the second hidden layer ('dense2').
    - Reset neurons 0 and 2 in 'dense2'.
    - dense2 weights (incoming): Rows 0 and 2 should match initial_weights['dense2']. Rows 1 and 3 should be original.
    - dense3 weights (outgoing from dense2): Columns 0 and 2 should become zero. Columns 1 and 3 should be original.
    - dense1 weights should remain unchanged.
    - Logs should report 0 resets for dense1, 2 for dense2, 0 for dense3.
    """
    original_weights = deepcopy(simple_weights)
    num_neurons = {name: w.shape[1] for name, w in simple_weights.items()} # {d1:3, d2:4, d3:2}

    # Reset neurons 0 and 2 in dense2
    dense2_mask = jnp.array([True, False, True, False], dtype=bool)
    reset_mask = {
        "dense1": jnp.zeros(num_neurons["dense1"], dtype=bool), # Don't reset dense1
        "dense2": dense2_mask,                                # Reset neurons 0, 2 in dense2
    }

    current_weights = deepcopy(simple_weights)
    updated_weights, logs = cbp.reset_weights(
        reset_mask,
        current_weights,
        simple_key_tree,
        initial_simple_weights
    )

    # 1. Check dense1 weights (unaffected)
    assert_array_equal(updated_weights["dense1"], original_weights["dense1"])

    # 2. Check dense2 weights (incoming to dense2)
    # Rows 0 and 2 should be from initial_weights, Rows 1 and 3 from original
    expected_dense2 = original_weights["dense2"].at[0, :].set(initial_simple_weights["dense2"][0, :])
    expected_dense2 = expected_dense2.at[2, :].set(initial_simple_weights["dense2"][2, :])
    assert_array_equal(updated_weights["dense2"], expected_dense2)

    # 3. Check dense3 weights (outgoing from dense2)
    # Columns 0 and 2 should be zero, Columns 1 and 3 from original
    expected_dense3 = original_weights["dense3"].at[:, 0].set(0.0)
    expected_dense3 = expected_dense3.at[:, 2].set(0.0)
    assert_array_equal(updated_weights["dense3"], expected_dense3)

    # 4. Check logs
    assert logs["dense1"]["nodes_reset"] == 0
    assert logs["dense2"]["nodes_reset"] == 2 # We reset 2 neurons in dense2
    assert logs["dense3"]["nodes_reset"] == 0 # Last layer always 0
    assert len(logs) == 3

    # 5. Ensure overall structure is different from original
    assert_pytrees_not_equal(original_weights, updated_weights)


def test_reset_weights_multi_layer_reset(simple_weights, initial_simple_weights, simple_key_tree):
    """
    Test case: Reset neurons in multiple layers simultaneously.
    - Reset neuron 1 in 'dense1'.
    - Reset neuron 0 in 'dense2'.
    """
    original_weights = deepcopy(simple_weights)
    num_neurons = {name: w.shape[1] for name, w in simple_weights.items()}

    reset_mask = {
        "dense1": jnp.array([False, True, False], dtype=bool), # Reset neuron 1
        "dense2": jnp.array([True, False, False, False], dtype=bool), # Reset neuron 0
    }

    current_weights = deepcopy(simple_weights)
    updated_weights, logs = cbp.reset_weights(
        reset_mask,
        current_weights,
        simple_key_tree,
        initial_simple_weights
    )

    # --- Check effects of dense1 reset ---
    # dense1 incoming: row 1 should be initial, rows 0, 2 original
    expected_dense1 = original_weights["dense1"].at[0, 1].set(initial_simple_weights["dense1"][0, 1]) # Input size is 1
    assert_array_equal(updated_weights["dense1"], expected_dense1)

    # dense2 outgoing from dense1: col 1 should be zero, cols 0, 2, 3 original *initially*
    # BUT dense2 is ALSO modified by the dense2 reset mask. Need to combine effects.
    expected_dense2_after_d1 = original_weights["dense2"].at[:, 1].set(0.0)

    # --- Check effects of dense2 reset ---
    # dense2 incoming: row 0 should be initial, rows 1, 2, 3 original (applied *after* dense1 effect)
    expected_dense2_final = expected_dense2_after_d1.at[0, :].set(initial_simple_weights["dense2"][0, :])
    assert_array_equal(updated_weights["dense2"], expected_dense2_final)

    # dense3 outgoing from dense2: col 0 should be zero, cols 1 original
    expected_dense3 = original_weights["dense3"].at[:, 0].set(0.0)
    assert_array_equal(updated_weights["dense3"], expected_dense3)

    # --- Check logs ---
    assert logs["dense1"]["nodes_reset"] == 1
    assert logs["dense2"]["nodes_reset"] == 1
    assert logs["dense3"]["nodes_reset"] == 0
    assert len(logs) == 3

    # --- Ensure overall structure is different ---
    assert_pytrees_not_equal(original_weights, updated_weights)

# You can add more tests, e.g., for different network structures if needed.
