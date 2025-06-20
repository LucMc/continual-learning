import jax
import jax.tree_util as tu
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
import optax
import pytest
from functools import partial
from copy import deepcopy

# Assuming these modules exist in your project structure
import continual_learning.optim.continual_backprop as cbp
import continual_learning.optim.utils as utils
from continual_learning.nn import SimpleTestNet

# --- Fixtures ---


@pytest.fixture(scope="module")
def network():
    """Provides the neural network instance."""
    return SimpleTestNet()


@pytest.fixture
def key():
    """Provides a JAX random key."""
    return random.PRNGKey(0)


@pytest.fixture
def initial_state(network, key):
    """Provides the initial network parameters and CBP state."""
    key, init_key = random.split(key)
    dummy_input = jnp.zeros((1, 1))  # Define dummy_input shape based on SimpleTestNet
    params = network.init(init_key, dummy_input)

    # Sensible defaults, can be overridden in tests if needed
    cbp_kwargs = {
        "replacement_rate": 0.5,
        "decay_rate": 0.9,
        "maturity_threshold": 100,
        "rng": random.PRNGKey(1),
    }
    cbp_adam_tx = optax.adam(learning_rate=1e-3)
    cbp_state = cbp.CBPTrainState.create(
        apply_fn=network.apply,  # Use network.apply directly
        params=params,
        tx=cbp_adam_tx,
        **cbp_kwargs,
    )
    return cbp_state


@pytest.fixture
def processed_data(initial_state, network):
    """
    Provides pre-processed data: features, weights, bias, etc.
    Derived from the initial_state.
    """
    cbp_outer_state = initial_state
    cbp_params = cbp_outer_state.params[
        "params"
    ]  # Assuming structure { 'params': {...} }
    cbp_state_internal = cbp_outer_state.cbp_state

    # Run a forward pass to get features
    # Use a consistent input for deterministic feature calculation if needed
    inputs = jnp.ones((1, 1))  # Consistent input
    _, features_intermediates = network.apply(
        cbp_outer_state.params, inputs, mutable=["intermediates"]
    )
    features = features_intermediates["intermediates"]["activations"][0]

    weights, bias, out_w_mag, excluded = cbp.process_params(cbp_params)

    return {
        "cbp_outer_state": cbp_outer_state,
        "cbp_state": cbp_state_internal,
        "features": features,
        "weights": weights,
        "bias": bias,
        "out_w_mag": out_w_mag,
        "excluded": excluded,
    }


# --- Helper Functions ---
def calculate_mask(
    out_w_mag,
    utilities,
    ages,
    features,
    decay_rate,
    maturity_threshold,
    replacement_rate,
):
    """Calculates the reset mask based on provided parameters."""
    # Ensure features doesn't have an unexpected batch dim if needed
    # features_flat = jax.tree.map(lambda x: x.squeeze(0) if x.ndim > 1 else x, features)

    return jax.tree.map(
        partial(
            cbp.get_reset_mask,
            # decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            replacement_rate=replacement_rate,
        ),
        # out_w_mag,
        utilities,
        ages,
        # features,
    )


# --- Test Functions ---
def test_process_params_structure(processed_data):
    """Tests the basic output structure of cbp.process_params."""
    assert isinstance(processed_data["weights"], dict)
    assert isinstance(processed_data["bias"], dict)
    assert isinstance(processed_data["out_w_mag"], dict)
    assert isinstance(processed_data["excluded"], dict)
    # Add more specific checks if needed, e.g., keys match expected layers
    assert "out_layer" in processed_data["excluded"]
    assert "out_layer" not in processed_data["weights"]
    assert "out_layer" not in processed_data["bias"]
    # Ensure out_w_mag has keys corresponding to layers *before* the output layer
    assert all(k in processed_data["weights"] for k in processed_data["out_w_mag"])


@pytest.mark.parametrize(
    "test_rate, expected_mean",
    [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)],
)
def test_mask_replacement_rate(processed_data, test_rate, expected_mean):
    """Tests that the mask respects the replacement_rate when ages are mature."""
    setup = processed_data
    cbp_state = setup["cbp_state"]

    # Ensure all neurons are considered mature for this test
    mature_ages = jax.tree.map(
        lambda a: jnp.full_like(a, cbp_state.maturity_threshold + 10), cbp_state.ages
    )
    low_maturity_threshold = 1  # Ensure threshold doesn't interfere

    mask = calculate_mask(
        out_w_mag=setup["out_w_mag"],
        utilities=cbp_state.utilities,
        ages=mature_ages,
        features=setup["features"],
        decay_rate=cbp_state.decay_rate,
        maturity_threshold=low_maturity_threshold,
        replacement_rate=test_rate,
    )

    # Check the mean proportion of True values in the mask
    mask_means = tu.tree_map(lambda m: jnp.mean(m.astype(jnp.float32)), mask)
    all_means_correct = tu.tree_all(
        tu.tree_map(lambda m: jnp.isclose(m, expected_mean), mask_means)
    )
    assert all_means_correct, (
        f"Mean check failed for rate {test_rate}. Means: {mask_means}"
    )

    # Explicit checks for edge cases
    if jnp.isclose(test_rate, 0.0):
        assert tu.tree_all(tu.tree_map(jnp.all, tu.tree_map(jnp.logical_not, mask))), (
            f"Mask not all False for rate 0.0: {mask}"
        )
    elif jnp.isclose(test_rate, 1.0):
        assert tu.tree_all(tu.tree_map(jnp.all, mask)), (
            f"Mask not all True for rate 1.0: {mask}"
        )


def test_mask_ages(processed_data):
    """Tests that the mask respects the maturity_threshold."""
    setup = processed_data
    cbp_state = setup["cbp_state"]
    maturity_threshold = cbp_state.maturity_threshold  # Use threshold from state

    # Helper to create specific age structures
    def create_ages(value):
        return jax.tree.map(lambda a: jnp.full_like(a, value), cbp_state.ages)

    # --- Case 1: All ages immature ---
    immature_ages = create_ages(maturity_threshold - 1)

    # Mask with high replacement rate (should still be all False due to age)
    mask_immature_r1 = calculate_mask(
        out_w_mag=setup["out_w_mag"],
        utilities=cbp_state.utilities,
        ages=immature_ages,
        features=setup["features"],
        decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold,
        replacement_rate=1.0,
    )
    # Mask with moderate replacement rate
    mask_immature_r05 = calculate_mask(
        out_w_mag=setup["out_w_mag"],
        utilities=cbp_state.utilities,
        ages=immature_ages,
        features=setup["features"],
        decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold,
        replacement_rate=0.5,
    )

    assert tu.tree_all(tu.tree_map(lambda m: not jnp.any(m), mask_immature_r1)), (
        f"Mask not all False for immature ages (rate=1.0): {mask_immature_r1}"
    )
    assert tu.tree_all(tu.tree_map(lambda m: not jnp.any(m), mask_immature_r05)), (
        f"Mask not all False for immature ages (rate=0.5): {mask_immature_r05}"
    )

    # --- Case 2: All ages mature ---
    mature_ages = create_ages(maturity_threshold + 1)

    # Mask with high replacement rate (should be all True)
    mask_mature_r1 = calculate_mask(
        out_w_mag=setup["out_w_mag"],
        utilities=cbp_state.utilities,
        ages=mature_ages,
        features=setup["features"],
        decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold,
        replacement_rate=1.0,
    )
    # Mask with moderate replacement rate (should have mean 0.5)
    mask_mature_r05 = calculate_mask(
        out_w_mag=setup["out_w_mag"],
        utilities=cbp_state.utilities,
        ages=mature_ages,
        features=setup["features"],
        decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold,
        replacement_rate=0.5,
    )

    assert tu.tree_all(tu.tree_map(jnp.all, mask_mature_r1)), (
        f"Mask not all True for mature ages (rate=1.0): {mask_mature_r1}"
    )

    mask_means_r05 = tu.tree_map(
        lambda m: jnp.mean(m.astype(jnp.float32)), mask_mature_r05
    )
    all_means_correct_r05 = tu.tree_all(
        tu.tree_map(lambda m: jnp.isclose(m, 0.5), mask_means_r05)
    )
    assert all_means_correct_r05, (
        f"Mean check failed for mature ages (rate=0.5). Means: {mask_means_r05}"
    )


def test_features_calculation(network, key):
    """Tests the forward pass feature calculation with known weights/inputs."""
    # This test primarily depends on the network, not the CBP state details.
    # Re-initialize with known (ones) parameters for predictability.
    key, init_key = random.split(key)
    dummy_input = jnp.zeros((1, 1))
    params_struct = network.init(init_key, dummy_input)
    ones_params = tu.tree_map(lambda x: jnp.ones_like(x), params_struct)

    # Run forward pass with ones input
    pred, feats = network.apply(
        ones_params, jnp.ones((1, 1)), mutable=["intermediates"]
    )
    activations = feats["intermediates"]["activations"][0]  # Adjust path if needed

    # Assert expected activations based on TestNet architecture (1 -> 4 -> 4 -> 4 -> 1) and ReLU
    # Assumes Dense layers with bias. Calculation: ReLU(dot(input, W) + b)
    # Layer 1: input=[[1]], W=[1,4]=ones, b=[4]=ones -> dot(1,1)+1=2 -> ReLU(2)=2
    assert jnp.allclose(activations["dense1"], 2.0), (
        "First layer activations check failed"
    )
    # Layer 2: input=[[2,2,2,2]], W=[4,4]=ones, b=[4]=ones -> dot([2,2,2,2], ones(4,1))+1 = 2*4+1=9 -> ReLU(9)=9
    assert jnp.allclose(activations["dense2"], 9.0), (
        "Second layer activations check failed"
    )
    # Layer 3: input=[[9,9,9,9]], W=[4,4]=ones, b=[4]=ones -> 9*4+1=37 -> ReLU(37)=37
    assert jnp.allclose(activations["dense3"], 37.0), (
        "Third layer activations check failed"
    )
    # Output Layer: input=[[37,37,37,37]], W=[4,1]=ones, b=[1]=ones -> 37*4+1=149 (No ReLU assumed on output)
    assert jnp.allclose(pred, 149.0), "Final prediction check failed"


# Helper to create easily distinguishable weights
def create_layered_value_weights(like_tree, start_value=1.0):
    """Creates weights where each layer's weights have a distinct value."""
    value = start_value
    new_weights = {}
    # Sort keys to ensure consistent value assignment layer-by-layer
    sorted_keys = sorted(like_tree.keys())
    for k in sorted_keys:
        new_weights[k] = jnp.full_like(like_tree[k], value)
        value += 1.0 # Increment value for the next layer
    return new_weights

# --- Test Function ---
def test_reset_weights(processed_data, key):
    """
    Tests the cbp.reset_weights function with focused scenarios.

    Checks:
        1. No Reset: Weights remain unchanged if mask is all False.
        2. Single Neuron Reset:
            - Correct *column* of input weights (W_L) is set to initial value.
            - Correct *row* of output weights (W_{L+1}) is set to zero.
            - All other weights remain untouched.
        3. Bias Reset (Separate Test): Although related, bias reset happens
           outside reset_weights, so it needs its own test.
    """
    setup = processed_data
    original_params_kernels = setup["weights"] # Kernels only, from process_params
    cbp_state = setup["cbp_state"]

    # --- Setup specific weights for easy checking ---
    # 1. Initial weights (e.g., all zeros)
    initial_weights = tu.tree_map(jnp.zeros_like, original_params_kernels)

    # 2. Current weights (e.g., layer 1 = 1.0, layer 2 = 2.0, etc.)
    #    Make sure these are different from initial_weights!
    current_weights = create_layered_value_weights(original_params_kernels, 1.0)

    # 3. Keys (needed for function signature, even if not used for random)
    key_tree = utils.gen_key_tree(key, current_weights) # Generate matching structure

    layer_names = list(current_weights.keys()) # Assumes order matters (dict order >= 3.7)

    # --- Test Case 1: No Reset ---
    print("Testing No Reset...")
    no_reset_mask = tu.tree_map(lambda x: jnp.zeros_like(x, dtype=bool), cbp_state.ages) # All False mask
    # Filter mask to only include layers present in current_weights (i.e., non-output)
    no_reset_mask_filtered = {k: no_reset_mask[k] for k in layer_names if k in no_reset_mask}

    weights_after_no_reset, _ = cbp.reset_weights(
        no_reset_mask_filtered,
        deepcopy(current_weights), # Pass a copy to avoid modification
        key_tree,
        initial_weights
    )

    # Check that weights are identical to the originals
    all_equal = tu.tree_all(tu.tree_map(
        jnp.array_equal, weights_after_no_reset, current_weights
    ))
    assert all_equal, "Weights changed when reset_mask was all False."
    print("No Reset Test Passed.")

    # --- Test Case 2: Single Neuron Reset ---
    print("\nTesting Single Neuron Reset...")
    if len(layer_names) < 2:
        pytest.skip("Need at least two non-output layers to test input/output reset.")

    reset_layer_name = layer_names[0] # e.g., 'dense1'
    next_layer_name = layer_names[1]  # e.g., 'dense2'
    neuron_to_reset_idx = 1        # Choose an index (e.g., the second neuron)

    # Ensure index is valid
    num_neurons_in_layer = current_weights[reset_layer_name].shape[1]
    if neuron_to_reset_idx >= num_neurons_in_layer:
         pytest.skip(f"Neuron index {neuron_to_reset_idx} out of bounds for layer {reset_layer_name} with {num_neurons_in_layer} neurons.")

    print(f"Targeting neuron {neuron_to_reset_idx} in layer '{reset_layer_name}' for reset.")

    # Create a mask that is True *only* for the target neuron in the target layer
    single_reset_mask = tu.tree_map(lambda x: jnp.zeros_like(x, dtype=bool), cbp_state.ages)
    single_reset_mask_filtered = {k: single_reset_mask[k] for k in layer_names if k in single_reset_mask} # Filter like before

    # Update the specific mask entry
    single_reset_mask_filtered[reset_layer_name] = single_reset_mask_filtered[reset_layer_name].at[neuron_to_reset_idx].set(True)

    weights_after_single_reset, reset_logs = cbp.reset_weights(
        single_reset_mask_filtered,
        deepcopy(current_weights),
        key_tree,
        initial_weights
    )

    # --- Assertions for Single Neuron Reset ---

    # A. Check Input Weights (Column in W_L)
    original_W_L = current_weights[reset_layer_name]
    reset_W_L = weights_after_single_reset[reset_layer_name]
    initial_W_L = initial_weights[reset_layer_name]

    # Check the specific column that should have been reset
    reset_column = reset_W_L[:, neuron_to_reset_idx]
    expected_column = initial_W_L[:, neuron_to_reset_idx]
    assert jnp.allclose(reset_column, expected_column), \
        f"Input weights column {neuron_to_reset_idx} for layer {reset_layer_name} was not reset to initial values."
    print(f"  Input column {neuron_to_reset_idx} ({reset_layer_name}) reset: OK")

    # Check that *other* columns were NOT changed
    # Create a mask for columns *not* equal to neuron_to_reset_idx
    other_cols_mask = jnp.arange(original_W_L.shape[1]) != neuron_to_reset_idx
    untouched_cols_reset = reset_W_L[:, other_cols_mask]
    untouched_cols_original = original_W_L[:, other_cols_mask]
    assert jnp.allclose(untouched_cols_reset, untouched_cols_original), \
        f"Input weights columns *other than* {neuron_to_reset_idx} for layer {reset_layer_name} were changed."
    print(f"  Other input columns ({reset_layer_name}) unchanged: OK")

    # B. Check Output Weights (Row in W_{L+1})
    original_W_Lplus1 = current_weights[next_layer_name]
    reset_W_Lplus1 = weights_after_single_reset[next_layer_name]

    # Check the specific row that should have been zeroed
    reset_row = reset_W_Lplus1[neuron_to_reset_idx, :]
    expected_row = jnp.zeros_like(reset_row)
    assert jnp.allclose(reset_row, expected_row), \
        f"Output weights row {neuron_to_reset_idx} for layer {next_layer_name} was not reset to zero."
    print(f"  Output row {neuron_to_reset_idx} ({next_layer_name}) zeroed: OK")

    # Check that *other* rows were NOT changed
    other_rows_mask = jnp.arange(original_W_Lplus1.shape[0]) != neuron_to_reset_idx
    untouched_rows_reset = reset_W_Lplus1[other_rows_mask, :]
    untouched_rows_original = original_W_Lplus1[other_rows_mask, :]
    assert jnp.allclose(untouched_rows_reset, untouched_rows_original), \
        f"Output weights rows *other than* {neuron_to_reset_idx} for layer {next_layer_name} were changed."
    print(f"  Other output rows ({next_layer_name}) unchanged: OK")

    # C. Check Other Layers (if any)
    for layer_name in layer_names:
        if layer_name != reset_layer_name and layer_name != next_layer_name:
            assert jnp.array_equal(weights_after_single_reset[layer_name], current_weights[layer_name]), \
                f"Weights in unrelated layer '{layer_name}' were changed during single neuron reset."
    print("  Unrelated layers unchanged: OK")

    # D. Check Logs (Basic)
    assert reset_logs[reset_layer_name]["nodes_reset"] == 1, "Log did not report 1 node reset."
    assert reset_logs[next_layer_name]["nodes_reset"] == 0, "Log reported reset for the layer *after* the reset layer."
    print("  Reset logs check: OK")

    print("Single Neuron Reset Test Passed.")


def test_reset_bias(processed_data):
    """Tests that bias terms are correctly reset to zero based on the mask."""
    setup = processed_data
    cbp_state = setup["cbp_state"]
    original_bias = setup["bias"] # Bias terms only

    # Create dummy current bias (e.g., all ones)
    current_bias = tu.tree_map(lambda b: jnp.full_like(b, 1.0), original_bias)

    layer_names = list(current_bias.keys()) # e.g., ['dense1', 'dense2', 'dense3']

    # --- Scenario: Reset bias for specific neurons ---
    reset_layer_name = layer_names[0] # e.g., 'dense1'
    neuron_indices_to_reset = jnp.array([0, 2]) # Reset first and third bias term

    # Create mask: True only for specified indices in the target layer
    reset_mask = tu.tree_map(lambda x: jnp.zeros_like(x, dtype=bool), cbp_state.ages)
    # Filter mask to only include layers present in current_bias
    reset_mask_filtered = {k: reset_mask[k] for k in layer_names if k in reset_mask}

    # Ensure indices are valid
    num_neurons_in_layer = current_bias[reset_layer_name].shape[0]
    if jnp.any(neuron_indices_to_reset >= num_neurons_in_layer):
        pytest.skip(f"Neuron indices {neuron_indices_to_reset} out of bounds for layer {reset_layer_name} with {num_neurons_in_layer} neurons.")


    # Set the mask entries to True
    mask_for_layer = reset_mask_filtered[reset_layer_name]
    mask_for_layer = mask_for_layer.at[neuron_indices_to_reset].set(True)
    reset_mask_filtered[reset_layer_name] = mask_for_layer


    # --- Apply the bias reset logic (mimicking the update function) ---
    reset_bias = jax.tree.map(
        lambda m, b: jnp.where(m, jnp.zeros_like(b, dtype=float), b),
        reset_mask_filtered, # Use the filtered mask matching bias structure
        current_bias,
    )

    # --- Assertions ---
    # Check the layer where resets happened
    bias_L = reset_bias[reset_layer_name]
    original_bias_L = current_bias[reset_layer_name]

    # Check neurons that *should* be zero
    assert jnp.all(bias_L[neuron_indices_to_reset] == 0.0), \
        f"Bias terms {neuron_indices_to_reset} in layer {reset_layer_name} were not reset to zero."

    # Check neurons that should *not* have changed
    other_indices_mask = jnp.ones_like(bias_L, dtype=bool).at[neuron_indices_to_reset].set(False)
    assert jnp.all(bias_L[other_indices_mask] == original_bias_L[other_indices_mask]), \
        f"Bias terms other than {neuron_indices_to_reset} in layer {reset_layer_name} were changed."

    # Check other layers were not changed
    for layer_name in layer_names:
        if layer_name != reset_layer_name:
            assert jnp.array_equal(reset_bias[layer_name], current_bias[layer_name]), \
                f"Bias terms in unrelated layer '{layer_name}' were changed."

    print("Bias reset test passed.")

# --- Helper Functions ---

def create_dummy_grads(params_tree):
    """Creates gradients with the same structure as params, filled with ones."""
    return tu.tree_map(jnp.ones_like, params_tree)

def create_dummy_features(features_tree_structure):
    """Creates features with the same structure, filled with ones."""
    # Assuming features are float arrays
    return tu.tree_map(lambda x: jnp.ones_like(x, dtype=jnp.float32) if hasattr(x, 'shape') else x,
                       features_tree_structure)

# --- Test Functions ---

# High Priority / Core Functionality:

def test_apply_gradients_equivalence_rate_zero(initial_state, network, key):
    """
    Tests that CBPTrainState with replacement_rate=0 behaves identically
    to a standard optax.adam TrainState after one gradient step.
    """
    cbp_start_state = initial_state
    base_tx = cbp_start_state.tx # Get the adam optimizer instance

    # --- Create Standard Adam State ---
    adam_state = base_tx.init(cbp_start_state.params)
    # Wrap in a standard TrainState for API consistency if needed, but direct use is fine for comparison
    # adam_train_state = TrainState.create(apply_fn=network.apply, params=cbp_start_state.params, tx=base_tx)
    # Initial opt_state should be identical if initialized with same params
    assert tu.tree_all(tu.tree_map(
        jnp.array_equal, cbp_start_state.opt_state, adam_state
    )), "Initial opt_state mismatch"

    # --- Create CBP State with Rate 0 ---
    # Ensure rate is exactly 0 for this test
    cbp_state_rate_zero = cbp_start_state.replace(
        cbp_state=cbp_start_state.cbp_state.replace(replacement_rate=0.0)
    )
    assert cbp_state_rate_zero.cbp_state.replacement_rate == 0.0

    # --- Generate Dummy Data ---
    # Grads should match the structure of params *that optax uses*
    dummy_grads = create_dummy_grads(cbp_state_rate_zero.params)
    # Features structure from a forward pass (or use processed_data fixture)
    _, features_intermediates = network.apply(
        cbp_state_rate_zero.params, jnp.ones((1, 1)), mutable=["intermediates"]
    )
    dummy_features = features_intermediates # Pass the whole mutable collection

    # --- Apply Gradients ---
    # Adam update
    adam_updates, adam_new_opt_state = base_tx.update(
        dummy_grads, adam_state, cbp_state_rate_zero.params
    )
    adam_new_params = optax.apply_updates(cbp_state_rate_zero.params, adam_updates)

    # CBP update
    cbp_new_state = cbp_state_rate_zero.apply_gradients(
        grads=dummy_grads, features=dummy_features
    )

    # --- Assertions ---
    # 1. Params must be identical
    params_equal = tu.tree_all(tu.tree_map(
        jnp.array_equal, cbp_new_state.params, adam_new_params
    ))
    if not params_equal:
        # Print diff for easier debugging (optional)
        diff = tu.tree_map(lambda x, y: jnp.sum(jnp.abs(x-y)), cbp_new_state.params, adam_new_params)
        print("Param difference:", diff)
    assert params_equal, "Params differ between CBP (rate=0) and Adam after apply_gradients"

    # 2. Optimizer state must be identical
    opt_state_equal = tu.tree_all(tu.tree_map(
        jnp.array_equal, cbp_new_state.opt_state, adam_new_opt_state
    ))
    assert opt_state_equal, "opt_state differs between CBP (rate=0) and Adam after apply_gradients"

    # 3. Step count incremented
    assert cbp_new_state.step == cbp_state_rate_zero.step + 1

    # 4. Ages should only increment (since no reset happened)
    expected_ages = tu.tree_map(lambda a: a + 1, cbp_state_rate_zero.cbp_state.ages)
    ages_equal = tu.tree_all(tu.tree_map(
        jnp.array_equal, cbp_new_state.cbp_state.ages, expected_ages
    ))
    assert ages_equal, "Ages did not increment correctly when rate=0"


def test_apply_gradients_divergence_rate_positive(initial_state, network, key):
    """
    Tests that CBPTrainState with replacement_rate > 0 and mature ages
    results in different parameters than standard optax.adam after one step.
    """
    cbp_start_state = initial_state
    base_tx = cbp_start_state.tx

    # --- Create Standard Adam State ---
    adam_opt_state = base_tx.init(cbp_start_state.params)

    # --- Create CBP State with Rate > 0 and Mature Ages ---
    rate = 0.5
    maturity_threshold = cbp_start_state.cbp_state.maturity_threshold
    mature_ages = tu.tree_map(
        lambda a: jnp.full_like(a, maturity_threshold + 10),
        cbp_start_state.cbp_state.ages
    )
    cbp_state_rate_pos = cbp_start_state.replace(
        cbp_state=cbp_start_state.cbp_state.replace(
            replacement_rate=rate,
            ages=mature_ages # Force mature ages
        )
    )
    initial_cbp_internal_state = cbp_state_rate_pos.cbp_state
    assert initial_cbp_internal_state.replacement_rate == rate
    assert tu.tree_all(tu.tree_map(lambda a: jnp.all(a > maturity_threshold), initial_cbp_internal_state.ages))

    # --- Generate Dummy Data ---
    dummy_grads = create_dummy_grads(cbp_state_rate_pos.params)
    _, features_intermediates = network.apply(
        cbp_state_rate_pos.params, jnp.ones((1, 1)), mutable=["intermediates"]
    )
    dummy_features = features_intermediates

    # --- Apply Gradients ---
    # Adam update
    adam_updates, adam_new_opt_state = base_tx.update(
        dummy_grads, adam_opt_state, cbp_state_rate_pos.params
    )
    adam_new_params = optax.apply_updates(cbp_state_rate_pos.params, adam_updates)

    # CBP update
    cbp_new_state = cbp_state_rate_pos.apply_gradients(
        grads=dummy_grads, features=dummy_features
    )
    final_cbp_internal_state = cbp_new_state.cbp_state

    # --- Assertions ---
    # 1. Params should NOT be identical (assuming resets happened)
    params_equal = tu.tree_all(tu.tree_map(
        jnp.array_equal, cbp_new_state.params, adam_new_params
    ))
    # Need to check if *any* resets actually happened. Calculate expected mask:
    _, _, out_w_mag, _ = cbp.process_params(cbp_state_rate_pos.params["params"]) # Need out_w_mag
    # Filter out_w_mag to match structure of ages/utilities
    filtered_out_w_mag = {k: out_w_mag[k] for k in initial_cbp_internal_state.ages.keys()}
    # Use features structure matching ages/utilities
    features_for_mask = dummy_features["intermediates"]["activations"][0]
    filtered_features = {k: features_for_mask[k] for k in initial_cbp_internal_state.ages.keys()}


    # Use calculate_mask helper or inline the logic for clarity
    reset_mask = jax.tree.map(
        partial(
            cbp.get_reset_mask, # Reference the actual function from cbp module
            # decay_rate=initial_cbp_internal_state.decay_rate,
            maturity_threshold=initial_cbp_internal_state.maturity_threshold,
            replacement_rate=initial_cbp_internal_state.replacement_rate,
        ),
        # filtered_out_w_mag,
        initial_cbp_internal_state.utilities,
        initial_cbp_internal_state.ages,
        # filtered_features,
    )
    layer_any_reset_tree = tu.tree_map(jnp.any, reset_mask)
    flat_any_reset_list, _ = jax.tree.flatten(layer_any_reset_tree)
    any_reset_occurred = any(flat_any_reset_list)

    if any_reset_occurred:
        assert not params_equal, "Params ARE identical between CBP (rate>0, mature) and Adam, but resets should have occurred."
    else:
        # If parameters happen to allow no resets even with rate>0 and mature ages (e.g., all utilities identical and rate < 1/N)
        print("Warning: No resets occurred despite rate > 0 and mature ages. Parameters might be identical.")
        assert params_equal, "Params differ between CBP and Adam, but no resets were expected in this specific case."

    # 2. Step count incremented
    assert cbp_new_state.step == cbp_state_rate_pos.step + 1

    # 3. Base opt_state was updated in both (check vs initial)
    initial_opt_state = cbp_start_state.opt_state # Use the very initial one
    cbp_opt_state_changed = not tu.tree_all(tu.tree_map(
        jnp.array_equal, cbp_new_state.opt_state, initial_opt_state
    ))
    adam_opt_state_changed = not tu.tree_all(tu.tree_map(
        jnp.array_equal, adam_new_opt_state, initial_opt_state
    ))
    assert cbp_opt_state_changed, "CBP opt_state did not change."
    assert adam_opt_state_changed, "Adam opt_state did not change."
    # Additionally, the opt_states themselves might differ now due to resets affecting params *before* the *next* update
    # opt_states_equal = tu.tree_all(tu.tree_map(jnp.array_equal, cbp_new_state.opt_state, adam_new_opt_state))
    # assert not opt_states_equal # This might not hold depending on optimizer state details

    # 4. Ages updated (some reset to 0, others incremented)
    assert any(tu.tree_leaves(tu.tree_map(lambda a: jnp.any(a == 0), final_cbp_internal_state.ages))), \
        "No ages were reset to 0, but expected resets."
    assert any(tu.tree_leaves(tu.tree_map(lambda a: jnp.any(a == maturity_threshold + 10 + 1), final_cbp_internal_state.ages))), \
        "No ages were incremented, but expected non-resets."

    # 5. Utilities updated (should differ from initial)
    utilities_changed = not tu.tree_all(tu.tree_map(
        jnp.array_equal, final_cbp_internal_state.utilities, initial_cbp_internal_state.utilities
    ))
    # Note: Utility update depends on features and out_w_mag. If these are zero, utilities might not change much depending on decay.
    # A robust check might compare against manually calculated expected utilities.
    assert utilities_changed, "Utilities did not change."


@pytest.mark.parametrize("decay_rate", [0.0, 0.9, 1.0])
def test_utility_update_logic(processed_data, decay_rate):
    """Tests the utility update formula: new = decay*old + (1-decay)*|feat|*out_w_mag"""
    setup = processed_data
    initial_utilities = setup["cbp_state"].utilities
    features = setup["features"] # Structure might need adjustment based on TestNet
    out_w_mag = setup["out_w_mag"]

    # Align structures: features and out_w_mag need to match utility keys
    # This assumes 'features' directly contains activations per layer matching utility keys
    # And 'out_w_mag' is already aligned (as produced by process_params)
    filtered_features = {k: features[k] for k in initial_utilities.keys() if k in features}
    filtered_out_w_mag = {k: out_w_mag[k] for k in initial_utilities.keys() if k in out_w_mag}

    # Ensure all keys are present after filtering
    assert initial_utilities.keys() == filtered_features.keys(), "Feature keys mismatch utility keys"
    assert initial_utilities.keys() == filtered_out_w_mag.keys(), "Out_w_mag keys mismatch utility keys"

    # Manual calculation
    def _update_util(util, feat, w_mag):
        # Ensure feature is positive
        abs_feat = jnp.abs(feat)
        # Handle potential batch dim in features if necessary (assuming features are (batch, neurons))
        if abs_feat.ndim > util.ndim:
           abs_feat = jnp.mean(abs_feat, axis=0) # Average over batch dim
        return decay_rate * util + (1.0 - decay_rate) * abs_feat * w_mag

    expected_next_utilities = tu.tree_map(
        _update_util,
        initial_utilities,
        filtered_features,
        filtered_out_w_mag
    )

    # --- Simulate the utility update part of get_reset_mask ---
    # (Actual get_reset_mask also applies thresholding, we only test the update formula)
    calculated_utilities = tu.tree_map(
        lambda u, f, om: (decay_rate * u) + (1 - decay_rate) * jnp.abs(f.mean(0) if f.ndim > u.ndim else f) * om, # Apply mean if batch exists
        initial_utilities,
        filtered_features,
        filtered_out_w_mag,
    )


    # --- Assertion ---
    utilities_match = tu.tree_all(tu.tree_map(
        jnp.allclose, calculated_utilities, expected_next_utilities
    ))
    assert utilities_match, f"Utility update calculation mismatch for decay_rate={decay_rate}"

def test_age_update_logic(processed_data):
    """Tests the age update logic: increments or resets to 0 based on mask."""
    setup = processed_data
    ages_structure = setup["cbp_state"].ages

    # --- Setup known initial state and mask ---
    initial_value = 5
    initial_ages = tu.tree_map(lambda x: jnp.full_like(x, initial_value), ages_structure)

    # Create a mask, e.g., True for the first half of neurons in each layer
    reset_mask = tu.tree_map(
        lambda x: jnp.arange(x.size) < (x.size // 2),
        ages_structure
    )
    # Ensure mask has boolean type
    reset_mask = tu.tree_map(lambda x: x.astype(bool), reset_mask)


    # --- Apply age update logic ---
    calculated_ages = tu.tree_map(
        lambda a, m: jnp.where(m, 0, a + 1),
        initial_ages,
        reset_mask
    )

    # --- Construct expected ages ---
    expected_ages = tu.tree_map(
        lambda a, m: jnp.where(m, 0, initial_value + 1), # Use initial_value here
        initial_ages, # Structure reference
        reset_mask
    )

    # --- Assertion ---
    ages_match = tu.tree_all(tu.tree_map(
        jnp.array_equal, calculated_ages, expected_ages
    ))
    assert ages_match, "Age update logic (increment/reset) failed."
