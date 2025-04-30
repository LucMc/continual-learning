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
from continual_learning.nn import SimpleTestNet # Make sure TestNet is importable

# --- Fixtures ---

@pytest.fixture(scope="module") # Scope to module if SimpleTestNet() is expensive/stateless
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
    dummy_input = jnp.zeros((1, 1)) # Define dummy_input shape based on TestNet
    params = network.init(init_key, dummy_input)

    # Sensible defaults, can be overridden in tests if needed
    cbp_kwargs = {
        "replacement_rate": 0.5,
        "decay_rate": 0.9,
        "maturity_threshold": 100,
        "rng": random.PRNGKey(1), # Separate RNG for CBP state
    }
    cbp_adam_tx = optax.adam(learning_rate=1e-3)
    cbp_state = cbp.CBPTrainState.create(
        apply_fn=network.apply, # Use network.apply directly
        params=params,
        tx=cbp_adam_tx,
        **cbp_kwargs
    )
    return cbp_state

@pytest.fixture
def processed_data(initial_state, network):
    """
    Provides pre-processed data: features, weights, bias, etc.
    Derived from the initial_state.
    """
    cbp_outer_state = initial_state
    cbp_params = cbp_outer_state.params["params"] # Assuming structure { 'params': {...} }
    cbp_state_internal = cbp_outer_state.cbp_state

    # Run a forward pass to get features
    # Use a consistent input for deterministic feature calculation if needed
    inputs = jnp.ones((1, 1)) # Consistent input
    _, features_intermediates = network.apply(
        cbp_outer_state.params, inputs, mutable=["intermediates"]
    )
    # Adjust path based on actual TestNet intermediate collection
    features = features_intermediates["intermediates"]["activations"][0]

    # Process parameters
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
            cbp.get_reset_mask, # Reference the actual function from cbp module
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            replacement_rate=replacement_rate,
        ),
        out_w_mag,
        utilities,
        ages,
        features, # Use potentially flattened features if required by get_reset_mask
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
        lambda a: jnp.full_like(a, cbp_state.maturity_threshold + 10),
        cbp_state.ages
    )
    low_maturity_threshold = 1 # Ensure threshold doesn't interfere

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
    assert all_means_correct, f"Mean check failed for rate {test_rate}. Means: {mask_means}"

    # Explicit checks for edge cases
    if jnp.isclose(test_rate, 0.0):
        assert tu.tree_all(tu.tree_map(jnp.all, tu.tree_map(jnp.logical_not, mask))), \
            f"Mask not all False for rate 0.0: {mask}"
    elif jnp.isclose(test_rate, 1.0):
        assert tu.tree_all(tu.tree_map(jnp.all, mask)), \
            f"Mask not all True for rate 1.0: {mask}"


def test_mask_ages(processed_data):
    """Tests that the mask respects the maturity_threshold."""
    setup = processed_data
    cbp_state = setup["cbp_state"]
    maturity_threshold = cbp_state.maturity_threshold # Use threshold from state

    # Helper to create specific age structures
    def create_ages(value):
        return jax.tree.map(lambda a: jnp.full_like(a, value), cbp_state.ages)

    # --- Case 1: All ages immature ---
    immature_ages = create_ages(maturity_threshold - 1)

    # Mask with high replacement rate (should still be all False due to age)
    mask_immature_r1 = calculate_mask(
        out_w_mag=setup["out_w_mag"], utilities=cbp_state.utilities, ages=immature_ages,
        features=setup["features"], decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold, replacement_rate=1.0,
    )
    # Mask with moderate replacement rate
    mask_immature_r05 = calculate_mask(
        out_w_mag=setup["out_w_mag"], utilities=cbp_state.utilities, ages=immature_ages,
        features=setup["features"], decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold, replacement_rate=0.5,
    )

    assert tu.tree_all(tu.tree_map(lambda m: not jnp.any(m), mask_immature_r1)), \
        f"Mask not all False for immature ages (rate=1.0): {mask_immature_r1}"
    assert tu.tree_all(tu.tree_map(lambda m: not jnp.any(m), mask_immature_r05)), \
        f"Mask not all False for immature ages (rate=0.5): {mask_immature_r05}"


    # --- Case 2: All ages mature ---
    mature_ages = create_ages(maturity_threshold + 1)

    # Mask with high replacement rate (should be all True)
    mask_mature_r1 = calculate_mask(
        out_w_mag=setup["out_w_mag"], utilities=cbp_state.utilities, ages=mature_ages,
        features=setup["features"], decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold, replacement_rate=1.0,
    )
    # Mask with moderate replacement rate (should have mean 0.5)
    mask_mature_r05 = calculate_mask(
        out_w_mag=setup["out_w_mag"], utilities=cbp_state.utilities, ages=mature_ages,
        features=setup["features"], decay_rate=cbp_state.decay_rate,
        maturity_threshold=maturity_threshold, replacement_rate=0.5,
    )

    assert tu.tree_all(tu.tree_map(jnp.all, mask_mature_r1)), \
        f"Mask not all True for mature ages (rate=1.0): {mask_mature_r1}"

    mask_means_r05 = tu.tree_map(lambda m: jnp.mean(m.astype(jnp.float32)), mask_mature_r05)
    all_means_correct_r05 = tu.tree_all(
        tu.tree_map(lambda m: jnp.isclose(m, 0.5), mask_means_r05)
    )
    assert all_means_correct_r05, \
        f"Mean check failed for mature ages (rate=0.5). Means: {mask_means_r05}"


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
    activations = feats["intermediates"]["activations"][0] # Adjust path if needed

    # Assert expected activations based on TestNet architecture (1 -> 4 -> 4 -> 4 -> 1) and ReLU
    # Assumes Dense layers with bias. Calculation: ReLU(dot(input, W) + b)
    # Layer 1: input=[[1]], W=[1,4]=ones, b=[4]=ones -> dot(1,1)+1=2 -> ReLU(2)=2
    assert jnp.allclose(activations["dense1"], 2.0), "First layer activations check failed"
    # Layer 2: input=[[2,2,2,2]], W=[4,4]=ones, b=[4]=ones -> dot([2,2,2,2], ones(4,1))+1 = 2*4+1=9 -> ReLU(9)=9
    assert jnp.allclose(activations["dense2"], 9.0), "Second layer activations check failed"
    # Layer 3: input=[[9,9,9,9]], W=[4,4]=ones, b=[4]=ones -> 9*4+1=37 -> ReLU(37)=37
    assert jnp.allclose(activations["dense3"], 37.0), "Third layer activations check failed"
    # Output Layer: input=[[37,37,37,37]], W=[4,1]=ones, b=[1]=ones -> 37*4+1=149 (No ReLU assumed on output)
    assert jnp.allclose(pred, 149.0), "Final prediction check failed"


def test_reset_weights_placeholder():
    """Placeholder for testing the weight resetting logic."""
    # TODO: Implement detailed tests for cbp.reset_weights
    # - Check that the correct number of weights are reset based on mask
    # - Check that input weights (columns) and output weights (rows) corresponding
    #   to reset neurons are handled correctly (set to initial/zeroed).
    # - Check bias resetting.
    pytest.skip("Weight reset tests not yet implemented")
