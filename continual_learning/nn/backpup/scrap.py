import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training import train_state
from typing import Any, Dict, Tuple, List, Optional
from dataclasses import dataclass
import flax.linen as nn
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax._src.typing import Array
from typing import TypeVar, Generic
import numpy as np

PyTree = Any
Float = TypeVar('Float', bound=Array)
PRNGKey = jnp.ndarray
PRNGKeyArray = jnp.ndarray

UTIL_TYPES = ["contribution", "weight", "adaptation", "zero_contribution", 
              "adaptable_contribution", "feature_by_input"]

@dataclass
class CBPOptimState:
    # Things you shouldn't really mess with
    utilities: Dict[str, Array]  # Utility values per layer
    ages: Dict[str, Array]       # Age counters per layer
    accumulated_features_to_replace: Dict[str, float]  # Accumulated fractional replacements

    rng: PRNGKeyArray  # Random key for initialization
    
    # Hyperparameters
    step_size: float = 0.001
    replacement_rate: float = 0.001
    decay_rate: float = 0.9
    maturity_threshold: int = 100
    accumulate: bool = False
    util_type: str = "contribution"

def get_layer_bounds(params):
    """Calculate initialization bounds for each layer."""
    bounds = {}
    
    for layer_name, layer_params in params.items():
        if 'kernel' in layer_params:
            kernel = layer_params['kernel']
            # For Dense layers, bound is based on input dimension
            if len(kernel.shape) == 2:
                in_features = kernel.shape[0]
                bound = jnp.sqrt(1.0 / in_features)
            # For Conv layers
            elif len(kernel.shape) == 4:
                in_channels = kernel.shape[2]
                kernel_size = kernel.shape[0] * kernel.shape[1]
                bound = jnp.sqrt(1.0 / (in_channels * kernel_size))
            else:
                bound = 0.01  # Default fallback
                
            bounds[layer_name] = bound
    
    return bounds

def get_bottom_k_mask(values, n_to_replace):
    """Create a boolean mask for the bottom k elements by value."""
    # Get array size
    size = values.size
    
    # Create positions for tie-breaking
    positions = jnp.arange(size)
    
    # Compute ranks (smaller values → smaller ranks)
    # Double argsort trick to get ranks with tie-breaking
    eps = jnp.finfo(values.dtype).eps * 10.0  # Add small epsilon to avoid equal values
    ranks = jnp.argsort(jnp.argsort(values + positions * eps))
    
    # Create mask for values with rank < n_to_replace
    mask = ranks < n_to_replace
    
    return mask

def gen_key_tree(key, tree):
    """Creates a PyTree of random keys matching the structure of the input tree."""
    leaves, treedef = jax.tree_flatten(tree)
    subkeys = jax.random.split(key, len(leaves))
    return jax.tree_unflatten(treedef, subkeys)

def find_output_layer_for_input(input_layer, params):
    """Find the likely output layer for a given input layer based on shapes."""
    input_shape = params[input_layer]['kernel'].shape
    output_units = input_shape[1] if len(input_shape) == 2 else input_shape[3]
    
    # Find layers that have the input dimension matching our output units
    candidate_layers = []
    for layer_name, layer_params in params.items():
        if layer_name != input_layer and 'kernel' in layer_params:
            out_weight_shape = layer_params['kernel'].shape
            # For dense layers, input dimension is first axis
            if len(out_weight_shape) == 2 and out_weight_shape[0] == output_units:
                candidate_layers.append(layer_name)
    
    # Return the first matching layer or None
    return candidate_layers[0] if candidate_layers else None

def map_layers(params):
    """Create a mapping of input layers to their likely output layers."""
    layer_map = {}
    
    # Find potential input layers (exclude the output layer)
    input_layers = [name for name in params if name != 'out_layer' and 'kernel' in params[name]]
    
    for layer in input_layers:
        output_layer = find_output_layer_for_input(layer, params)
        if output_layer:
            layer_map[layer] = output_layer
    
    return layer_map

def continual_backprop(
    util_type: str = "contribution", 
    decay_rate: float = 0.9,
    replacement_rate: float = 0.001,
    maturity_threshold: int = 100,
    step_size: float = 0.001,
) -> optax.GradientTransformation:
    """
    Continual Backpropagation optimizer.
    
    Args:
        util_type: How to calculate unit utility ("contribution", "weight", etc.)
        decay_rate: EMA decay rate for feature tracking
        replacement_rate: Fraction of mature units to replace
        maturity_threshold: Minimum age before a unit can be replaced
        step_size: Learning rate (not used directly)
        
    Returns:
        An optax.GradientTransformation for use with JAX optimizers
    """
    assert util_type in UTIL_TYPES, ValueError(
        f"Invalid util type, select from ({'|'.join(UTIL_TYPES)})"
    )
    
    def init(params: optax.Params):
        """Initialize the optimizer state."""
        # Extract parameters we want to track
        params_dict = params["params"]
        
        # Initialize utilities and ages for each layer with activations
        utilities = {}
        ages = {}
        accumulated = {}
        
        for layer_name, layer_params in params_dict.items():
            if 'kernel' in layer_params:
                # Get number of units (output dimension)
                kernel_shape = layer_params['kernel'].shape
                if len(kernel_shape) == 2:  # Dense layer
                    n_units = kernel_shape[1]
                elif len(kernel_shape) == 4:  # Conv layer
                    n_units = kernel_shape[3]
                else:
                    continue  # Skip layers we don't understand
                
                # Initialize tracking for this layer
                utilities[layer_name] = jnp.ones(n_units)
                ages[layer_name] = jnp.zeros(n_units)
                accumulated[layer_name] = 0.0
        
        return CBPOptimState(
            utilities=utilities,
            ages=ages,
            accumulated_features_to_replace=accumulated,
            rng=random.PRNGKey(0),
            decay_rate=decay_rate,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            step_size=step_size,
            util_type=util_type
        )
    
    def update(
        updates: optax.Updates,  # Gradients
        state: CBPOptimState,
        params: optax.Params | None = None,
        features: Dict | None = None,
    ) -> tuple[optax.Updates, CBPOptimState]:
        """Update parameters and perform selective reinitialization."""
        if params is None or features is None:
            return updates, state
        
        # Extract the params we'll work with
        params_dict = params["params"]
        
        # Create a layer mapping if not already available
        layer_map = map_layers(params_dict)
        
        # Layer bounds for reinitialization
        bounds = get_layer_bounds(params_dict)
        
        # Get activations from features
        activations = features.get("intermediates", {}).get("activations", {})
        
        # Create new random key
        new_key, rng = random.split(state.rng)
        
        # Initialize new dictionaries for updated state
        new_utilities = dict(state.utilities)
        new_ages = {k: v + 1 for k, v in state.ages.items()}  # Increment all ages
        new_accumulated = dict(state.accumulated_features_to_replace)
        
        # Create a dictionary for reinited parameters
        reinit_params = {}
        
        # Process each layer
        for layer_name, layer_activations in activations.items():
            # Skip if not in our tracking
            if layer_name not in state.utilities or layer_name not in layer_map:
                continue
                
            # Get output layer
            output_layer = layer_map[layer_name]
            
            # Calculate utilities
            input_weights = params_dict[layer_name]['kernel']
            output_weights = params_dict[output_layer]['kernel']
            
            # Get activation features for this layer
            features_abs = jnp.abs(layer_activations)
            
            # For dense layers
            if len(input_weights.shape) == 2:
                # Calculate output weight magnitude (equivalent to PyTorch output_weight_mag)
                output_weight_mag = jnp.abs(output_weights).mean(axis=0)
                
                # Calculate utility similar to PyTorch: output_weight_mag * features.abs().mean()
                utility = output_weight_mag * jnp.mean(features_abs, axis=0)
            else:
                # Simplified handling for non-dense layers
                utility = jnp.mean(features_abs, axis=tuple(range(features_abs.ndim - 1)))
            
            # Update the EMA of utility
            new_utilities[layer_name] = (
                state.decay_rate * state.utilities[layer_name] + 
                (1 - state.decay_rate) * utility
            )
            
            # Find eligible units (mature enough)
            ages = new_ages[layer_name]
            eligible_mask = ages > state.maturity_threshold
            n_eligible = jnp.sum(eligible_mask)
            
            # Calculate number of units to replace
            n_to_replace_float = n_eligible * state.replacement_rate
            n_to_replace = jnp.floor(n_to_replace_float).astype(jnp.int32)
            
            # Update accumulated replacements
            accumulated = new_accumulated[layer_name] + (n_to_replace_float - n_to_replace)
            if accumulated >= 1.0:
                n_to_replace += 1
                accumulated -= 1.0
            new_accumulated[layer_name] = accumulated
            
            # Create mask for units to replace
            if n_to_replace > 0 and n_eligible > 0:
                # Get utility for eligible units
                masked_utility = jnp.where(eligible_mask, new_utilities[layer_name], jnp.inf)
                
                # Find bottom k elements
                replace_mask = get_bottom_k_mask(masked_utility, n_to_replace)
                
                # Get initialization bound
                bound = bounds.get(layer_name, 0.01)
                
                # Generate a random key for this layer
                layer_key, rng = random.split(rng)
                
                # Get current parameters
                layer_w = params_dict[layer_name]['kernel']
                layer_b = params_dict[layer_name].get('bias', None)
                
                # Handle dense layers
                if len(layer_w.shape) == 2:
                    # Reinitialize input weights for selected units
                    new_w = jnp.where(
                        replace_mask[None, :],  # Broadcast across input dimension
                        random.uniform(layer_key, (layer_w.shape[0], jnp.sum(replace_mask)), 
                                       minval=-bound, maxval=bound),
                        layer_w
                    )
                    
                    # Reset bias if present
                    if layer_b is not None:
                        new_b = jnp.where(replace_mask, 0.0, layer_b)
                    else:
                        new_b = None
                    
                    # Zero out corresponding weights in the output layer
                    output_w = params_dict[output_layer]['kernel']
                    new_output_w = jnp.where(
                        replace_mask[:, None],  # Broadcast across output dimension
                        0.0,
                        output_w
                    )
                    
                    # Store reinited parameters
                    reinit_params[layer_name] = {
                        'kernel': new_w,
                        'bias': new_b if new_b is not None else layer_b
                    }
                    reinit_params[output_layer] = {
                        'kernel': new_output_w,
                        'bias': params_dict[output_layer].get('bias', None)
                    }
                    
                    # Reset ages for reinited units
                    new_ages[layer_name] = jnp.where(replace_mask, 0, ages)
        
        # Create new state
        new_state = CBPOptimState(
            utilities=new_utilities,
            ages=new_ages,
            accumulated_features_to_replace=new_accumulated,
            rng=rng,
            decay_rate=state.decay_rate,
            replacement_rate=state.replacement_rate,
            maturity_threshold=state.maturity_threshold,
            step_size=state.step_size,
            util_type=state.util_type
        )
        
        # Apply reinitialization to updates
        for layer_name, layer_params in reinit_params.items():
            for param_name, param_value in layer_params.items():
                if param_name in params_dict[layer_name]:
                    # Adjust the update to achieve the desired parameter value
                    param_path = (layer_name, param_name)
                    current_param = params_dict[layer_name][param_name]
                    desired_param = param_value
                    
                    # The update should be: desired - current
                    update_path = ('params', *param_path)
                    current_update = updates
                    for p in update_path[:-1]:
                        current_update = current_update[p]
                    
                    # Set the update to achieve the desired value
                    final_update = desired_param - current_param
                    
                    # We need to reach into the nested update structure
                    update_dict = updates
                    for p in update_path[:-1]:
                        update_dict = update_dict[p]
                    update_dict[update_path[-1]] = final_update
        
        return updates, new_state
    
    return optax.GradientTransformation(init=init, update=update)

class CBPTrainState(train_state.TrainState):
    """TrainState for Continual Backpropagation."""
    def apply_gradients(self, *, grads, features, **kwargs):
        """Apply gradients with features for CBP to use."""
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, params=self.params, features=features
        )
        
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


# FFNN class from provided code
class FFNN(nn.Module):
    """Simple Flax neural network"""

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        layers = ["dense1", "dense2", "dense3"] 

        for i in range(len(layers)):
            x = nn.Dense(features=128, name=layers[i])(x)
            x = nn.relu(x)
            intermediates[layers[i]] = x

        self.sow('intermediates', 'activations', intermediates) 

        # Output layer
        x = nn.Dense(features=10, name="out_layer")(x)

        return x

    # @jax.jit
    def predict(self, params, x):
        return self.apply({"params": params}, x, capture_intermediates=True)


# Testing function
def test_optim():
    print("Testing Continual Backpropagation optimizer...")
    
    # Initialize random key
    key = random.PRNGKey(0)

    # Create dummy data (batch_size=2, input_dim=784 for MNIST)
    batch_size = 1
    input_dim = 784
    dummy_input = random.normal(key, (batch_size, input_dim))
    dummy_labels = jax.nn.one_hot(random.randint(key, (batch_size,), 0, 10), 10)

    net_custom = FFNN()
    params = net_custom.init(key, dummy_input)

    # Set up CBP optimizer with appropriate hyperparameters
    tx = optax.chain(
        continual_backprop(
            util_type="contribution",
            decay_rate=0.9,
            replacement_rate=0.001,
            maturity_threshold=100,
            step_size=0.01
        ),
    )
    tx_adam = optax.adam(1e-3)  # Comparison optimizer

    # Create train states
    net_ts = CBPTrainState.create(apply_fn=net_custom.predict, params=params, tx=tx)
    net_ts_adam = train_state.TrainState.create(apply_fn=net_custom.predict, params=params, tx=tx_adam)
    
    # Print model structure
    print("Network structure:")
    for name, param in params["params"].items():
        if "kernel" in param:
            print(f"{name}: {param['kernel'].shape}")

    # Define loss function
    def loss_fn(params, inputs, labels):
        logits, features = net_custom.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
        return loss, (logits, features)

    # Create JIT-compiled versions for speed
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    # Save initial weights
    initial_weights = net_ts.params["params"]["dense1"]["kernel"]

    # Training loop
    num_steps = 100
    print(f"Training for {num_steps} steps...")
    
    losses = []
    
    for i in range(num_steps):
        # Generate new random data for each step
        key, subkey = random.split(key)
        inputs = random.normal(subkey, (batch_size, input_dim))
        labels = jax.nn.one_hot(random.randint(subkey, (batch_size,), 0, 10), 10)
        
        # Compute gradients
        (loss, (logits, features)), grads = grad_fn(net_ts.params, inputs, labels)
        losses.append(loss)
        
        # Update parameters
        net_ts = net_ts.apply_gradients(grads=grads, features=features)
        net_ts_adam = net_ts_adam.apply_gradients(grads=grads)
        
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
    
    # Compare final weights
    updated_weights = net_ts.params["params"]["dense1"]["kernel"]
    updated_weights_adam = net_ts_adam.params["params"]["dense1"]["kernel"]

    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Print weight statistics
    weight_diff = jnp.abs(updated_weights - initial_weights)
    weight_diff_adam = jnp.abs(updated_weights_adam - initial_weights)

    print("\nCBP Weight Changes:")
    print(f"Max weight change: {weight_diff.max():.4f}")
    print(f"Mean weight change: {weight_diff.mean():.4f}")
    print(f"Mean weight value: {updated_weights.mean():.4f}")
    print(f"Number of weights > 0: {(updated_weights > 0).sum()} out of {updated_weights.size}")

    print("\nAdam Weight Changes:")
    print(f"Max weight change: {weight_diff_adam.max():.4f}")
    print(f"Mean weight change: {weight_diff_adam.mean():.4f}")
    print(f"Mean weight value: {updated_weights_adam.mean():.4f}")
    print(f"Number of weights > 0: {(updated_weights_adam > 0).sum()} out of {updated_weights_adam.size}")
    
    # Check if any reinitialization happened
    ages = net_ts.opt_state[0].ages
    utilities = net_ts.opt_state[0].utilities
    
    print("\nCBP State:")
    for layer, layer_ages in ages.items():
        num_reinits = (layer_ages == 0).sum()
        print(f"Layer {layer}: {num_reinits} units reinitialized")
        print(f"  Min utility: {utilities[layer].min():.4f}, Max utility: {utilities[layer].max():.4f}")
    
    return net_ts, losses

if __name__ == "__main__":
    model, losses = test_optim()
