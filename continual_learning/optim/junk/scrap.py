from flax import struct
from flax.core import FrozenDict
from jax.random import PRNGKey
from jaxtyping import (
    Array,
    Float,
    Bool,
    PRNGKeyArray,
    PyTree,
    Int,
)
from flax.training.train_state import TrainState
from typing import Tuple, Dict
from chex import dataclass
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from functools import partial


@dataclass
class RedoOptimState:
    initial_weights: PyTree[Float[Array, "..."]]
    rng: PRNGKeyArray
    time_step: int = 0
    update_frequency: int = 1000  # Paper default
    threshold: float = 0.1  # τ-dormant threshold from paper


# -------------- Overall optimizer TrainState ---------------
class RedoTrainState(TrainState):
    redo_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        redo_state = redo().init(params, **kwargs)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            redo_state=redo_state,
        )

    def apply_gradients(self, *, grads, features, **kwargs):
        # Apply base optimizer updates first
        tx_updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params
        )
        params_after_tx = optax.apply_updates(self.params, tx_updates)

        # Apply ReDo updates
        params_after_redo, new_redo_state = redo().update(
            grads["params"],
            self.redo_state,
            params_after_tx["params"],
            features=features,  # Pass all layer features
        )

        return self.replace(
            step=self.step + 1,
            params={"params": params_after_redo},
            opt_state=new_opt_state,
            redo_state=new_redo_state[0],
            **kwargs,
        )


# -------------- Neuron score calculation ---------------
def calculate_neuron_score(
    features: Float[Array, "batch neurons"],
) -> Float[Array, "neurons"]:
    """Calculate normalized neuron scores according to paper Definition 3.1"""
    # Average absolute activation across batch
    mean_act_per_neuron = jnp.mean(jnp.abs(features), axis=0)
    
    # Normalize so scores sum to 1 within layer
    sum_activations = jnp.sum(mean_act_per_neuron)
    score = mean_act_per_neuron / (sum_activations + 1e-9)
    
    return score


# -------------- Get dormant mask ---------------
def get_dormant_mask(
    scores: Float[Array, "neurons"],
    threshold: Float[Array, ""] = 0.1,
) -> Bool[Array, "neurons"]:
    """Identify τ-dormant neurons"""
    return scores <= threshold


# -------------- Reset weights ---------------
def reset_dormant_neurons(
    layer_params: Dict,
    dormant_masks: Dict,
    initial_weights: Dict,
    rng_keys: Dict,
):
    """Reset incoming and outgoing weights for dormant neurons"""
    layer_names = list(layer_params.keys())
    new_params = {}
    logs = {"nodes_reset": 0}
    
    for i, layer_name in enumerate(layer_names):
        layer = layer_params[layer_name]
        
        # Skip if no kernel/bias
        if not isinstance(layer, dict) or "kernel" not in layer:
            new_params[layer_name] = layer
            continue
            
        # Get dormant mask for this layer
        if layer_name not in dormant_masks:
            new_params[layer_name] = layer
            continue
            
        dormant_mask = dormant_masks[layer_name]
        n_dormant = jnp.sum(dormant_mask)
        logs["nodes_reset"] += n_dormant
        
        # Reset bias to zero for dormant neurons
        new_bias = jnp.where(
            dormant_mask,
            jnp.zeros_like(layer["bias"]),
            layer["bias"]
        )
        
        # Reinitialize incoming weights (connections TO dormant neurons)
        # For Dense layers: kernel shape is (in_features, out_features)
        # We want to reinitialize columns corresponding to dormant neurons
        kernel = layer["kernel"]
        if kernel.ndim == 2:  # Dense layer
            dormant_mask_expanded = dormant_mask.reshape(1, -1)
            
            # Reinitialize incoming weights
            if layer_name in initial_weights:
                reinitialized = initial_weights[layer_name]["kernel"]
            else:
                # Generate new random weights with same distribution
                key = rng_keys.get(layer_name)
                if key is not None:
                    reinitialized = jax.nn.initializers.xavier_uniform()(
                        key, shape=kernel.shape
                    )
                else:
                    reinitialized = kernel
                    
            new_kernel = jnp.where(
                dormant_mask_expanded,
                reinitialized,
                kernel
            )
        else:
            # For conv layers, would need different handling
            new_kernel = kernel
            
        new_params[layer_name] = {
            "kernel": new_kernel,
            "bias": new_bias
        }
        
        # Zero out outgoing weights (connections FROM dormant neurons)
        # This affects the NEXT layer's incoming weights
        if i + 1 < len(layer_names):
            next_layer_name = layer_names[i + 1]
            next_layer = layer_params[next_layer_name]
            
            if isinstance(next_layer, dict) and "kernel" in next_layer:
                next_kernel = next_layer["kernel"]
                if next_kernel.ndim == 2:  # Dense layer
                    # Zero out rows corresponding to dormant neurons
                    dormant_mask_expanded = dormant_mask.reshape(-1, 1)
                    new_next_kernel = jnp.where(
                        dormant_mask_expanded,
                        jnp.zeros_like(next_kernel),
                        next_kernel
                    )
                    
                    if next_layer_name not in new_params:
                        new_params[next_layer_name] = dict(next_layer)
                    new_params[next_layer_name]["kernel"] = new_next_kernel
    
    return new_params, logs


# -------------- Main ReDo Optimizer ---------------
def redo(**kwargs) -> optax.GradientTransformation:
    def init(params: optax.Params, **kwargs):
        # Extract and store initial weights
        initial_weights = {}
        
        def extract_weights(path, value):
            if isinstance(value, dict) and "kernel" in value:
                initial_weights["/".join(path)] = {
                    "kernel": jnp.array(value["kernel"]),
                    "bias": jnp.array(value["bias"])
                }
        
        # Walk the parameter tree
        def walk_tree(d, path=()):
            for k, v in d.items():
                new_path = path + (k,)
                if isinstance(v, dict):
                    walk_tree(v, new_path)
                extract_weights(new_path, v)
        
        if "params" in params:
            walk_tree(params["params"])
        
        return RedoOptimState(
            initial_weights=initial_weights,
            **kwargs,
        )

    @jax.jit
    def update(
        updates: optax.Updates,
        state: RedoOptimState,
        params: optax.Params | None = None,
        features: Dict | None = None,
    ) -> tuple[optax.Updates, RedoOptimState]:
        def no_update(updates):
            new_state = state.replace(time_step=state.time_step + 1)
            return params, (new_state,)

        def _redo(updates) -> Tuple[optax.Updates, RedoOptimState]:
            # Extract layer features and calculate scores
            dormant_masks = {}
            
            if features is not None and "intermediates" in features:
                activations = features["intermediates"]["activations"]
                
                # Calculate scores for each layer
                for layer_idx, layer_features in enumerate(activations):
                    # Ensure we have batch dimension
                    if layer_features.ndim == 2:
                        scores = calculate_neuron_score(layer_features)
                        mask = get_dormant_mask(scores, state.threshold)
                        
                        # Map to parameter names (this is simplified)
                        # In practice, you'd need proper layer name mapping
                        layer_name = f"layer_{layer_idx}"
                        dormant_masks[layer_name] = mask

            # Generate random keys for reinitialization
            new_rng, *subkeys = random.split(state.rng, len(dormant_masks) + 1)
            rng_keys = {name: key for name, key in zip(dormant_masks.keys(), subkeys)}
            
            # Reset weights
            new_params, logs = reset_dormant_neurons(
                params,
                dormant_masks,
                state.initial_weights,
                rng_keys
            )
            
            new_state = state.replace(
                rng=new_rng,
                time_step=state.time_step + 1
            )
            
            return new_params, (new_state,)

        return jax.lax.cond(
            state.time_step % state.update_frequency == 0,
            _redo,
            no_update,
            updates
        )

    return optax.GradientTransformation(init=init, update=update)
