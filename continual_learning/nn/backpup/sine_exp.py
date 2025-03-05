import flax.linen as nn
import jax
import os
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training.train_state import TrainState
import numpy as np
import time
from typing import Dict, Any, Tuple, List
import tyro

from dataclasses import dataclass
import tyro
from typing import Optional


# Import the continual backprop optimizer
from continual_learning.optim.continual_backprop import (
    continual_backprop,
    CBPTrainState,
)
import continual_learning.nn.utils as utils


class SineNet(nn.Module):
    """Simple Flax neural network for sine wave regression"""

    @nn.compact
    def __call__(self, x):
        intermediates = {}

        layers = [
            "dense1",
            "dense2",
            "dense3",
        ]

        for i, layer_name in enumerate(layers):
            x = nn.Dense(features=128, name=layer_name)(x)
            x = nn.relu(x)
            intermediates[layer_name] = x

        self.sow("intermediates", "activations", intermediates)

        # Single output for regression
        x = nn.Dense(features=1, name="out_layer")(x)

        return x

    @jax.jit
    def predict(self, params, x):
        return self.apply({"params": params}, x, capture_intermediates=True)


def generate_sine_data(
    key, batch_size, phase_shift=0.0, amplitude=1.0, noise_level=0.0
):
    """Generate a batch of sine wave data with a given phase shift."""
    key1, key2 = random.split(key)
    x = random.uniform(key1, (batch_size, 1)) * 2 * jnp.pi  # Input between 0 and 2π
    noise = noise_level * random.normal(key2, (batch_size, 1))
    y = amplitude * jnp.sin(x + phase_shift) + noise
    return x, y





def continual_sine_learning(
    num_phase_shifts=20000,  # Total number of phase shifts to perform
    epochs_per_phase=100,  # Epochs to train on each phase
    batch_size=64,  # Batch size for training
    learning_rate=1e-3,  # Learning rate for optimizer
    weight_decay=0.01,  # Weight decay for AdamW
    phase_shift_step=0.1,  # Amount to shift the phase by each time
    eval_interval=100,  # How often to evaluate forgetting on previous tasks
    save_interval=1000,  # How often to save metrics
    verbose=True,  # Whether to print progress
):
    """
    Run a continual learning experiment with a sine wave regression task.

    The task involves learning a sine wave, then gradually shifting the phase and
    continuing to learn, measuring how well the model adapts and whether it forgets
    previously learned phases.
    """
    if verbose:
        print(
            f"Starting continual learning experiment with {num_phase_shifts} phase shifts"
        )
        print(f"Training for {epochs_per_phase} epochs per phase")

    # Initialize random key
    key = random.PRNGKey(0)

    # Initialize network
    key, init_key = random.split(key)
    net = SineNet()

    # Create dummy input for initialization
    dummy_input = jnp.zeros((1, 1))
    params = net.init(init_key, dummy_input)

    # Initialize optimizers
    adam_tx = optax.adam(learning_rate)
    cbp_adam_tx = optax.adam(learning_rate)
    adamw_tx = optax.adamw(learning_rate, weight_decay=weight_decay)

    # Create train states
    cbp_state = CBPTrainState.create(
        apply_fn=net.predict, params=params, tx=cbp_adam_tx
    )
    adam_state = TrainState.create(apply_fn=net.predict, params=params, tx=adam_tx)
    adamw_state = TrainState.create(apply_fn=net.predict, params=params, tx=adamw_tx)

    # Define loss function
    def loss_fn(params, inputs, targets):
        predictions, features = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)  # MSE loss
        return loss, (predictions, features)

    # Gradient function
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # JIT-compiled training steps
    @jax.jit
    def train_cbp_step(state, inputs, targets):
        (loss, (preds, features)), grads = grad_fn(state.params, inputs, targets)
        new_state = state.apply_gradients(grads=grads, features=features)
        return new_state, loss

    @jax.jit
    def train_adam_step(state, inputs, targets):
        (loss, (preds, features)), grads = grad_fn(state.params, inputs, targets)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    @jax.jit
    def train_adamw_step(state, inputs, targets):
        (loss, (preds, features)), grads = grad_fn(state.params, inputs, targets)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    @jax.jit
    def evaluate(params, inputs, targets):
        predictions, _ = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)
        return loss

    # Storage for metrics
    cbp_metrics = []
    adam_metrics = []
    adamw_metrics = []

    # Storage for best performance on each task
    cbp_best_losses = {}
    adam_best_losses = {}
    adamw_best_losses = {}

    # Current performance on each task
    cbp_current_losses = {}
    adam_current_losses = {}
    adamw_current_losses = {}

    # print weight sizes
    for name, param in cbp_state.params["params"].items():
        print(name, param["kernel"].shape)

    # Main training loop for each phase shift
    start_time = time.time()
    for shift_idx in range(num_phase_shifts):
        phase_shift = phase_shift_step * shift_idx

        # Save initial parameters for plasticity metrics
        cbp_initial_params = jax.tree_map(
            lambda x: x.copy(), cbp_state.params["params"]
        )
        adam_initial_params = jax.tree_map(
            lambda x: x.copy(), adam_state.params["params"]
        )
        adamw_initial_params = jax.tree_map(
            lambda x: x.copy(), adamw_state.params["params"]
        )

        # Track losses during training
        cbp_phase_losses = []
        adam_phase_losses = []
        adamw_phase_losses = []

        # Generate evaluation data for this phase
        key, eval_key = random.split(key)
        eval_inputs, eval_targets = generate_sine_data(eval_key, 1000, phase_shift)

        # Evaluate initial performance on this phase
        cbp_initial_loss = evaluate(cbp_state.params, eval_inputs, eval_targets)
        adam_initial_loss = evaluate(adam_state.params, eval_inputs, eval_targets)
        adamw_initial_loss = evaluate(adamw_state.params, eval_inputs, eval_targets)

        # Train for multiple epochs on this phase
        for epoch in range(epochs_per_phase):
            key, data_key = random.split(key)

            # Generate a batch of data for this phase
            inputs, targets = generate_sine_data(data_key, batch_size, phase_shift)

            # Perform training steps
            cbp_state, cbp_loss = train_cbp_step(cbp_state, inputs, targets)
            adam_state, adam_loss = train_adam_step(adam_state, inputs, targets)
            adamw_state, adamw_loss = train_adamw_step(adamw_state, inputs, targets)

            # Store losses
            cbp_phase_losses.append(float(cbp_loss))
            adam_phase_losses.append(float(adam_loss))
            adamw_phase_losses.append(float(adamw_loss))


        # Evaluate final performance on this phase
        cbp_final_loss = evaluate(cbp_state.params, eval_inputs, eval_targets)
        adam_final_loss = evaluate(adam_state.params, eval_inputs, eval_targets)
        adamw_final_loss = evaluate(adamw_state.params, eval_inputs, eval_targets)

        # Store best and current losses for this task
        task_id = float(phase_shift)
        cbp_best_losses[task_id] = min(
            cbp_best_losses.get(task_id, float("inf")), float(cbp_final_loss)
        )
        adam_best_losses[task_id] = min(
            adam_best_losses.get(task_id, float("inf")), float(adam_final_loss)
        )
        adamw_best_losses[task_id] = min(
            adamw_best_losses.get(task_id, float("inf")), float(adamw_final_loss)
        )

        cbp_current_losses[task_id] = float(cbp_final_loss)
        adam_current_losses[task_id] = float(adam_final_loss)
        adamw_current_losses[task_id] = float(adamw_final_loss)

        # Compute plasticity metrics
        cbp_plasticity = utils.compute_plasticity_metrics(
            cbp_initial_params, cbp_state.params["params"]
        )
        adam_plasticity = utils.compute_plasticity_metrics(
            adam_initial_params, adam_state.params["params"]
        )
        adamw_plasticity = utils.compute_plasticity_metrics(
            adamw_initial_params, adamw_state.params["params"]
        )

        # Compute adaptation metrics
        cbp_adaptation = float(cbp_initial_loss - cbp_final_loss)
        adam_adaptation = float(adam_initial_loss - adam_final_loss)
        adamw_adaptation = float(adamw_initial_loss - adamw_final_loss)

        # Initialize forgetting metrics
        cbp_forgetting = {"avg_forgetting": 0.0, "max_forgetting": 0.0}
        adam_forgetting = {"avg_forgetting": 0.0, "max_forgetting": 0.0}
        adamw_forgetting = {"avg_forgetting": 0.0, "max_forgetting": 0.0}

        # Evaluate forgetting on previous tasks periodically
        if shift_idx > 0 and (
            shift_idx % eval_interval == 0 or shift_idx == num_phase_shifts - 1
        ):
            # Sample previous phases to evaluate
            previous_task_ids = [
                phase_shift_step * i for i in range(0, shift_idx, eval_interval)
            ]

            # Evaluate on each previous task
            for prev_shift in previous_task_ids:
                key, prev_key = random.split(key)
                prev_inputs, prev_targets = generate_sine_data(
                    prev_key, 1000, prev_shift
                )

                # Evaluate on previous task
                cbp_prev_loss = evaluate(cbp_state.params, prev_inputs, prev_targets)
                adam_prev_loss = evaluate(adam_state.params, prev_inputs, prev_targets)
                adamw_prev_loss = evaluate(
                    adamw_state.params, prev_inputs, prev_targets
                )

                # Update current losses
                task_id = float(prev_shift)
                cbp_current_losses[task_id] = float(cbp_prev_loss)
                adam_current_losses[task_id] = float(adam_prev_loss)
                adamw_current_losses[task_id] = float(adamw_prev_loss)

            # Compute overall forgetting metrics
            cbp_forgetting = utils.compute_forgetting_metrics(
                cbp_current_losses, cbp_best_losses
            )
            adam_forgetting = utils.compute_forgetting_metrics(
                adam_current_losses, adam_best_losses
            )
            adamw_forgetting = utils.compute_forgetting_metrics(
                adamw_current_losses, adamw_best_losses
            )

        # Calculate stability-plasticity tradeoff
        cbp_tradeoff = utils.stability_plasticity_tradeoff(
            cbp_adaptation, cbp_forgetting["avg_forgetting"]
        )
        adam_tradeoff = utils.stability_plasticity_tradeoff(
            adam_adaptation, adam_forgetting["avg_forgetting"]
        )
        adamw_tradeoff = utils.stability_plasticity_tradeoff(
            adamw_adaptation, adamw_forgetting["avg_forgetting"]
        )

        # Store metrics for this phase
        phase_metrics = {
            "phase": shift_idx,
            "phase_shift": phase_shift,
            "cbp_initial_loss": float(cbp_initial_loss),
            "cbp_final_loss": float(cbp_final_loss),
            "adam_initial_loss": float(adam_initial_loss),
            "adam_final_loss": float(adam_final_loss),
            "adamw_initial_loss": float(adamw_initial_loss),
            "adamw_final_loss": float(adamw_final_loss),
            "cbp_plasticity": cbp_plasticity["total_plasticity"],
            "adam_plasticity": adam_plasticity["total_plasticity"],
            "adamw_plasticity": adamw_plasticity["total_plasticity"],
            "cbp_adaptation": cbp_adaptation,
            "adam_adaptation": adam_adaptation,
            "adamw_adaptation": adamw_adaptation,
            "cbp_forgetting": cbp_forgetting["avg_forgetting"],
            "adam_forgetting": adam_forgetting["avg_forgetting"],
            "adamw_forgetting": adamw_forgetting["avg_forgetting"],
            "cbp_tradeoff": cbp_tradeoff,
            "adam_tradeoff": adam_tradeoff,
            "adamw_tradeoff": adamw_tradeoff,
        }

        # Add to metrics lists for plotting
        cbp_metrics.append(
            {
                "phase": shift_idx,
                "phase_shift": phase_shift,
                "final_loss": float(cbp_final_loss),
                "plasticity": cbp_plasticity["total_plasticity"],
                "adaptation": cbp_adaptation,
                "forgetting": cbp_forgetting["avg_forgetting"],
                "tradeoff": cbp_tradeoff,
            }
        )

        adam_metrics.append(
            {
                "phase": shift_idx,
                "phase_shift": phase_shift,
                "final_loss": float(adam_final_loss),
                "plasticity": adam_plasticity["total_plasticity"],
                "adaptation": adam_adaptation,
                "forgetting": adam_forgetting["avg_forgetting"],
                "tradeoff": adam_tradeoff,
            }
        )

        adamw_metrics.append(
            {
                "phase": shift_idx,
                "phase_shift": phase_shift,
                "final_loss": float(adamw_final_loss),
                "plasticity": adamw_plasticity["total_plasticity"],
                "adaptation": adamw_adaptation,
                "forgetting": adamw_forgetting["avg_forgetting"],
                "tradeoff": adamw_tradeoff,
            }
        )

        # Print progress
        if verbose and (shift_idx % 10 == 0 or shift_idx == num_phase_shifts - 1):
            elapsed = time.time() - start_time
            print(":: Optimiser metrics ::")
            print(
                f"Phase {shift_idx}/{num_phase_shifts}, Shift: {phase_shift:.2f}, Time: {elapsed:.1f}s"
            )
            print(
                f"  CBP:    Loss: {cbp_final_loss:.6f}, Plasticity: {cbp_plasticity['total_plasticity']:.6f}"
            )
            print(
                f"  Adam:   Loss: {adam_final_loss:.6f}, Plasticity: {adam_plasticity['total_plasticity']:.6f}"
            )
            print(
                f"  AdamW:  Loss: {adamw_final_loss:.6f}, Plasticity: {adamw_plasticity['total_plasticity']:.6f}"
            )

            if shift_idx > 0 and shift_idx % eval_interval == 0:
                print(
                    f"  CBP Forgetting: {cbp_forgetting['avg_forgetting']:.6f}, Tradeoff: {cbp_tradeoff:.6f}"
                )
                print(
                    f"  Adam Forgetting: {adam_forgetting['avg_forgetting']:.6f}, Tradeoff: {adam_tradeoff:.6f}"
                )
                print(
                    f"  AdamW Forgetting: {adamw_forgetting['avg_forgetting']:.6f}, Tradeoff: {adamw_tradeoff:.6f}"
                )

            # Extra logs
            cbp_logs = cbp_state.cbp_state.logs # ["dense1", ..., "dense3"]
            first_value = next(iter(cbp_logs.values()))
            extra_logs = {k: 0 for k in first_value.keys()}  # initialise metrics

            for k, v in cbp_logs.items():
                extra_logs["nodes_reset"] += v["nodes_reset"] 
                extra_logs["n_mature"] += v["n_mature"]      
                extra_logs["avg_age"] += v["avg_age"] / len(cbp_logs)

            print(":: Extra Metrics ::")
            print("nodes reset", extra_logs["nodes_reset"])
            print("avg node age", jnp.mean(extra_logs["avg_age"]))
            print("n_mature", jnp.mean(extra_logs["n_mature"]))

            print("---")
        # Save and plot results periodically
        if shift_idx % save_interval == 0 or shift_idx == num_phase_shifts - 1:
            utils.plot_results(
                cbp_metrics, adam_metrics, adamw_metrics, f"results_{shift_idx}"
            )

    # Final analysis
    utils.print_summary_metrics(cbp_metrics, adam_metrics, adamw_metrics)

    return cbp_metrics, adam_metrics, adamw_metrics



@dataclass
class Args:
    """Arguments for continual sine learning experiment"""
    debug: bool = True
    num_phase_shifts: Optional[int] = None
    epochs_per_phase: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    phase_shift_step: Optional[float] = None
    eval_interval: Optional[int] = None
    save_interval: Optional[int] = None
    verbose: Optional[bool] = None

def main(args: Args):
    # Set default configuration based on debug mode
    config = {}
    
    if args.debug:
        config = {
            "num_phase_shifts": 50,  # Reduced number of shifts for testing
            "epochs_per_phase": 10,  # Fewer epochs per phase
            "batch_size": 1,
            "eval_interval": 5,
            "save_interval": 10,
            "verbose": True,
        }
    else:
        config = {
            "num_phase_shifts": 20000,  # Total number of phase shifts to perform
            "epochs_per_phase": 100,  # Epochs to train on each phase
            "batch_size": 1,  # Batch size for training
            "learning_rate": 1e-3,  # Learning rate for optimizer
            "weight_decay": 0.01,  # Weight decay for AdamW
            "phase_shift_step": 0.1,  # Amount to shift the phase by each time
            "eval_interval": 100,  # How often to evaluate forgetting on previous tasks
            "save_interval": 1000,  # How often to save metrics
            "verbose": True,  # Whether to print progress
        }
    
    # Override defaults with any explicitly provided CLI arguments
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key != "debug" and value is not None:
            config[key] = value
    
    # Run the experiment with the final configuration
    continual_sine_learning(**config)

if __name__ == "__main__":
    # Parse command line arguments using tyro and the Args dataclass
    args = tyro.cli(Args)
    main(args)
