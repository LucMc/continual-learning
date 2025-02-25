import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Dict, Any, Tuple, List

# Import the continual backprop optimizer
from continual_learning.optim.continual_backprop import (
    continual_backprop,
    CBPTrainState,
)


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


def compute_plasticity_metrics(old_params, new_params):
    """Compute metrics related to neural plasticity."""
    metrics = {}

    # Calculate weight changes for each layer
    total_abs_change = 0
    total_weights = 0
    layer_metrics = {}

    for layer_name, layer_params in old_params.items():
        if isinstance(layer_params, dict) and "kernel" in layer_params:
            old_weights = layer_params["kernel"]
            new_weights = new_params[layer_name]["kernel"]

            # Calculate changes
            abs_changes = jnp.abs(new_weights - old_weights)

            # Per-layer metrics
            layer_metrics[layer_name] = {
                "mean_change": float(abs_changes.mean()),
                "max_change": float(abs_changes.max()),
                "positive_changes": float(
                    (new_weights > old_weights).sum() / old_weights.size
                ),
                "negative_changes": float(
                    (new_weights < old_weights).sum() / old_weights.size
                ),
            }

            # Update totals
            total_abs_change += jnp.sum(abs_changes)
            total_weights += old_weights.size

    # Overall metrics
    metrics["total_plasticity"] = float(total_abs_change / max(total_weights, 1))
    metrics["layer_metrics"] = layer_metrics

    return metrics


def compute_forgetting_metrics(current_losses, best_losses):
    """
    Compute metrics related to catastrophic forgetting.

    Args:
        current_losses: Dictionary mapping task IDs to current losses
        best_losses: Dictionary mapping task IDs to best achieved losses
    """
    if not current_losses or not best_losses:
        return {"avg_forgetting": 0.0, "max_forgetting": 0.0}

    # Calculate forgetting for each task
    forgetting_by_task = {}
    total_forgetting = 0.0
    max_forgetting = 0.0

    for task_id, best_loss in best_losses.items():
        if task_id in current_losses:
            forgetting = max(
                0.0, current_losses[task_id] - best_loss
            )  # Only positive forgetting
            forgetting_by_task[task_id] = forgetting
            total_forgetting += forgetting
            max_forgetting = max(max_forgetting, forgetting)

    # Average forgetting
    avg_forgetting = total_forgetting / max(len(forgetting_by_task), 1)

    return {
        "avg_forgetting": float(avg_forgetting),
        "max_forgetting": float(max_forgetting),
        "forgetting_by_task": forgetting_by_task,
    }


def stability_plasticity_tradeoff(adaptation, forgetting):
    """Calculate the stability-plasticity tradeoff metric."""
    stability = 1.0 / (1.0 + forgetting)  # Higher forgetting means lower stability
    return (
        stability * adaptation
    )  # Good tradeoff means high adaptation with high stability


def continual_sine_learning(
    num_phase_shifts=20000,  # Total number of phase shifts to perform
    epochs_per_phase=100,  # Epochs to train on each phase
    batch_size=64,  # Batch size for training
    learning_rate=1e-3,  # Learning rate for optimizer
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
    cbp_tx = optax.chain(continual_backprop(), optax.adam(learning_rate))
    adam_tx = optax.adam(learning_rate)

    # Create train states
    cbp_state = CBPTrainState.create(apply_fn=net.predict, params=params, tx=cbp_tx)
    adam_state = TrainState.create(apply_fn=net.predict, params=params, tx=adam_tx)

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

    # JIT-compiled evaluation
    @jax.jit
    def evaluate(params, inputs, targets):
        predictions, _ = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)
        return loss

    # Storage for metrics
    cbp_metrics = []
    adam_metrics = []

    # Storage for best performance on each task
    cbp_best_losses = {}
    adam_best_losses = {}

    # Current performance on each task
    cbp_current_losses = {}
    adam_current_losses = {}

    # print weight sizes
    for name, param in cbp_state.params["params"].items(): print(name, param["kernel"].shape)

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

        # Track losses during training
        cbp_phase_losses = []
        adam_phase_losses = []

        # Generate evaluation data for this phase
        key, eval_key = random.split(key)
        eval_inputs, eval_targets = generate_sine_data(eval_key, 1000, phase_shift)

        # Evaluate initial performance on this phase
        cbp_initial_loss = evaluate(cbp_state.params, eval_inputs, eval_targets)
        adam_initial_loss = evaluate(adam_state.params, eval_inputs, eval_targets)

        # Train for multiple epochs on this phase
        for epoch in range(epochs_per_phase):
            key, data_key = random.split(key)

            # Generate a batch of data for this phase
            inputs, targets = generate_sine_data(data_key, batch_size, phase_shift)

            
            # Perform training steps
            cbp_state, cbp_loss = train_cbp_step(cbp_state, inputs, targets)
            adam_state, adam_loss = train_adam_step(adam_state, inputs, targets)

            # Store losses
            cbp_phase_losses.append(float(cbp_loss))
            adam_phase_losses.append(float(adam_loss))

        # Evaluate final performance on this phase
        cbp_final_loss = evaluate(cbp_state.params, eval_inputs, eval_targets)
        adam_final_loss = evaluate(adam_state.params, eval_inputs, eval_targets)

        # Store best and current losses for this task
        task_id = float(phase_shift)
        cbp_best_losses[task_id] = min(
            cbp_best_losses.get(task_id, float("inf")), float(cbp_final_loss)
        )
        adam_best_losses[task_id] = min(
            adam_best_losses.get(task_id, float("inf")), float(adam_final_loss)
        )

        cbp_current_losses[task_id] = float(cbp_final_loss)
        adam_current_losses[task_id] = float(adam_final_loss)

        # Compute plasticity metrics
        cbp_plasticity = compute_plasticity_metrics(
            cbp_initial_params, cbp_state.params["params"]
        )
        adam_plasticity = compute_plasticity_metrics(
            adam_initial_params, adam_state.params["params"]
        )

        # Compute adaptation metrics
        cbp_adaptation = float(cbp_initial_loss - cbp_final_loss)
        adam_adaptation = float(adam_initial_loss - adam_final_loss)

        # Initialize forgetting metrics
        cbp_forgetting = {"avg_forgetting": 0.0, "max_forgetting": 0.0}
        adam_forgetting = {"avg_forgetting": 0.0, "max_forgetting": 0.0}

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

                # Update current losses
                task_id = float(prev_shift)
                cbp_current_losses[task_id] = float(cbp_prev_loss)
                adam_current_losses[task_id] = float(adam_prev_loss)

            # Compute overall forgetting metrics
            cbp_forgetting = compute_forgetting_metrics(
                cbp_current_losses, cbp_best_losses
            )
            adam_forgetting = compute_forgetting_metrics(
                adam_current_losses, adam_best_losses
            )

        # Calculate stability-plasticity tradeoff
        cbp_tradeoff = stability_plasticity_tradeoff(
            cbp_adaptation, cbp_forgetting["avg_forgetting"]
        )
        adam_tradeoff = stability_plasticity_tradeoff(
            adam_adaptation, adam_forgetting["avg_forgetting"]
        )

        # Store metrics for this phase
        phase_metrics = {
            "phase": shift_idx,
            "phase_shift": phase_shift,
            "cbp_initial_loss": float(cbp_initial_loss),
            "cbp_final_loss": float(cbp_final_loss),
            "adam_initial_loss": float(adam_initial_loss),
            "adam_final_loss": float(adam_final_loss),
            "cbp_plasticity": cbp_plasticity["total_plasticity"],
            "adam_plasticity": adam_plasticity["total_plasticity"],
            "cbp_adaptation": cbp_adaptation,
            "adam_adaptation": adam_adaptation,
            "cbp_forgetting": cbp_forgetting["avg_forgetting"],
            "adam_forgetting": adam_forgetting["avg_forgetting"],
            "cbp_tradeoff": cbp_tradeoff,
            "adam_tradeoff": adam_tradeoff,
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

        # Print progress
        if verbose and (shift_idx % 10 == 0 or shift_idx == num_phase_shifts - 1):
            elapsed = time.time() - start_time
            print(
                f"Phase {shift_idx}/{num_phase_shifts}, Shift: {phase_shift:.2f}, Time: {elapsed:.1f}s"
            )
            print(
                f"  CBP:  Loss: {cbp_final_loss:.6f}, Plasticity: {cbp_plasticity['total_plasticity']:.6f}"
            )
            print(
                f"  Adam: Loss: {adam_final_loss:.6f}, Plasticity: {adam_plasticity['total_plasticity']:.6f}"
            )

            if shift_idx > 0 and shift_idx % eval_interval == 0:
                print(
                    f"  CBP Forgetting: {cbp_forgetting['avg_forgetting']:.6f}, Tradeoff: {cbp_tradeoff:.6f}"
                )
                print(
                    f"  Adam Forgetting: {adam_forgetting['avg_forgetting']:.6f}, Tradeoff: {adam_tradeoff:.6f}"
                )

            print("---")

        # Save and plot results periodically
        if shift_idx % save_interval == 0 or shift_idx == num_phase_shifts - 1:
            plot_results(cbp_metrics, adam_metrics, f"results_{shift_idx}")

    # Final analysis
    print_summary_metrics(cbp_metrics, adam_metrics)

    return cbp_metrics, adam_metrics


def plot_results(cbp_metrics, adam_metrics, filename_prefix="results"):
    """Plot metrics from the continual learning experiment."""
    # Extract data for plotting
    phases = np.array([m["phase"] for m in cbp_metrics])

    # Create multi-panel figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Plot 1: Final loss
    axes[0, 0].plot(phases, [m["final_loss"] for m in cbp_metrics], "b-", label="CBP")
    axes[0, 0].plot(phases, [m["final_loss"] for m in adam_metrics], "r-", label="Adam")
    axes[0, 0].set_xlabel("Phase")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title("Final Loss After Each Phase")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Plasticity
    axes[0, 1].plot(phases, [m["plasticity"] for m in cbp_metrics], "b-", label="CBP")
    axes[0, 1].plot(phases, [m["plasticity"] for m in adam_metrics], "r-", label="Adam")
    axes[0, 1].set_xlabel("Phase")
    axes[0, 1].set_ylabel("Plasticity")
    axes[0, 1].set_title("Neural Plasticity After Each Phase")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Adaptation
    axes[1, 0].plot(phases, [m["adaptation"] for m in cbp_metrics], "b-", label="CBP")
    axes[1, 0].plot(phases, [m["adaptation"] for m in adam_metrics], "r-", label="Adam")
    axes[1, 0].set_xlabel("Phase")
    axes[1, 0].set_ylabel("Improvement (Initial - Final Loss)")
    axes[1, 0].set_title("Adaptation to New Tasks")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Forgetting
    axes[1, 1].plot(phases, [m["forgetting"] for m in cbp_metrics], "b-", label="CBP")
    axes[1, 1].plot(phases, [m["forgetting"] for m in adam_metrics], "r-", label="Adam")
    axes[1, 1].set_xlabel("Phase")
    axes[1, 1].set_ylabel("Forgetting")
    axes[1, 1].set_title("Catastrophic Forgetting")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot 5: Stability-Plasticity Tradeoff
    axes[2, 0].plot(phases, [m["tradeoff"] for m in cbp_metrics], "b-", label="CBP")
    axes[2, 0].plot(phases, [m["tradeoff"] for m in adam_metrics], "r-", label="Adam")
    axes[2, 0].set_xlabel("Phase")
    axes[2, 0].set_ylabel("Tradeoff Metric")
    axes[2, 0].set_title("Stability-Plasticity Tradeoff")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Plot 6: Forgetting vs. Plasticity Scatterplot
    axes[2, 1].scatter(
        [m["plasticity"] for m in cbp_metrics],
        [m["forgetting"] for m in cbp_metrics],
        color="blue",
        alpha=0.7,
        label="CBP",
    )
    axes[2, 1].scatter(
        [m["plasticity"] for m in adam_metrics],
        [m["forgetting"] for m in adam_metrics],
        color="red",
        alpha=0.7,
        label="Adam",
    )
    axes[2, 1].set_xlabel("Plasticity")
    axes[2, 1].set_ylabel("Forgetting")
    axes[2, 1].set_title("Plasticity vs. Forgetting")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png")
    plt.close()


def print_summary_metrics(cbp_metrics, adam_metrics):
    """Print summary statistics for the continual learning experiment."""
    # Calculate average metrics
    cbp_avg_loss = np.mean([m["final_loss"] for m in cbp_metrics])
    adam_avg_loss = np.mean([m["final_loss"] for m in adam_metrics])

    cbp_avg_plasticity = np.mean([m["plasticity"] for m in cbp_metrics])
    adam_avg_plasticity = np.mean([m["plasticity"] for m in adam_metrics])

    cbp_avg_adaptation = np.mean([m["adaptation"] for m in cbp_metrics])
    adam_avg_adaptation = np.mean([m["adaptation"] for m in adam_metrics])

    cbp_avg_forgetting = np.mean([m["forgetting"] for m in cbp_metrics])
    adam_avg_forgetting = np.mean([m["forgetting"] for m in adam_metrics])

    cbp_avg_tradeoff = np.mean([m["tradeoff"] for m in cbp_metrics])
    adam_avg_tradeoff = np.mean([m["tradeoff"] for m in adam_metrics])

    # Print summary table
    print("\n===== CONTINUAL LEARNING SUMMARY METRICS =====")
    print(f"{'Metric':<20} {'CBP':<15} {'Adam':<15} {'Ratio (CBP/Adam)':<20}")
    print("=" * 70)
    print(
        f"{'Average Loss':<20} {cbp_avg_loss:<15.6f} {adam_avg_loss:<15.6f} {cbp_avg_loss / adam_avg_loss:<20.6f}"
    )
    print(
        f"{'Average Plasticity':<20} {cbp_avg_plasticity:<15.6f} {adam_avg_plasticity:<15.6f} {cbp_avg_plasticity / adam_avg_plasticity:<20.6f}"
    )
    print(
        f"{'Average Adaptation':<20} {cbp_avg_adaptation:<15.6f} {adam_avg_adaptation:<15.6f} {cbp_avg_adaptation / adam_avg_adaptation:<20.6f}"
    )
    print(
        f"{'Average Forgetting':<20} {cbp_avg_forgetting:<15.6f} {adam_avg_forgetting:<15.6f} {cbp_avg_forgetting / max(adam_avg_forgetting, 1e-6):<20.6f}"
    )
    print(
        f"{'S-P Tradeoff':<20} {cbp_avg_tradeoff:<15.6f} {adam_avg_tradeoff:<15.6f} {cbp_avg_tradeoff / max(adam_avg_tradeoff, 1e-6):<20.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    # Use reasonable defaults for quick testing
    # For the full 20,000 shifts experiment, set debug_mode = False
    debug_mode = True

    if debug_mode:
        # Debug settings for quick testing
        continual_sine_learning(
            num_phase_shifts=50,  # Reduced number of shifts for testing
            epochs_per_phase=10,  # Fewer epochs per phase
            batch_size=1,
            eval_interval=5,
            save_interval=10,
            verbose=True,
        )
    else:
        # Full experiment with 20,000 phase shifts
        continual_sine_learning(
            num_phase_shifts=20000,  # Total number of phase shifts to perform
            epochs_per_phase=100,  # Epochs to train on each phase
            batch_size=64,  # Batch size for training
            learning_rate=1e-3,  # Learning rate for optimizer
            phase_shift_step=0.1,  # Amount to shift the phase by each time
            eval_interval=100,  # How often to evaluate forgetting on previous tasks
            save_interval=1000,  # How often to save metrics
            verbose=True  # Whether to print progress
        )
