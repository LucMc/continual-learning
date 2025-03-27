import flax.linen as nn
import jax
import os
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training.train_state import TrainState
import numpy as np
import time
from typing import Dict, Any, Tuple, List, Callable, Literal
from dataclasses import dataclass, field

from continual_learning.optim.continual_backprop import (
    continual_backprop,
    CBPTrainState,
)
import continual_learning.nn.utils as utils
from continual_learning.nn import networks

import tyro

# --- Data Generation ---
def generate_sine_data(
    key, batch_size, phase_shift=0.0, amplitude=1.0, noise_level=0.0
):
    """Generate a batch of sine wave data with a given phase shift."""
    key1, key2 = random.split(key)
    x = random.uniform(key1, (batch_size, 1)) * 2 * jnp.pi  # Input between 0 and 2π
    noise = noise_level * random.normal(key2, (batch_size, 1))
    y = amplitude * jnp.sin(x + phase_shift) + noise
    return x, y


# --- Define a configuration structure for each method ---
@dataclass
class MethodConfig:
    name: str
    optimizer_fn: Callable
    optimizer_kwargs: Dict[str, Any]
    state_class: Callable
    state_kwargs: Dict[str, Any]
    train_step_fn: Callable


def continual_sine_learning(
    num_phase_shifts=20000,
    epochs_per_phase=100,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.01,
    phase_shift_step=0.1,
    eval_interval=100,
    save_interval=1000,
    verbose=True,
    cbp_kwargs={},
):
    """
    Run a continual learning experiment with a sine wave regression task.
    """
    if verbose:
        print(f"Starting continual learning with {num_phase_shifts} phase shifts")
        print(f"Training for {epochs_per_phase} epochs per phase")

    # Initialize random key
    key = random.PRNGKey(0)

    # Initialize network
    key, init_key = random.split(key)
    net = networks.SimpleNet(n_out=1, h_size=128)
    dummy_input = jnp.zeros((1, 1))
    base_params = net.init(init_key, dummy_input) # Use the same initial params for all

    # --- Define Loss and Grad Functions ---
    def loss_fn(params, inputs, targets):
        predictions, features = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)  # MSE loss
        return loss, (predictions, features)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # --- Define Train Step Functions ---
    # Note: features are needed by CBP's apply_gradients
    @jax.jit
    def cbp_train_step(state, inputs, targets, grad_fn_local=grad_fn):
        (loss, (preds, features)), grads = grad_fn_local(state.params, inputs, targets)
        # Extract features from the specific layer CBP operates on if needed,
        # or pass the whole features dictionary if CBPTrainState handles it.
        # Assuming CBPTrainState.apply_gradients expects the 'features' dict from loss_fn aux
        new_state = state.apply_gradients(grads=grads, features=features)
        return new_state, loss

    @jax.jit
    def standard_train_step(state, inputs, targets, grad_fn_local=grad_fn):
        (loss, aux), grads = grad_fn_local(state.params, inputs, targets)
        # Standard TrainState doesn't need features for apply_gradients
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    @jax.jit
    def evaluate(params, inputs, targets):
        # Assuming mutable="intermediates" might still be needed if apply_fn internally uses it
        # If not, you can simplify net.apply call here.
        predictions, _ = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)
        return loss

    # --- Define Method Configurations ---
    method_configs = [
        MethodConfig(
            name='CBP',
            optimizer_fn=optax.adam,
            optimizer_kwargs={'learning_rate': learning_rate},
            state_class=CBPTrainState,
            state_kwargs=cbp_kwargs,
            train_step_fn=cbp_train_step,
        ),
        MethodConfig(
            name='CBP_AdamW',
            optimizer_fn=optax.adamw,
            optimizer_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay},
            state_class=CBPTrainState,
            state_kwargs=cbp_kwargs,
            train_step_fn=cbp_train_step,
        ),
        MethodConfig(
            name='Adam',
            optimizer_fn=optax.adam,
            optimizer_kwargs={'learning_rate': learning_rate},
            state_class=TrainState,
            state_kwargs={},
            train_step_fn=standard_train_step,
        ),
        MethodConfig(
            name='SGD',
            optimizer_fn=optax.sgd,
            optimizer_kwargs={'learning_rate': learning_rate},
            state_class=TrainState,
            state_kwargs={},
            train_step_fn=standard_train_step,
        ),
        MethodConfig(
            name='AdamW',
            optimizer_fn=optax.adamw,
            optimizer_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay},
            state_class=TrainState,
            state_kwargs={},
            train_step_fn=standard_train_step,
        ),
    ]

    # --- Initialize States and Metric Storage using Dictionaries ---
    train_states = {}
    all_metrics = {config.name: [] for config in method_configs}
    all_best_losses = {config.name: {} for config in method_configs}
    all_current_losses = {config.name: {} for config in method_configs}

    for config in method_configs:
        tx = config.optimizer_fn(**config.optimizer_kwargs)
        # Use tree_map to ensure each state starts with an independent copy of params
        initial_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), base_params)
        train_states[config.name] = config.state_class.create(
            apply_fn=net.predict, # Or net.apply if predict doesn't exist / mutable needed
            params=initial_params_copy,
            tx=tx,
            **config.state_kwargs
        )

    method_names = [config.name for config in method_configs]

    # print weight sizes (only need to do once)
    print("Model Parameter Shapes:")
    for name, param in base_params["params"].items():
        print(f"  {name}: {param.get('kernel', param.get('embedding', 'N/A')).shape}") # Handle different param types

    # --- Main Training Loop ---
    start_time = time.time()
    for shift_idx in range(num_phase_shifts):
        phase_shift = phase_shift_step * shift_idx
        task_id = float(phase_shift) # Use consistent task identifier

        # --- Per-Phase Initialization ---
        initial_params = {}
        initial_losses = {}
        phase_losses = {name: [] for name in method_names}

        key, eval_key = random.split(key)
        eval_inputs, eval_targets = generate_sine_data(eval_key, 1000, phase_shift)

        for name in method_names:
            # Store initial parameters for plasticity
            current_params = train_states[name].params
            param_dict = current_params.get('params', current_params) if isinstance(current_params, dict) else current_params
            initial_params[name] = jax.tree_util.tree_map(lambda x: x.copy(), param_dict)

            # Evaluate initial performance
            initial_losses[name] = evaluate(train_states[name].params, eval_inputs, eval_targets)


        # --- Epoch Loop ---
        for epoch in range(epochs_per_phase):
            key, data_key = random.split(key)
            inputs, targets = generate_sine_data(data_key, batch_size, phase_shift)

            for config in method_configs:
                name = config.name
                state = train_states[name]
                # Perform training step using the configured function
                new_state, loss = config.train_step_fn(state, inputs, targets)
                train_states[name] = new_state
                phase_losses[name].append(float(loss))

        # --- Post-Phase Evaluation and Metrics ---
        final_losses = {}
        plasticity_metrics = {}
        adaptation_metrics = {}
        forgetting_metrics = {name: {"avg_forgetting": 0.0, "max_forgetting": 0.0} for name in method_names}
        tradeoff_metrics = {}
        current_params_dict = {}

        for name in method_names:
            # Evaluate final performance
            state = train_states[name]
            final_losses[name] = evaluate(state.params, eval_inputs, eval_targets)

            # Update best/current losses
            all_best_losses[name][task_id] = min(
                all_best_losses[name].get(task_id, float("inf")), float(final_losses[name])
            )
            all_current_losses[name][task_id] = float(final_losses[name])

            # Compute plasticity
            current_params = state.params
            param_dict = current_params.get('params', current_params) if isinstance(current_params, dict) else current_params
            current_params_dict[name] = param_dict # Store for potential reuse
            plasticity_metrics[name] = utils.compute_plasticity_metrics(
                initial_params[name], param_dict
            )

            # Compute adaptation
            adaptation_metrics[name] = float(initial_losses[name] - final_losses[name])

        # --- Forgetting Evaluation (Periodically) ---
        if shift_idx > 0 and (
            shift_idx % eval_interval == 0 or shift_idx == num_phase_shifts - 1
        ):
            previous_task_ids = [
                float(phase_shift_step * i) for i in range(0, shift_idx, eval_interval)
            ] # Ensure float keys

            for prev_task_id in previous_task_ids:
                key, prev_key = random.split(key)
                prev_inputs, prev_targets = generate_sine_data(
                    prev_key, 1000, prev_task_id # Use task_id (float) for phase shift
                )

                for name in method_names:
                    state = train_states[name]
                    prev_loss = evaluate(state.params, prev_inputs, prev_targets)
                    all_current_losses[name][prev_task_id] = float(prev_loss)

            # Compute overall forgetting metrics
            for name in method_names:
                 forgetting_metrics[name] = utils.compute_forgetting_metrics(
                    all_current_losses[name], all_best_losses[name]
                )

        # --- Compute Tradeoff and Store Metrics ---
        for name in method_names:
            tradeoff_metrics[name] = utils.stability_plasticity_tradeoff(
                adaptation_metrics[name], forgetting_metrics[name]["avg_forgetting"]
            )

            # Append metrics for this phase
            all_metrics[name].append(
                {
                    "phase": shift_idx,
                    "phase_shift": phase_shift,
                    "initial_loss": float(initial_losses[name]),
                    "final_loss": float(final_losses[name]),
                    "plasticity": plasticity_metrics[name]["total_plasticity"],
                    "adaptation": adaptation_metrics[name],
                    "forgetting": forgetting_metrics[name]["avg_forgetting"],
                    "tradeoff": tradeoff_metrics[name],
                }
            )

        # --- Print Progress ---
        if verbose and (shift_idx % 10 == 0 or shift_idx == num_phase_shifts - 1):
            elapsed = time.time() - start_time
            print(f"\n--- Phase {shift_idx}/{num_phase_shifts}, Shift: {phase_shift:.2f}, Time: {elapsed:.1f}s ---")
            print(":: Optimizer Metrics ::")
            for name in method_names:
                print(
                    f"  {name:<10}: Loss: {final_losses[name]:.6f}, "
                    f"Plasticity: {plasticity_metrics[name]['total_plasticity']:.6f}"
                )

            if shift_idx > 0 and (shift_idx % eval_interval == 0 or shift_idx == num_phase_shifts -1):
                 print(":: Forgetting & Tradeoff ::")
                 for name in method_names:
                    print(
                        f"  {name:<10}: Avg Forgetting: {forgetting_metrics[name]['avg_forgetting']:.6f}, "
                        f"Tradeoff: {tradeoff_metrics[name]:.6f}"
                    )

            # --- Extra CBP Logs (Handle specifically if needed) ---
            if 'CBP' in train_states: # Check if CBP methods are included
                print(":: Extra CBP Metrics ::")
                for name in method_names:
                    if isinstance(train_states[name], CBPTrainState):
                        try:
                            cbp_logs = train_states[name].cbp_state.logs
                            # Aggregate logs across layers if structure allows
                            nodes_reset = sum(v.get("nodes_reset", 0) for v in cbp_logs.values())
                            avg_age_sum = sum(v.get("avg_age", 0) for v in cbp_logs.values())
                            num_layers = len(cbp_logs) if cbp_logs else 1 # Avoid division by zero
                            avg_age = avg_age_sum / num_layers

                            print(f"  {name:<10}: Nodes Reset: {nodes_reset}, Avg Node Age: {avg_age:.2f}")
                        except AttributeError:
                            print(f"  {name:<10}: Could not retrieve CBP logs.")


            print("-" * (30 + len(f"Phase {shift_idx}/{num_phase_shifts}...")),"\n") # Dynamic separator

        # --- Save and Plot Results Periodically ---
        if shift_idx % save_interval == 0 or shift_idx == num_phase_shifts - 1:
            # Unpack the metrics dictionary for the plotting function
            utils.plot_results(
                *[all_metrics[name] for name in method_names], # Pass lists in order
                method_names=method_names, # Pass names for labels
                filename_prefix=f"results_{shift_idx}",
                avg_window=20,
            )
            # Note: You might need to modify plot_results to accept method_names
            # or ensure the order matches what plot_results expects.

    # --- Final Analysis ---
    utils.print_summary_metrics(
        *[all_metrics[name] for name in method_names],
        method_names=method_names # Also pass names here
    )

    return all_metrics # Return the consolidated dictionary


AllowedOptimizers = Literal["cbp", "cbp", "cbp_adam", "sgd", "adam", "adamw", "all"]

# --- Main Execution Block ---
@dataclass
class Args:
    methods: List[AllowedOptimizers] = field(default_factory=lambda: ["all"])
    debug: bool = True
    shifts: int = 50
    epochs: int = 10
    batch_size: int = 1 # Added batch size arg
    lr: float = 1e-3      # Added learning rate arg
    wd: float = 0.01      # Added weight decay arg
    save_interval: int = 1000 # Changed default to save less often in debug
    eval_interval: int = 5
    jit: bool = True
    replacement_rate: float = 0.1
    maturity_threshold: int = 3

if __name__ == "__main__":
    args = tyro.cli(Args)

    # Common settings
    common_kwargs = {
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "batch_size": args.batch_size,
        "phase_shift_step": 0.1,
        "verbose": True,
    }

    # CBP specific settings
    cbp_kwargs = {
        "maturity_threshold": args.maturity_threshold,
        "replacement_rate": args.replacement_rate,
    }
    print(f"--- Running: {', '.join(args.methods)} ---")

    if args.debug:
        from contextlib import nullcontext
        print("--- Running in DEBUG mode ---")
        # Example: import bpdb; bpdb.set_trace()
        with jax.disable_jit() if not args.jit else nullcontext():
            metrics = continual_sine_learning(
                num_phase_shifts=args.shifts,
                epochs_per_phase=args.epochs,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval if args.save_interval != 0 else float('inf'), # Handle 0 save interval
                cbp_kwargs=cbp_kwargs,
                **common_kwargs
            )
    else:
        print("--- Running in FULL experiment mode ---")
        # Adjust CBP kwargs for full run if needed
        # cbp_kwargs_full = {"maturity_threshold": 100, "replacement_rate": 0.01} # Example
        metrics = continual_sine_learning(
            num_phase_shifts=20000,
            epochs_per_phase=100,
            eval_interval=100,
            save_interval=1000,
            cbp_kwargs=cbp_kwargs, # Or use cbp_kwargs_full
            **common_kwargs
        )

    print("\n--- Experiment Finished ---")
