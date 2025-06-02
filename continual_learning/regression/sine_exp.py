import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, get_args

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import tyro
from flax.training.train_state import TrainState

import continual_learning.nn.utils as utils
from continual_learning.nn import SimpleNet, SimpleNetLayerNorm
from continual_learning.optim.continual_backprop import (
    CBPTrainState,
)

METHODS = Literal[
    "cbp",
    "cbp_adamw",
    "sgd",
    "adam",
    "adamw",
    "layer_norm",
    "layer_norm_wd",
    "layer_norm_cbp",
    "layer_norm_cbplnwd",
    "all",
]


# --- Data Generation ---
def generate_sine_data(key, batch_size, phase_shift=0.0, amplitude=1.0, noise_level=0.0):
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
    net: type[nn.Module]
    optimizer_fn: Callable
    optimizer_kwargs: dict[str, Any]
    state_class: Callable
    state_kwargs: dict[str, Any]
    train_step_fn: Callable


def continual_sine_learning(
    methods=["all"],
    # net_name="standard", # TODO: Make list like methods
    num_phase_shifts=20000,
    epochs_per_phase=100,
    batch_size=1,
    learning_rate=1e-3,
    weight_decay=0.01,
    phase_shift_step=0.1,
    eval_interval=100,
    save_interval: float = 1000,
    verbose=True,
    seed=0,
    cbp_kwargs={},
    label="standard",
):
    """
    Run a continual learning experiment with a sine wave regression task.
    """
    if verbose:
        print(f"Starting continual learning with {num_phase_shifts} phase shifts")
        print(f"Training for {epochs_per_phase} epochs per phase")

    def initialise_network(net, seed):
        # key, init_key = random.split(key)
        dummy_input = jnp.zeros((1, 1))
        base_params = net.init(key, dummy_input)  # Use the same initial params for all
        return base_params

    # --- Define Loss and Grad Functions ---
    @partial(jax.jit, static_argnames=["net"])
    def loss_fn(params, inputs, targets, net):
        predictions, features = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)  # MSE loss
        return loss, (predictions, features)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # --- Define Train Step Functions ---
    # Note: features are needed by CBP's apply_gradients
    @partial(jax.jit, static_argnames=["net"])
    def cbp_train_step(net, state, inputs, targets, grad_fn_local=grad_fn):
        (loss, (preds, features)), grads = grad_fn_local(state.params, inputs, targets, net)
        # Extract features from the specific layer CBP operates on if needed,
        # or pass the whole features dictionary if CBPTrainState handles it.
        # Assuming CBPTrainState.apply_gradients expects the 'features' dict from loss_fn aux
        new_state = state.apply_gradients(grads=grads, features=features)
        return new_state, loss

    @partial(jax.jit, static_argnums=0)
    def standard_train_step(net, state, inputs, targets, grad_fn_local=grad_fn):
        (loss, aux), grads = grad_fn_local(state.params, inputs, targets, net)
        # Standard TrainState doesn't need features for apply_gradients
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    @partial(jax.jit, static_argnums=0)
    def evaluate(net, params, inputs, targets):
        # Assuming mutable="intermediates" might still be needed if apply_fn internally uses it
        # If not, you can simplify net.apply call here.
        predictions, _ = net.apply(params, inputs, mutable="intermediates")
        loss = jnp.mean((predictions - targets) ** 2)
        return loss

    # --- Define Method Configurations ---
    method_configs = []

    if "cbp" in methods:
        method_configs.append(
            MethodConfig(
                name="CBP",
                net=SimpleNet,
                optimizer_fn=optax.adam,
                optimizer_kwargs={"learning_rate": learning_rate},
                state_class=CBPTrainState,
                state_kwargs=cbp_kwargs,
                train_step_fn=cbp_train_step,
            )
        )

    if "cbp_adamw" in methods:
        method_configs.append(
            MethodConfig(
                name="CBP_AdamW",
                net=SimpleNet,
                optimizer_fn=optax.adamw,
                optimizer_kwargs={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
                state_class=CBPTrainState,
                state_kwargs=cbp_kwargs,
                train_step_fn=cbp_train_step,
            )
        )

    if "adam" in methods:
        method_configs.append(
            MethodConfig(
                name="Adam",
                net=SimpleNet,
                optimizer_fn=optax.adam,
                optimizer_kwargs={"learning_rate": learning_rate},
                state_class=TrainState,
                state_kwargs={},
                train_step_fn=standard_train_step,
            )
        )

    if "sgd" in methods:
        method_configs.append(
            MethodConfig(
                name="SGD",
                net=SimpleNet,
                optimizer_fn=optax.sgd,
                optimizer_kwargs={"learning_rate": learning_rate},
                state_class=TrainState,
                state_kwargs={},
                train_step_fn=standard_train_step,
            )
        )

    if "adamw" in methods:
        method_configs.append(
            MethodConfig(
                name="AdamW",
                net=SimpleNet,
                optimizer_fn=optax.adamw,
                optimizer_kwargs={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
                state_class=TrainState,
                state_kwargs={},
                train_step_fn=standard_train_step,
            )
        )

    if "layer_norm" in methods:
        method_configs.append(
            MethodConfig(
                name="layer_norm",
                net=SimpleNetLayerNorm,
                optimizer_fn=optax.adam,
                optimizer_kwargs={
                    "learning_rate": learning_rate,
                },
                state_class=TrainState,
                state_kwargs={},
                train_step_fn=standard_train_step,
            )
        )
    if "layer_norm_cbp" in methods:
        method_configs.append(
            MethodConfig(
                name="layer_norm_cbp",
                net=SimpleNetLayerNorm,
                optimizer_fn=optax.adam,
                optimizer_kwargs={
                    "learning_rate": learning_rate,
                },
                state_class=CBPTrainState,
                state_kwargs=cbp_kwargs,
                train_step_fn=cbp_train_step,
            )
        )
    if "layer_norm_wd" in methods:
        method_configs.append(
            MethodConfig(
                name="layer_norm_wd",
                net=SimpleNetLayerNorm,
                optimizer_fn=optax.adamw,
                optimizer_kwargs={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
                state_class=TrainState,
                state_kwargs={},
                train_step_fn=standard_train_step,
            )
        )
    if "layer_norm_cbplnwd" in methods:
        method_configs.append(
            MethodConfig(
                name="layer_norm_cbplnwd",
                net=SimpleNetLayerNorm,
                optimizer_fn=optax.adamw,
                optimizer_kwargs={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
                state_class=CBPTrainState,
                state_kwargs=cbp_kwargs,
                train_step_fn=cbp_train_step,
            )
        )
    # --- Initialize States and Metric Storage using Dictionaries ---
    train_states = {}
    all_metrics = {config.name: [] for config in method_configs}
    all_best_losses = {config.name: {} for config in method_configs}
    all_current_losses = {config.name: {} for config in method_configs}
    key = random.PRNGKey(seed)

    for config in method_configs:
        net = config.net()
        base_params = initialise_network(net, key)  # Use same key for consistancy
        tx = config.optimizer_fn(**config.optimizer_kwargs)
        # Use tree_map to ensure each state starts with an independent copy of params
        initial_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), base_params)
        train_states[config.name] = config.state_class.create(
            apply_fn=net.predict,  # Or net.apply if predict doesn't exist / mutable needed
            params=initial_params_copy,
            tx=tx,
            **config.state_kwargs,
        )

    method_names = [config.name for config in method_configs]

    # print weight sizes (only need to do once)
    print("Model Parameter Shapes:")
    # for name, param in base_params["params"].items():
    # print(
    #     f"  {name}: {param.get('kernel', param.get('embedding', 'N/A')).shape}"
    # )  # Handle different param types

    # --- Main Training Loop ---
    start_time = time.time()
    for shift_idx in range(num_phase_shifts):
        phase_shift = phase_shift_step * shift_idx
        task_id = float(phase_shift)  # Use consistent task identifier

        # --- Per-Phase Initialization ---
        initial_params = {}
        initial_losses = {}
        phase_losses = {name: [] for name in method_names}

        key, eval_key = random.split(key)
        eval_inputs, eval_targets = generate_sine_data(eval_key, 1000, phase_shift)

        for config in method_configs:
            name = config.name
            # Store initial parameters for plasticity
            current_params = train_states[name].params
            param_dict = (
                current_params.get("params", current_params)
                if isinstance(current_params, dict)
                else current_params
            )
            initial_params[name] = jax.tree_util.tree_map(lambda x: x.copy(), param_dict)

            # Evaluate initial performance
            initial_losses[name] = evaluate(
                config.net(), train_states[name].params, eval_inputs, eval_targets
            )

        # --- Epoch Loop ---
        for epoch in range(epochs_per_phase):
            key, data_key = random.split(key)
            inputs, targets = generate_sine_data(data_key, batch_size, phase_shift)

            for config in method_configs:
                name = config.name
                state = train_states[name]
                # Perform training step using the configured function
                new_state, loss = config.train_step_fn(config.net(), state, inputs, targets)
                train_states[name] = new_state
                phase_losses[name].append(float(loss))

        # --- Post-Phase Evaluation and Metrics ---
        final_losses = {}
        plasticity_metrics = {}
        adaptation_metrics = {}
        forgetting_metrics = {
            name: {"avg_forgetting": 0.0, "max_forgetting": 0.0} for name in method_names
        }
        tradeoff_metrics = {}
        current_params_dict = {}

        for config in method_configs:
            name = config.name
            # Evaluate final performance
            state = train_states[name]
            final_losses[name] = evaluate(
                config.net(), state.params, eval_inputs, eval_targets
            )

            # Update best/current losses
            all_best_losses[name][task_id] = min(
                all_best_losses[name].get(task_id, float("inf")),
                float(final_losses[name]),
            )
            all_current_losses[name][task_id] = float(final_losses[name])

            # Compute plasticity
            current_params = state.params
            param_dict = (
                current_params.get("params", current_params)
                if isinstance(current_params, dict)
                else current_params
            )
            current_params_dict[name] = param_dict  # Store for potential reuse
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
            ]  # Ensure float keys

            for prev_task_id in previous_task_ids:
                key, prev_key = random.split(key)
                prev_inputs, prev_targets = generate_sine_data(
                    prev_key,
                    1000,
                    prev_task_id,  # Use task_id (float) for phase shift
                )

                for config in method_configs:
                    name = config.name
                    state = train_states[name]
                    prev_loss = evaluate(config.net(), state.params, prev_inputs, prev_targets)
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
            print(
                f"\n--- Phase {shift_idx}/{num_phase_shifts}, Shift: {phase_shift:.2f}, Time: {elapsed:.1f}s ---"
            )
            print(":: Optimizer Metrics ::")

            for name in method_names:
                print(
                    f"  {name:<10}: Loss: {final_losses[name]:.6f}, "
                    f"Plasticity: {plasticity_metrics[name]['total_plasticity']:.6f}"
                )

            if shift_idx > 0 and (
                shift_idx % eval_interval == 0 or shift_idx == num_phase_shifts - 1
            ):
                print(":: Forgetting & Tradeoff ::")
                for name in method_names:
                    print(
                        f"  {name:<10}: Avg Forgetting: {forgetting_metrics[name]['avg_forgetting']:.6f}, "
                        f"Tradeoff: {tradeoff_metrics[name]:.6f}"
                    )

            # --- Extra CBP Logs (Handle specifically if needed) ---
            if "CBP" in train_states:  # Check if CBP methods are included
                print(":: Extra CBP Metrics ::")
                for name in method_names:
                    if isinstance(train_states[name], CBPTrainState):
                        try:
                            cbp_logs = train_states[name].cbp_state.logs
                            # Aggregate logs across layers if structure allows
                            nodes_reset = sum(
                                v.get("nodes_reset", 0) for v in cbp_logs.values()
                            )
                            avg_age_sum = sum(v.get("avg_age", 0) for v in cbp_logs.values())
                            num_layers = (
                                len(cbp_logs) if cbp_logs else 1
                            )  # Avoid division by zero
                            avg_age = avg_age_sum / num_layers

                            print(
                                f"  {name:<10}: Nodes Reset: {nodes_reset}, Avg Node Age: {avg_age:.2f}"
                            )
                        except AttributeError:
                            print(f"  {name:<10}: Could not retrieve CBP logs.")

            print(
                "-" * (30 + len(f"Phase {shift_idx}/{num_phase_shifts}...")), "\n"
            )  # Dynamic separator

        # --- Save and Plot Results Periodically ---
        if shift_idx % save_interval == 0 or shift_idx == num_phase_shifts - 1:
            # Unpack the metrics dictionary for the plotting function
            utils.plot_results(
                all_metrics=all_metrics,
                # [all_metrics[name] for name in method_names], # Pass lists in order
                # method_names=method_names, # Pass names for labels
                filename_prefix=f"{label}_{len(methods)}m_{shift_idx}",
                avg_window=25,
            )
            # Note: You might need to modify plot_results to accept method_names
            # or ensure the order matches what plot_results expects.

    # --- Final Analysis ---
    utils.print_summary_metrics(
        all_metrics=all_metrics
        # *[all_metrics[name] for name in method_names],
        # method_names=method_names # Also pass names here
    )

    return all_metrics  # Return the consolidated dictionary


# --- Main Execution Block ---
@dataclass
class Args:
    # net_name: Literal["standard", "layer_norm"] = "standard"
    methods: list[METHODS] = field(default_factory=lambda: ["all"])
    seed: int = 0
    debug: bool = False
    shifts: int = 20_000
    epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-3
    wd: float = 0.01
    save_interval: int = 1000
    eval_interval: int = 100
    jit: bool = True
    replacement_rate: float = 0.01
    maturity_threshold: int = 100
    label: str = "standard"
    # preset: Literal["layer_norm", "standard", "none"] = "none" # Load config files for fine grained experiments


if __name__ == "__main__":
    args = tyro.cli(Args)
    methods = list(get_args(METHODS))[:-1] if "all" in args.methods else args.methods

    # Continual backprop optim kwargs
    cbp_kwargs = {
        "maturity_threshold": args.maturity_threshold,
        "replacement_rate": args.replacement_rate,
        "rng": random.PRNGKey(args.seed),
    }

    # Common settings
    common_kwargs = {
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "batch_size": args.batch_size,
        "phase_shift_step": 0.1,
        "verbose": True,
        "seed": args.seed,
        "methods": methods,
        "cbp_kwargs": cbp_kwargs,
        "label": args.label,
    }

    print(f"--- Running: {', '.join(methods)} ---")

    if args.debug:
        from contextlib import nullcontext

        print("--- Running in DEBUG mode ---")
        # Example: import bpdb; bpdb.set_trace()
        with jax.default_device("cpu"):
            with jax.disable_jit() if not args.jit else nullcontext():
                print("!! disabled jit !!")
                metrics = continual_sine_learning(
                    num_phase_shifts=args.shifts,
                    epochs_per_phase=args.epochs,
                    eval_interval=args.eval_interval,
                    save_interval=args.save_interval
                    if args.save_interval != 0
                    else float("inf"),  # Handle 0 save interval
                    **common_kwargs,
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
            **common_kwargs,
        )

    print("\n--- Experiment Finished ---")
