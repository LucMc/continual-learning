import flax.linen as nn
import jax
import jax.numpy as jnp
import altair as alt
import polars as pl
import os
import numpy as np


def name_prefix(module: nn.Module) -> str:
    return module.name + "_" if module.name else ""


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


def plot_results(cbp_metrics, cbp_adamw_metrics, adam_metrics, adamw_metrics, filename_prefix="results"):
    """Plot metrics from the continual learning experiment using Altair."""

    # Prepare data for Altair
    def prepare_data(metrics_list, algorithm_name):
        data = []
        for metric in metrics_list:
            entry = {
                "phase": metric["phase"],
                "algorithm": algorithm_name,
                "final_loss": metric["final_loss"],
                "plasticity": metric["plasticity"],
                "adaptation": metric["adaptation"],
                "forgetting": metric["forgetting"],
                "tradeoff": metric["tradeoff"],
            }
            data.append(entry)
        return data

    # Combine data from all algorithms
    data = (
        prepare_data(cbp_metrics, "CBP")
        + prepare_data(cbp_adamw_metrics, "CBP+AdamW")
        + prepare_data(adam_metrics, "Adam")
        + prepare_data(adamw_metrics, "AdamW")
    )
    df = pl.DataFrame(data)

    # Set some common properties for all charts
    width = 300
    height = 250

    # Create individual charts

    # Plot 1: Final loss
    final_loss_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("final_loss:Q", title="MSE Loss"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "final_loss"],
        )
        .properties(width=width, height=height, title="Final Loss After Each Phase")
    )

    # Plot 2: Plasticity
    plasticity_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("plasticity:Q", title="Plasticity"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "plasticity"],
        )
        .properties(
            width=width, height=height, title="Neural Plasticity After Each Phase"
        )
    )

    # Plot 3: Adaptation
    adaptation_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("adaptation:Q", title="Improvement (Initial - Final Loss)"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "adaptation"],
        )
        .properties(width=width, height=height, title="Adaptation to New Tasks")
    )

    # Plot 4: Forgetting
    forgetting_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("forgetting:Q", title="Forgetting"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "forgetting"],
        )
        .properties(width=width, height=height, title="Catastrophic Forgetting")
    )

    # Plot 5: Stability-Plasticity Tradeoff
    tradeoff_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("tradeoff:Q", title="Tradeoff Metric"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "tradeoff"],
        )
        .properties(width=width, height=height, title="Stability-Plasticity Tradeoff")
    )

    # Plot 6: Forgetting vs. Plasticity Scatterplot
    scatter_chart = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("plasticity:Q", title="Plasticity"),
            y=alt.Y("forgetting:Q", title="Forgetting"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "plasticity", "forgetting"],
        )
        .properties(width=width, height=height, title="Plasticity vs. Forgetting")
    )

    # Combine charts into a grid layout
    row1 = alt.hconcat(final_loss_chart, plasticity_chart)
    row2 = alt.hconcat(adaptation_chart, forgetting_chart)
    row3 = alt.hconcat(tradeoff_chart, scatter_chart)
    final_chart = alt.vconcat(row1, row2, row3)

    # Save the chart
    os.makedirs("results", exist_ok=True) # Create results dir if not exists

    final_chart.save("./results/" + f"{filename_prefix}.svg")

    # Return the chart for display in notebooks
    return final_chart


def print_summary_metrics(cbp_metrics, cbp_adamw_metrics, adam_metrics, adamw_metrics):
    """Print summary statistics for the continual learning experiment."""
    # Calculate average metrics
    cbp_avg_loss = np.mean([m["final_loss"] for m in cbp_metrics])
    cbp_adamw_avg_loss = np.mean([m["final_loss"] for m in cbp_adamw_metrics])
    adam_avg_loss = np.mean([m["final_loss"] for m in adam_metrics])
    adamw_avg_loss = np.mean([m["final_loss"] for m in adamw_metrics])

    cbp_avg_plasticity = np.mean([m["plasticity"] for m in cbp_metrics])
    cbp_adamw_avg_plasticity = np.mean([m["plasticity"] for m in cbp_adamw_metrics])
    adam_avg_plasticity = np.mean([m["plasticity"] for m in adam_metrics])
    adamw_avg_plasticity = np.mean([m["plasticity"] for m in adamw_metrics])

    cbp_avg_adaptation = np.mean([m["adaptation"] for m in cbp_metrics])
    cbp_adamw_avg_adaptation = np.mean([m["adaptation"] for m in cbp_adamw_metrics])
    adam_avg_adaptation = np.mean([m["adaptation"] for m in adam_metrics])
    adamw_avg_adaptation = np.mean([m["adaptation"] for m in adamw_metrics])

    cbp_avg_forgetting = np.mean([m["forgetting"] for m in cbp_metrics])
    cbp_adamw_avg_forgetting = np.mean([m["forgetting"] for m in cbp_adamw_metrics])
    adam_avg_forgetting = np.mean([m["forgetting"] for m in adam_metrics])
    adamw_avg_forgetting = np.mean([m["forgetting"] for m in adamw_metrics])

    cbp_avg_tradeoff = np.mean([m["tradeoff"] for m in cbp_metrics])
    cbp_adamw_avg_tradeoff = np.mean([m["tradeoff"] for m in cbp_adamw_metrics])
    adam_avg_tradeoff = np.mean([m["tradeoff"] for m in adam_metrics])
    adamw_avg_tradeoff = np.mean([m["tradeoff"] for m in adamw_metrics])

    # Print summary table
    print("\n===== CONTINUAL LEARNING SUMMARY METRICS =====")
    print(
        f"{'Metric':<20} {'CBP':<15} {'CBP+AdamW':<15} {'Adam':<15} {'AdamW':<15}"
    )
    print("=" * 80)
    print(
        f"{'Average Loss':<20} {cbp_avg_loss:<15.6f} {cbp_adamw_avg_loss:<15.6f} {adam_avg_loss:<15.6f} {adamw_avg_loss:<15.6f}"
    )
    print(
        f"{'Average Plasticity':<20} {cbp_avg_plasticity:<15.6f} {cbp_adamw_avg_plasticity:<15.6f} {adam_avg_plasticity:<15.6f} {adamw_avg_plasticity:<15.6f}"
    )
    print(
        f"{'Average Adaptation':<20} {cbp_avg_adaptation:<15.6f} {cbp_adamw_avg_adaptation:<15.6f} {adam_avg_adaptation:<15.6f} {adamw_avg_adaptation:<15.6f}"
    )
    print(
        f"{'Average Forgetting':<20} {cbp_avg_forgetting:<15.6f} {cbp_adamw_avg_forgetting:<15.6f} {adam_avg_forgetting:<15.6f} {adamw_avg_forgetting:<15.6f}"
    )
    print(
        f"{'S-P Tradeoff':<20} {cbp_avg_tradeoff:<15.6f} {cbp_adamw_avg_tradeoff:<15.6f} {adam_avg_tradeoff:<15.6f} {adamw_avg_tradeoff:<15.6f}"
    )
    
    # Print relative comparisons
    print("\n===== RELATIVE COMPARISONS =====")
    print(
        f"{'Metric':<20} {'CBP/Adam':<15} {'CBP/AdamW':<15} {'CBP+AdamW/Adam':<15} {'CBP+AdamW/AdamW':<15}"
    )
    print("=" * 80)
    print(
        f"{'Average Loss':<20} {cbp_avg_loss / adam_avg_loss:<15.6f} {cbp_avg_loss / adamw_avg_loss:<15.6f} "
        f"{cbp_adamw_avg_loss / adam_avg_loss:<15.6f} {cbp_adamw_avg_loss / adamw_avg_loss:<15.6f}"
    )
    print(
        f"{'Average Plasticity':<20} {cbp_avg_plasticity / adam_avg_plasticity:<15.6f} {cbp_avg_plasticity / adamw_avg_plasticity:<15.6f} "
        f"{cbp_adamw_avg_plasticity / adam_avg_plasticity:<15.6f} {cbp_adamw_avg_plasticity / adamw_avg_plasticity:<15.6f}"
    )
    print(
        f"{'Average Adaptation':<20} {cbp_avg_adaptation / adam_avg_adaptation:<15.6f} {cbp_avg_adaptation / adamw_avg_adaptation:<15.6f} "
        f"{cbp_adamw_avg_adaptation / adam_avg_adaptation:<15.6f} {cbp_adamw_avg_adaptation / adamw_avg_adaptation:<15.6f}"
    )
    print(
        f"{'Average Forgetting':<20} {cbp_avg_forgetting / max(adam_avg_forgetting, 1e-6):<15.6f} {cbp_avg_forgetting / max(adamw_avg_forgetting, 1e-6):<15.6f} "
        f"{cbp_adamw_avg_forgetting / max(adam_avg_forgetting, 1e-6):<15.6f} {cbp_adamw_avg_forgetting / max(adamw_avg_forgetting, 1e-6):<15.6f}"
    )
    print(
        f"{'S-P Tradeoff':<20} {cbp_avg_tradeoff / max(adam_avg_tradeoff, 1e-6):<15.6f} {cbp_avg_tradeoff / max(adamw_avg_tradeoff, 1e-6):<15.6f} "
        f"{cbp_adamw_avg_tradeoff / max(adam_avg_tradeoff, 1e-6):<15.6f} {cbp_adamw_avg_tradeoff / max(adamw_avg_tradeoff, 1e-6):<15.6f}"
    )
    print("=" * 80)


