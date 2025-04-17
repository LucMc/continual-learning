import flax.linen as nn
import jax
import jax.numpy as jnp
import altair as alt
import polars as pl
import pandas as pd
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


def plot_results(all_metrics, filename_prefix="results", avg_window=3):
    """
    Plot metrics from the continual learning experiment using Altair with moving averages.

    Parameters:
    -----------
    cbp_metrics, cbp_adamw_metrics, adam_metrics, adamw_metrics : list
        Lists of metric dictionaries for each algorithm
    filename_prefix : str
        Prefix for the output file
    avg_window : int
        Size of the moving average window to smooth the plots
    """

    def prepare_data(metrics_list, algorithm_name):
        data = []
        for metric in metrics_list:
            # Ensure 'phase' is present and handled correctly if missing
            phase = metric.get("phase", None)
            if phase is not None:
                 entry = {
                    "phase": phase,
                    "algorithm": algorithm_name,
                    "final_loss": metric.get("final_loss"), 
                    "plasticity": metric.get("plasticity"),
                    "adaptation": metric.get("adaptation"),
                    "forgetting": metric.get("forgetting"),
                    "tradeoff": metric.get("tradeoff"),
                }
                 data.append(entry)
        return data

    # Combine data from all algorithms
    data = []
    for name, metric in all_metrics.items(): data += prepare_data(metric, name)

    # --- Polars Section ---
    df = pl.DataFrame(data)

    # Filter out any rows where phase might be None, if prepare_data allowed them
    df = df.filter(pl.col("phase").is_not_null())

    # Define the metric columns for which to calculate moving averages
    metric_cols = ['final_loss', 'plasticity', 'adaptation', 'forgetting', 'tradeoff']

    # Calculate moving averages using Polars window functions
    df_avg = df.sort("algorithm", "phase").with_columns(
        [
            pl.col(col)
            .rolling_mean(window_size=avg_window, min_periods=1) # Calculate rolling mean
            .over("algorithm") # Partition by algorithm
            .alias(f"{col}_ma") # Name the new column
            for col in metric_cols if col in df.columns # Check if column exists
        ]
    )

    # ---  Generate Plots ---
    width = 600
    height = 450

    # Plot 1: Final loss
    final_loss_chart = (
        alt.Chart(df_avg)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("final_loss_ma:Q", title="MSE Loss (Moving Avg)"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "final_loss", "final_loss_ma"],
        )
        .properties(width=width, height=height, title=f"Final Loss After Each Phase (MA{avg_window})")
    )

    # Plot 2: Plasticity
    plasticity_chart = (
        alt.Chart(df_avg)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("plasticity_ma:Q", title="Plasticity (Moving Avg)",
                    scale=alt.Scale(domain=[0.0, 0.0006], clamp=True)),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "plasticity", "plasticity_ma"],
        )
        .properties(
            width=width, height=height, title=f"Neural Plasticity After Each Phase (MA{avg_window})"
        )
    )

    # Plot 3: Adaptation
    adaptation_chart = (
        alt.Chart(df_avg)
        .mark_line()
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("adaptation_ma:Q", title="Improvement (Initial - Final Loss) (Moving Avg)",
                    scale=alt.Scale(domain=[-0.1, 0.1], clamp=True)),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "adaptation", "adaptation_ma"],
        )
        .properties(width=width, height=height, title=f"Adaptation to New Tasks (MA{avg_window})")
    )

    # Plot 4: Forgetting
    # TODO: Change forgetting to bar chart as only tested at eval intervals
    df_filtered_forgetting_pl = df.filter(pl.col('forgetting') > 0) # Assuming forgetting is non-negative
    forgetting_chart = (
        alt.Chart(df_filtered_forgetting_pl)
        .mark_line(strokeWidth=12)
        .encode(
            x=alt.X("phase:Q", title="Phase"),
            y=alt.Y("forgetting:Q", title="Forgetting (Moving Avg)"),
            color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
            tooltip=["phase", "algorithm", "forgetting"],
        )
        .properties(width=width, height=height, title=f"Catastrophic Forgetting (MA{avg_window})")
    )


    # Combine charts into a grid layout
    row1 = alt.hconcat(final_loss_chart, plasticity_chart)
    row2 = alt.hconcat(adaptation_chart, forgetting_chart)
    # row3 = alt.hconcat(tradeoff_chart, scatter_chart) # Assuming tradeoff_chart might exist
    final_chart = alt.vconcat(row1, row2).properties(title=filename_prefix) # , row3)

    # Save the chart
    os.makedirs("results", exist_ok=True)  # Create results dir if not exists
    save_path = f"./results/{filename_prefix}_ma{avg_window}.png" # svg for paper? Use .png for broad compatibility
    print(f"Saving chart to: {save_path}") # Add print statement for confirmation
    try:
        final_chart.save(save_path)
    except Exception as e:
        print(f"Error saving chart: {e}")

    return final_chart



def print_summary_metrics(all_metrics, comparison_pairs=None):
    """
    Print adaptive, minimal summary statistics using only all_metrics.
    Derives methods and metrics directly from the input dictionary.

    Args:
        all_metrics: A dictionary where keys are method names (str)
                    and values are lists of dictionaries, each containing metrics
                    for a single phase/task (e.g., {'final_loss': ..., 'plasticity': ...}).
        comparison_pairs: Optional list of tuples defining which methods to compare in the
                         relative comparisons section. Each tuple is (method1, method2)
                         for comparing method1/method2. If None, a default set of pairs will be used.
    """
    import numpy as np
    
    # Extract method names
    methods = list(all_metrics.keys())
    
    # Define prettier display names for metrics
    metric_display = {
        "final_loss": "Average Loss",
        "plasticity": "Average Plasticity",
        "adaptation": "Average Adaptation",
        "forgetting": "Average Forgetting",
        "tradeoff": "S-P Tradeoff"
    }
    
    # Calculate average metrics for each method
    avg_metrics = {}
    # Determine metric names from the first entry of the first method
    if methods and all_metrics[methods[0]]:
        # Get metrics from first item of first method
        sample_metrics = all_metrics[methods[0]][0]
        metric_names = [m for m in sample_metrics.keys() if m in metric_display]
    else:
        metric_names = []
    
    for method in methods:
        avg_metrics[method] = {}
        for metric in metric_names:
            # Calculate average for each metric
            avg_metrics[method][metric] = np.mean([m[metric] for m in all_metrics[method]])
    
    # Print summary table
    print("\n===== CONTINUAL LEARNING SUMMARY METRICS =====")
    header = f"{'Metric':<20}"
    for method in methods:
        header += f" {method:<15}"

    print(header)
    print("=" * (20 + 15 * len(methods)))
    
    for metric in metric_names:
        display_name = metric_display.get(metric, metric)
        row = f"{display_name:<20}"
        for method in methods:
            row += f" {avg_metrics[method][metric]:<15.6f}"
        print(row)
    
    # Create relative comparison pairs
    if len(methods) > 1:
        if comparison_pairs is None:
            # If methods contain "CBP", "CBP+AdamW", "Adam", "AdamW", use original comparison pairs
            # if set(["CBP", "CBP+AdamW", "Adam", "AdamW"]).issubset(set(methods)):
            #     pairs = [
            #         ("CBP", "Adam"), 
            #         ("CBP", "AdamW"), 
            #         ("CBP+AdamW", "Adam"), 
            #         ("CBP+AdamW", "AdamW")
            #     ]
            # else:
                # Default to comparing first method with all others
            pairs = [(methods[0], m) for m in methods[1:]]
        else:
            # Use provided pairs, filtering out any invalid methods
            pairs = [(m1, m2) for m1, m2 in comparison_pairs 
                    if m1 in methods and m2 in methods]
        
        if pairs:
            print("\n===== RELATIVE COMPARISONS =====")
            header = f"{'Metric':<20}"
            for m1, m2 in pairs:
                header += f" {m1}/{m2:<15}"
            print(header)
            print("=" * (20 + 15 * len(pairs)))
            
            for metric in metric_names:
                display_name = metric_display.get(metric, metric)
                row = f"{display_name:<20}"
                for m1, m2 in pairs:
                    # For forgetting and tradeoff, use max with small epsilon to avoid division by zero
                    if metric in ["forgetting", "tradeoff"]:
                        denominator = max(avg_metrics[m2][metric], 1e-6)
                    else:
                        denominator = avg_metrics[m2][metric]
                    ratio = avg_metrics[m1][metric] / denominator
                    row += f" {ratio:<15.6f}"
                print(row)
            print("=" * (20 + 15 * len(pairs)))
    """

    # Plot 5: Stability-Plasticity Tradeoff
    # tradeoff_chart = (
    #     alt.Chart(df)
    #     .mark_line()
    #     .encode(
    #         x=alt.X("phase:Q", title="Phase"),
    #         y=alt.Y("tradeoff_ma:Q", title="Tradeoff Metric (Moving Avg)"),
    #         color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
    #         tooltip=["phase", "algorithm", "tradeoff", "tradeoff_ma"],
    #     )
    #     .properties(width=width, height=height, title=f"Stability-Plasticity Tradeoff (MA{window_size})")
    # )
    #
    # tradeoff_points = (
    #     alt.Chart(df)
    #     .mark_point(opacity=0.3, size=30)
    #     .encode(
    #         x="phase:Q",
    #         y="tradeoff:Q",
    #         color="algorithm:N",
    #     )
    # )
    #
    # tradeoff_chart = tradeoff_chart + tradeoff_points
    #
    # # Plot 6: Forgetting vs. Plasticity Scatterplot
    # # We'll use the moving average data for the scatter plot
    # scatter_chart = (
    #     alt.Chart(df)
    #     .mark_circle(size=60, opacity=0.7)
    #     .encode(
    #         x=alt.X("plasticity_ma:Q", title="Plasticity (Moving Avg)"),
    #         y=alt.Y("forgetting_ma:Q", title="Forgetting (Moving Avg)"),
    #         color=alt.Color("algorithm:N", legend=alt.Legend(title="Algorithm")),
    #         tooltip=["phase", "algorithm", "plasticity_ma", "forgetting_ma"],
    #     )
    #     .properties(width=width, height=height, title=f"Plasticity vs. Forgetting (MA{window_size})")
    # )
    """
