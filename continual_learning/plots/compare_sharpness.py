#!/usr/bin/env python3
"""
Compare runs with different sharpness values from the same W&B group.
This script automatically detects all sharpness values in the group and plots them.
"""

import pandas as pd
import tyro
from pathlib import Path
from typing import Optional
import altair as alt
import re
import wandb

from analyze import (
    fetch_and_process_data,
    create_chart,
    create_peak_final_bar_chart,
    canonicalize_metric_name,
    resolve_metric_labels,
    slugify_title,
    build_combined_metric_suffix,
    normalize_combined_metric_name,
    combined_total_metric_key,
    NETWORK_METRIC_COMBINATIONS,
)


def extract_sharpness_from_name(run_name: str) -> Optional[float]:
    """Extract sharpness value from run name."""
    match = re.search(r'sharpness=([0-9.]+)', run_name)
    if match:
        return float(match.group(1))
    return None


def main(
    wandb_entity: str = "lucmc",
    wandb_project: str = "crl_experiments",
    group: str = "slippery_ant_ccbp_sharpness_sweep",
    metric: str = "charts/mean_episodic_return",
    combine_networks: bool = False,
    normalize_networks: bool = False,
    plot_title: Optional[str] = None,
    output_dir: str = "./plots",
    ext: str = "png",
    show_iqr: bool = True,
    base_text_size: float = 20.0,
    bar_chart: bool = False,
    chart_width: Optional[int] = None,
    chart_height: Optional[int] = None,
    line_width: float = 8.0,
    y_tick_count: Optional[int] = 6,
    x_axis_max: float = 400.0,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    log_scale: bool = False,
):
    """
    Compare runs with different sharpness values.

    Args:
        wandb_entity: W&B entity name
        wandb_project: W&B project name
        group: W&B group name
        metric: Metric to plot
        combine_networks: Whether to combine actor/value network metrics
        normalize_networks: Whether to normalize combined metrics
        plot_title: Custom plot title
        output_dir: Output directory for plots
        ext: Output file extension
        show_iqr: Show interquartile range
        base_text_size: Base font size
        bar_chart: Generate peak vs final bar chart
        chart_width: Chart width in pixels
        chart_height: Chart height in pixels
        line_width: Line width
        y_tick_count: Number of y-axis ticks
        x_axis_max: Maximum x-axis value (in millions)
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        log_scale: Use log scale for y-axis
    """

    # Initialize wandb API
    api = wandb.Api()

    # Fetch all runs in the group to discover sharpness values
    print(f"\n=== Discovering sharpness values in group '{group}' ===")
    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"group": group}
    )

    # Extract unique sharpness values
    sharpness_values = set()
    for run in runs:
        sharpness = extract_sharpness_from_name(run.name)
        if sharpness is not None:
            sharpness_values.add(sharpness)

    if not sharpness_values:
        print("No sharpness values found in run names!")
        return

    # Sort sharpness values
    sharpness_values = sorted(sharpness_values)
    print(f"Found sharpness values: {sharpness_values}")

    # Canonicalize metric
    metric = canonicalize_metric_name(metric)

    # Handle network combinations
    metrics_to_fetch = [metric]
    if combine_networks:
        normalized_metric = normalize_combined_metric_name(metric)
        combo = NETWORK_METRIC_COMBINATIONS.get(normalized_metric)
        if combo is None:
            available = ", ".join(sorted(NETWORK_METRIC_COMBINATIONS))
            raise ValueError(
                f"combine_networks is only defined for the metrics [{available}]; "
                f"received '{metric}'."
            )
        metrics_to_fetch = [normalized_metric]

    # Fetch data for each sharpness value
    all_dfs = []
    summary_records = []

    for sharpness in sharpness_values:
        # Convert to int if it's a whole number for pattern matching
        sharpness_int = int(sharpness) if sharpness == int(sharpness) else sharpness
        label = f"sharpness={sharpness_int}"
        # Pattern to match runs with this sharpness value
        # Using wildcard pattern that includes sharpness value
        pattern = f"*sharpness={sharpness_int},*"

        print(f"\n=== Fetching data for {label} (pattern: {pattern}) ===")
        df = fetch_and_process_data(
            wandb_entity,
            wandb_project,
            group,
            metrics_to_fetch,
            combine_networks,
            normalize_networks,
            split_by=None,
            algorithm_name=label,
            name_pattern=pattern,
        )

        if not df.empty:
            all_dfs.append(df)
            records = df.attrs.get("performance_summary")
            if records:
                summary_records.extend(records)

    # Combine all dataframes
    if not all_dfs:
        print("No data found for any sharpness value!")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    if summary_records:
        df.attrs["performance_summary"] = summary_records

    # Determine the metric to plot
    if combine_networks:
        normalized_metric = normalize_combined_metric_name(metric)
        combined_total_metric = combined_total_metric_key(
            normalized_metric, normalize_networks
        )
        metric_label = resolve_metric_labels(combined_total_metric)[0]
        title_suffix = metric_label
        save_suffix = build_combined_metric_suffix(normalized_metric, normalize_networks)
    else:
        metric_label = resolve_metric_labels(metric)[0]
        title_suffix = metric_label
        save_suffix = metric.replace("/", "_")

    # Create title
    if plot_title:
        title = plot_title
    else:
        title = f"{group.replace('_', ' ').title()}: {title_suffix} (IQM)"

    # Create chart
    chart = create_chart(
        df,
        metrics_to_fetch,
        title,
        combine_networks,
        show_iqr=show_iqr,
        base_text_size=base_text_size,
        chart_width=chart_width,
        chart_height=chart_height,
        line_width=line_width,
        y_tick_count=y_tick_count,
        x_axis_max=x_axis_max,
        y_min=y_min,
        y_max=y_max,
        log_scale=log_scale,
    )

    # Save chart
    slugged_title = slugify_title(plot_title) if plot_title else None
    filename_stem = (
        slugged_title if slugged_title else f"{group}_sharpness_ablation_{save_suffix}_iqm_smoothed"
    )
    save_path = Path(output_dir) / ext / f"{filename_stem}.{ext}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"\nChart saved to: {save_path}")

    # Create bar chart if requested
    if bar_chart and summary_records:
        summary_df = pd.DataFrame(summary_records)
        bar_title = plot_title if plot_title else f"{group.replace('_', ' ').title()}: Peak vs Final IQM"
        bar_chart_obj = create_peak_final_bar_chart(
            summary_df,
            bar_title,
            metric_label,
            base_text_size=base_text_size,
            log_scale=log_scale,
        )
        bar_filename_stem = (
            f"{slugged_title}_peak_vs_final_bar"
            if slugged_title
            else f"{group}_sharpness_ablation_{save_suffix}_peak_vs_final_bar"
        )
        bar_save_path = Path(output_dir) / ext / f"{bar_filename_stem}.{ext}"
        bar_save_path.parent.mkdir(exist_ok=True, parents=True)
        bar_chart_obj.save(str(bar_save_path))
        print(f"Summary bar chart saved to: {bar_save_path}")

    return chart


if __name__ == "__main__":
    tyro.cli(main)
