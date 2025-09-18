#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
from collections import defaultdict
from typing import Optional, Union, List


def compute_iqm(values: np.ndarray) -> float:
    if len(values) < 4:
        return np.mean(values)
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return np.mean(values[mask]) if np.any(mask) else np.mean(values)


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
    finished = [r for r in runs if r.state == "finished"]
    print(f"Fetched {len(runs)} runs, {len(finished)} finished")
    return finished


def combine_network_metrics(df: pd.DataFrame, base_metric: str) -> pd.DataFrame:
    mapping = {
        "dormant_neurons": {
            "value": "nn/value_dormant_neurons/total_ratio",
            "actor": "nn/actor_dormant_neurons/total_ratio",
            "total": "nn/total_dormant_neurons/total_ratio",
        },
        "linearised_neurons": {
            "value": "nn/value_linearised_neurons/total_ratio",
            "actor": "nn/actor_linearised_neurons/total_ratio",
            "total": "nn/total_linearised_neurons/total_ratio",
        },
    }

    if base_metric not in mapping:
        return df
    metrics = mapping[base_metric]

    value_data = df[df["metric"] == metrics["value"]]
    actor_data = df[df["metric"] == metrics["actor"]]

    if value_data.empty or actor_data.empty:
        return df

    merged = pd.merge(
        value_data[["algorithm", "step", "iqm", "q25", "q75", "n_seeds", "n_values"]],
        actor_data[["algorithm", "step", "iqm", "q25", "q75"]],
        on=["algorithm", "step"],
        suffixes=("_value", "_actor"),
    )

    if merged.empty:
        return df

    total_data = merged[["algorithm", "step", "n_seeds", "n_values"]].copy()
    total_data["iqm"] = (merged["iqm_value"] + merged["iqm_actor"]) / 2
    total_data["q25"] = (merged["q25_value"] + merged["q25_actor"]) / 2
    total_data["q75"] = (merged["q75_value"] + merged["q75_actor"]) / 2
    total_data["metric"] = metrics["total"]

    return pd.concat([df, total_data], ignore_index=True)


def fetch_and_process_data(
    entity: str,
    project: str,
    group: str,
    metrics: Union[str, List[str]],
    combine_networks: bool = False,
) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group)

    # Handle single metric or list of metrics
    if isinstance(metrics, str):
        metrics = [metrics]

    # Expand combined network shortcuts
    expanded_metrics = []
    for metric in metrics:
        if metric == "dormant_neurons" and combine_networks:
            expanded_metrics.extend([
                "nn/value_dormant_neurons/total_ratio",
                "nn/actor_dormant_neurons/total_ratio",
            ])
        elif metric == "linearised_neurons" and combine_networks:
            expanded_metrics.extend([
                "nn/value_linearised_neurons/total_ratio",
                "nn/actor_linearised_neurons/total_ratio",
            ])
        else:
            expanded_metrics.append(metric)

    metrics = expanded_metrics

    algo_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    runs_per_algo = defaultdict(set)

    for run in runs:
        parts = run.name.split("_")
        if len(parts) >= 2:
            algo_name = "_".join(parts[:-1])
            seed_id = parts[-1]
        else:
            algo_name = run.name
            seed_id = "0"

        runs_per_algo[algo_name].add(seed_id)

        # Try to get history with all metrics
        history = run.history(keys=metrics + ["_step"], samples=5000)
        if history.empty:
            print(f"Warning: Run {run.name} has no data for any requested metrics")
            continue

        data_points = 0
        for _, row in history.iterrows():
            step = row.get("_step")
            if pd.notna(step):
                binned_step = int(round(step / 10_000) * 10_000)
                for metric in metrics:
                    value = row.get(metric)
                    if pd.notna(value):
                        algo_data[algo_name][metric][binned_step][seed_id].append(value)
                        data_points += 1

        if data_points > 0:
            print(f"  Run {run.name}: {data_points} data points across {len(metrics)} metrics")
        else:
            print(f"  Warning: Run {run.name} has no valid data points for any metric")

    print("\nUnique seeds per algorithm:")
    for algo, seeds in runs_per_algo.items():
        print(
            f"{algo}: {len(seeds)} seeds: {sorted(seeds)[:15]}{'>15' if len(seeds) > 15 else ''}"
        )

    data = []
    for algo_name, metric_dict in algo_data.items():
        for metric, step_dict in metric_dict.items():
            for step, seed_dict in step_dict.items():
                # Flatten all values from all seeds at this step
                all_values = []
                for seed_values in seed_dict.values():
                    all_values.extend(seed_values)

                if all_values:
                    values = np.array(all_values)
                    iqm_value = compute_iqm(values)

                    if len(values) > 1:
                        q25, q75 = np.percentile(values, [25, 75])
                    else:
                        q25, q75 = iqm_value, iqm_value

                    data.append({
                        "algorithm": algo_name,
                        "metric": metric,
                        "step": step / 1_000_000,
                        "iqm": iqm_value,
                        "q25": q25,
                        "q75": q75,
                        "n_seeds": len(seed_dict),  # Count unique seeds at this step
                        "n_values": len(all_values),  # Total values across seeds
                    })

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if df.empty:
        return df

    # Apply network combination if requested
    if combine_networks:
        for base_metric in ["dormant_neurons", "linearised_neurons"]:
            df = combine_network_metrics(df, base_metric)

    df = df.sort_values(["algorithm", "metric", "step"])

    # Print more detailed summary
    print("\nData summary:")
    for metric in df["metric"].unique():
        print(f"\nMetric: {metric}")
        metric_df = df[df["metric"] == metric]
        for algo in metric_df["algorithm"].unique():
            algo_df = metric_df[metric_df["algorithm"] == algo]
            max_seeds = algo_df["n_seeds"].max()
            total_points = len(algo_df)
            print(
                f"  - {algo}: up to {max_seeds} seeds, {total_points} aggregated data points"
            )

        # Provide quick stats that highlight how each algorithm performed overall
        algo_groups = metric_df.groupby("algorithm")
        if algo_groups.ngroups == 0:
            continue

        print("    Performance (IQM):")
        for algo, algo_df in algo_groups:
            iqm_series = algo_df["iqm"].dropna()
            if iqm_series.empty:
                print(f"      * {algo}: no IQM values available")
                continue

            avg_iqm = iqm_series.mean()
            avg_q25 = algo_df["q25"].dropna().mean()
            avg_q75 = algo_df["q75"].dropna().mean()
            peak_idx = iqm_series.idxmax()
            peak_row = algo_df.loc[peak_idx]
            peak_iqm = peak_row["iqm"]
            peak_step = peak_row["step"]
            peak_q25 = peak_row.get("q25")
            peak_q75 = peak_row.get("q75")
            final_idx = algo_df["step"].idxmax()
            final_row = algo_df.loc[final_idx]
            final_iqm = final_row["iqm"]
            final_step = final_row["step"]
            final_q25 = final_row.get("q25")
            final_q75 = final_row.get("q75")

            # Provide asymmetrical +/- distances from the final IQM to the interquartile bounds
            avg_lower = (
                max(avg_iqm - avg_q25, 0.0)
                if pd.notna(avg_q25)
                else 0.0
            )
            avg_upper = (
                max(avg_q75 - avg_iqm, 0.0)
                if pd.notna(avg_q75)
                else 0.0
            )
            peak_lower = (
                max(peak_iqm - peak_q25, 0.0)
                if pd.notna(peak_q25)
                else 0.0
            )
            peak_upper = (
                max(peak_q75 - peak_iqm, 0.0)
                if pd.notna(peak_q75)
                else 0.0
            )
            final_lower = (
                max(final_iqm - final_q25, 0.0)
                if pd.notna(final_q25)
                else 0.0
            )
            final_upper = (
                max(final_q75 - final_iqm, 0.0)
                if pd.notna(final_q75)
                else 0.0
            )

            avg_interval = f"{avg_iqm:.3f} (+{avg_upper:.3f}/-{avg_lower:.3f})"
            final_interval = f"{final_iqm:.3f} (+{final_upper:.3f}/-{final_lower:.3f})"
            peak_interval = f"{peak_iqm:.3f} (+{peak_upper:.3f}/-{peak_lower:.3f})"
            print(
                "      * "
                f"{algo}: avg={avg_interval}, final={final_interval} at {final_step:.1f}M steps, "
                f"peak={peak_interval} at {peak_step:.1f}M steps"
            )

    return df


def create_chart(
    df: pd.DataFrame,
    metrics: Union[str, List[str]],
    title: str = "",
    combine_networks: bool = False,
) -> alt.Chart:
    """Create a smoothed Altair chart with a line for IQM and a shaded area for IQR."""

    if isinstance(metrics, str):
        metrics = [metrics]

    # If we have multiple metrics, create subplots
    if len(df["metric"].unique()) > 1:
        charts = []

        for metric in df["metric"].unique():
            metric_df = df[df["metric"] == metric]

            # Create more descriptive titles for network metrics
            if "value_dormant_neurons" in metric:
                metric_name = "Value Network: Dormant Neuron Ratio"
            elif "actor_dormant_neurons" in metric:
                metric_name = "Actor Network: Dormant Neuron Ratio"
            elif "total_dormant_neurons" in metric:
                metric_name = "Combined Networks: Total Dormant Neuron Ratio"
            elif "value_linearised_neurons" in metric:
                metric_name = "Value Network: Linearized Neuron Ratio"
            elif "actor_linearised_neurons" in metric:
                metric_name = "Actor Network: Linearized Neuron Ratio"
            elif "total_linearised_neurons" in metric:
                metric_name = "Combined Networks: Total Linearized Neuron Ratio"
            elif "mean_episodic_return" in metric:
                metric_name = "Performance: Mean Episode Return"
            else:
                # Fallback to generic formatting
                metric_name = (
                    metric.split("/")[-1].replace("_", " ").title()
                    if "/" in metric
                    else metric.replace("_", " ").title()
                )

            base = alt.Chart(metric_df).encode(
                x=alt.X(
                    "step:Q",
                    title="Training Steps (Millions)",
                    scale=alt.Scale(domain=[0, 400], nice=True),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14),
                ),
                color=alt.Color(
                    "algorithm:N",
                    title="Algorithm",
                    legend=alt.Legend(
                        title=None,
                        symbolOpacity=1.0,
                        orient="bottom-left",
                        fillColor="rgba(255,255,255,1)",
                        strokeColor="gray",
                        padding=5,
                        cornerRadius=3,
                        labelFontSize=12,
                        symbolSize=150,
                    ),
                ),
            )

            # Apply smoothing
            smoothed_base = base.transform_window(
                frame=[-10, 10],
                groupby=["algorithm"],
                smooth_iqm="mean(iqm)",
                smooth_q25="mean(q25)",
                smooth_q75="mean(q75)",
            )

            # Create appropriate Y-axis label
            if "dormant_neurons" in metric or "linearised_neurons" in metric:
                y_label = "Ratio (0-1)"
            elif "mean_episodic_return" in metric:
                y_label = "Episode Return"
            else:
                y_label = metric_name

            # Create bands and lines
            bands = smoothed_base.mark_area(opacity=0.25).encode(
                y=alt.Y(
                    "smooth_q25:Q",
                    title=y_label,
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14),
                ),
                y2=alt.Y2("smooth_q75:Q", title=""),
            )

            lines = smoothed_base.mark_line(strokeWidth=2).encode(
                y=alt.Y("smooth_iqm:Q", title=""),
                tooltip=[
                    alt.Tooltip("algorithm:N", title="Algorithm"),
                    alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
                    alt.Tooltip("smooth_iqm:Q", title="Smoothed IQM", format=".3f"),
                    alt.Tooltip("n_seeds:Q", title="Seeds"),
                    alt.Tooltip("n_values:Q", title="Total Values"),
                ],
            )

            chart = (bands + lines).properties(
                width=800,
                height=300,
                title=alt.TitleParams(text=metric_name, fontSize=14, fontWeight="bold"),
            )

            charts.append(chart)

        # Combine charts vertically
        combined_chart = alt.vconcat(*charts).resolve_scale(color="independent")

        return (
            combined_chart.properties(
                title=alt.TitleParams(
                    text=title if title else "Network Metrics Comparison",
                    fontSize=16,
                    fontWeight="bold",
                )
            )
            .configure_axis(grid=True, gridOpacity=0.3, labelFontSize=12, titleFontSize=14)
            .interactive()
        )

    else:
        # Single metric - use original logic
        metric = df["metric"].iloc[0] if "metric" in df.columns else metrics[0]

        base = alt.Chart(df).encode(
            x=alt.X(
                "step:Q",
                title="Training Steps (Millions)",
                scale=alt.Scale(domain=[0, 400], nice=True),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16),
            ),
            color=alt.Color(
                "algorithm:N",
                title="Algorithm",
                legend=alt.Legend(
                    title=None,
                    symbolOpacity=1.0,
                    orient="bottom-left",
                    fillColor="rgba(255,255,255,1)",
                    strokeColor="gray",
                    padding=5,
                    cornerRadius=3,
                    labelFontSize=14,
                    symbolSize=200,
                ),
            ),
        )

    # Apply a rolling average transformation for smoothing
    smoothed_base = base.transform_window(
        frame=[-10, 10],
        groupby=["algorithm"],
        smooth_iqm="mean(iqm)",
        smooth_q25="mean(q25)",
        smooth_q75="mean(q75)",
    )

    # Create the shaded area using the SMOOTHED q25 and q75 values
    bands = smoothed_base.mark_area(opacity=0.25).encode(
        y=alt.Y(
            "smooth_q25:Q",
            title=metric.replace("_", " ").title(),
            axis=alt.Axis(labelFontSize=14, titleFontSize=16),
        ),
        y2=alt.Y2("smooth_q75:Q", title=""),
    )

    # Create the line chart using the SMOOTHED iqm value
    lines = smoothed_base.mark_line(strokeWidth=2).encode(
        y=alt.Y("smooth_iqm:Q", title=""),
        tooltip=[
            alt.Tooltip("algorithm:N", title="Algorithm"),
            alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
            alt.Tooltip("smooth_iqm:Q", title="Smoothed IQM", format=".2f"),
            alt.Tooltip("n_seeds:Q", title="Seeds"),
            alt.Tooltip("n_values:Q", title="Total Values"),
        ],
    )

    chart = bands + lines

    return (
        chart.properties(
            width=1000,
            height=400,
            title=alt.TitleParams(
                text=title
                if title
                else f"{metric.replace('_', ' ').title()} (IQM over Seeds)",
                fontSize=18,
                fontWeight="bold",
            ),
        )
        .configure_axis(grid=True, gridOpacity=0.3, labelFontSize=14, titleFontSize=16)
        .interactive()
    )


def main(
    wandb_entity: str,
    wandb_project: str = "crl_experiments",
    group: str = "default_group",
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    combine_networks: bool = False,
    output_dir: str = "./plots",
    ext: str = "png",
    debug: bool = False,
):
    if not metrics:
        metrics = [metric] if metric else ["eval_loss"]
    if len(metrics) == 1 and "," in metrics[0]:
        metrics = [m.strip() for m in metrics[0].split(",")]
    if combine_networks:
        metrics = [
            "dormant_neurons"
            if m == "dormant"
            else "linearised_neurons"
            if m in ["linearized", "linearised"]
            else m
            for m in metrics
        ]

    df = fetch_and_process_data(wandb_entity, wandb_project, group, metrics, combine_networks)
    if df.empty:
        return print(f"No data found for group: '{group}'")

    title = f"{group.replace('_', ' ').title()}: {'Network Metrics' if combine_networks else ', '.join(metrics)} (IQM)"
    save_suffix = (
        "combined_networks"
        if combine_networks
        else "_".join([m.replace("/", "_") for m in metrics])
    )
    chart = create_chart(df, metrics, title, combine_networks)

    save_path = Path(output_dir) / ext / f"{group}_{save_suffix}_iqm_smoothed.{ext}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"Chart saved to: {save_path}")
    return chart


if __name__ == "__main__":
    tyro.cli(main)
