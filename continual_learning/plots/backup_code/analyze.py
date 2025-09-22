#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
import re
from collections import defaultdict
from typing import Optional, Union, List, Tuple

from ablation_plot import parse_run_name, coerce_numeric_value


NETWORK_METRIC_COMBINATIONS = {
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


# HIDDEN_LAYER_SUFFIXES = [f"layer_{idx}_act" for idx in range(5)]


SRANK_LAYER_COMBINATIONS = {
    "value_srank_hidden": {
        "components": [f"nn/value_srank/{suffix}" for suffix in [f"layer_{idx}_act" for idx in range(5)]],
        "output_metric": "value_srank_hidden",
        "label": "Value Network: Hidden Layer S-Rank (Mean)",
        "reducer": np.mean,
    },
    "policy_srank_hidden": {
        "components": [f"nn/actor_srank/main/{suffix}" for suffix in [f"layer_{idx}_act" for idx in range(4)]],
        "output_metric": "policy_srank_hidden",
        "label": "Policy Network: Hidden Layer S-Rank (Mean)",
        "reducer": np.mean,
    },
}


CUSTOM_METRIC_TITLES = {
    combo["output_metric"]: combo["label"]
    for combo in SRANK_LAYER_COMBINATIONS.values()
}


def normalize_combined_metric_name(metric: str) -> str:
    """Canonicalize user-provided aliases for combined network metrics."""

    candidate = metric.strip()
    lowered = candidate.lower()
    if lowered in {"dormant", "dormant_neurons"}:
        return "dormant_neurons"
    if lowered in {
        "linearised",
        "linearised_neurons",
        "linearized",
        "linearized_neurons",
    }:
        return "linearised_neurons"
    return candidate


def build_combined_metric_suffix(metric: str) -> str:
    """Return a descriptive filename stem for combined network plots."""

    normalized = normalize_combined_metric_name(metric).lower()
    if normalized == "dormant_neurons":
        base = "dormant"
    elif normalized == "linearised_neurons":
        base = "linearised"
    else:
        base = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "metric"
    return f"{base}_combined"


def resolve_metric_labels(metric: str) -> Tuple[str, str]:
    """Return a human-readable title and y-axis label for a metric key."""

    if metric in CUSTOM_METRIC_TITLES:
        metric_name = CUSTOM_METRIC_TITLES[metric]
    elif "value_dormant_neurons" in metric:
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
        metric_name = "Mean Episode Return"
    else:
        metric_name = (
            metric.split("/")[-1].replace("_", " ").title()
            if "/" in metric
            else metric.replace("_", " ").title()
        )

    if "dormant_neurons" in metric or "linearised_neurons" in metric:
        y_label = "Ratio (0-1)"
    elif "srank" in metric:
        y_label = "S-Rank"
    elif "mean_episodic_return" in metric:
        y_label = "Episode Return"
    else:
        y_label = metric_name

    return metric_name, y_label


def format_split_value(value) -> str:
    """Format parameter values for display in legends."""

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return stripped
        try:
            numeric_value = int(stripped)
            return str(numeric_value)
        except ValueError:
            try:
                numeric_value = float(stripped)
                return format_split_value(numeric_value)
            except ValueError:
                return stripped

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if value == 0:
            return "0"
        abs_value = abs(float(value))
        if abs_value >= 1:
            formatted = f"{value:.2f}"
        elif abs_value >= 0.01:
            formatted = f"{value:.3f}"
        else:
            formatted = f"{value:.2e}"
        return formatted.rstrip("0").rstrip(".")

    return str(value)


def get_split_label(run, split_by: Optional[str]) -> Optional[str]:
    """Extract a readable split label from run config/name for grouping."""

    if not split_by:
        return None

    config_obj = getattr(run, "config", None)
    if config_obj is not None:
        try:
            value = config_obj.get(split_by)
        except Exception:
            value = None
        if value is not None:
            return format_split_value(value)

    run_name = getattr(run, "name", "") or ""
    if run_name:
        try:
            parsed = parse_run_name(run_name)
        except Exception:
            parsed = {}
        if split_by in parsed:
            return format_split_value(parsed[split_by])

    summary_obj = getattr(run, "summary", None)
    if summary_obj is not None:
        try:
            value = summary_obj.get(split_by)
        except Exception:
            value = None
        if value is not None:
            return format_split_value(value)

    return None


def normalize_algorithm_name(name: str) -> str:
    def _seed_repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return f"{prefix}seed=*"

    normalized = name or ""
    normalized = re.sub(r'(^|[,_])seed=[^,_]+', _seed_repl, normalized)
    normalized = re.sub(r'_s\d+$', '', normalized)
    normalized = re.sub(r'[,_-]+$', '', normalized)
    return normalized or "unknown_algorithm"


def extract_run_identity(run) -> Tuple[str, str]:
    run_name = getattr(run, "name", "") or ""
    algo_name = run_name
    seed_id: Optional[str] = None

    config_obj = getattr(run, "config", None)
    if config_obj is not None:
        try:
            config_seed = config_obj.get("seed")  # type: ignore[attr-defined]
        except Exception:
            config_seed = None
        if config_seed is not None:
            seed_id = str(config_seed)

    parts = run_name.split("_") if run_name else []
    if parts and parts[-1].isdigit():
        if seed_id is None:
            seed_id = parts[-1]
        algo_name = "_".join(parts[:-1])
    else:
        match = re.search(r'_s(\d+)$', run_name)
        if match:
            if seed_id is None:
                seed_id = match.group(1)
            algo_name = run_name[: match.start()]

    algo_name = normalize_algorithm_name(algo_name or run_name or getattr(run, "group", ""))

    if seed_id is None:
        seed_id = getattr(run, "id", None)
    if seed_id is None:
        seed_id = "0"

    return algo_name, str(seed_id)


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
    if base_metric not in NETWORK_METRIC_COMBINATIONS:
        return df
    metrics = NETWORK_METRIC_COMBINATIONS[base_metric]

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
    split_by: Optional[str] = None,
) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group)

    # Handle single metric or list of metrics
    if isinstance(metrics, str):
        metrics = [metrics]

    def append_unique(sequence: List[str], item: str) -> None:
        if item not in sequence:
            sequence.append(item)

    metrics_to_track: List[str] = []
    requested_metrics: List[str] = []
    layer_combinations: List[dict] = []
    seen_layer_aliases = set()

    for metric in metrics:
        alias_info = SRANK_LAYER_COMBINATIONS.get(metric)
        if alias_info is not None:
            if metric not in seen_layer_aliases:
                alias_record = dict(alias_info)
                components = list(alias_record.get("components", []))
                alias_record["alias"] = metric
                alias_record["components"] = components
                layer_combinations.append(alias_record)
                seen_layer_aliases.add(metric)
                for component_metric in components:
                    append_unique(metrics_to_track, component_metric)
            append_unique(requested_metrics, alias_info.get("output_metric", metric))
            continue

        if combine_networks and metric in NETWORK_METRIC_COMBINATIONS:
            combo = NETWORK_METRIC_COMBINATIONS[metric]
            append_unique(metrics_to_track, combo["value"])
            append_unique(metrics_to_track, combo["actor"])
            append_unique(requested_metrics, combo["total"])
            continue

        append_unique(metrics_to_track, metric)
        append_unique(requested_metrics, metric)

    metrics = metrics_to_track

    algo_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    runs_per_algo = defaultdict(set)

    missing_split_runs = []

    for run in runs:
        algo_name, seed_id = extract_run_identity(run)

        split_label = get_split_label(run, split_by)
        if split_by and split_label:
            algo_label = f"{algo_name}, {split_by}={split_label}"
        else:
            algo_label = algo_name
            if split_by:
                missing_split_runs.append(getattr(run, "name", algo_name))

        runs_per_algo[algo_label].add(seed_id)

        history_keys = metrics + ["_step"] if metrics else ["_step"]
        history = run.history(keys=history_keys, samples=5000)
        if history.empty:
            print(f"Warning: Run {run.name} has no data for any requested metrics")
            continue

        data_points = 0
        for _, row in history.iterrows():
            step = row.get("_step")
            if pd.isna(step):
                continue

            binned_step = int(round(step / 10_000) * 10_000)
            for metric in metrics:
                numeric_value = coerce_numeric_value(row.get(metric))
                if numeric_value is None:
                    continue
                algo_data[algo_label][metric][binned_step][seed_id].append(numeric_value)
                data_points += 1

            if layer_combinations:
                for combo in layer_combinations:
                    components = combo.get("components", [])
                    if not components:
                        continue
                    available_values = []
                    for component_metric in components:
                        component_value = coerce_numeric_value(row.get(component_metric))
                        if component_value is not None:
                            available_values.append(component_value)
                    if not available_values:
                        continue

                    reducer = combo.get("reducer", np.mean)
                    combined_value = float(reducer(available_values))
                    output_metric = combo.get("output_metric", combo["alias"])
                    algo_data[algo_label][output_metric][binned_step][seed_id].append(
                        combined_value
                    )
                    data_points += 1

        if data_points > 0:
            combo_count = sum(1 for combo in layer_combinations if combo.get("components"))
            tracked_metric_count = len(metrics) + combo_count
            print(
                f"  Run {run.name}: {data_points} data points across {tracked_metric_count} metrics"
            )
        else:
            print(f"  Warning: Run {run.name} has no valid data points for any metric")

    print("\nUnique seeds per algorithm:")
    if split_by and missing_split_runs:
        unique_missing = sorted(set(missing_split_runs))
        print(
            "\nWarning: missing split values for runs (falling back to base algorithm name):"
        )
        for run_name in unique_missing[:10]:
            print(f"  - {run_name}")
        if len(unique_missing) > 10:
            print(f"  ... {len(unique_missing) - 10} more")

    for algo, seeds in runs_per_algo.items():
        print(
            f"{algo}: {len(seeds)} seeds: {sorted(seeds)[:15]}{'>15' if len(seeds) > 15 else ''}"
        )

    data = []
    for algo_label, metric_dict in algo_data.items():
        for metric, step_dict in metric_dict.items():
            for step, seed_dict in step_dict.items():
                # Flatten all values from all seeds at this step
                all_values = []
                for seed_values in seed_dict.values():
                    all_values.extend(seed_values)

                if all_values:
                    try:
                        values = np.asarray(all_values, dtype=np.float64)
                    except (TypeError, ValueError):
                        continue
                    try:
                        iqm_value = compute_iqm(values)
                    except Exception as e:
                        print(e)
                        breakpoint()

                    if len(values) > 1:
                        q25, q75 = np.percentile(values, [25, 75])
                    else:
                        q25, q75 = iqm_value, iqm_value

                    data.append({
                        "algorithm": algo_label,
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

    if requested_metrics:
        requested_metric_set = set(requested_metrics)
        df = df[df["metric"].isin(requested_metric_set)]
        if df.empty:
            return df

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
            metric_name, y_label = resolve_metric_labels(metric)

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
        metric_name, y_label = resolve_metric_labels(metric)

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
            title=y_label,
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
                else f"{metric_name} (IQM over Seeds)",
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
    split_by: Optional[str] = None,
    output_dir: str = "./plots",
    ext: str = "png",
    debug: bool = False,
):
    if not metrics:
        metrics = [metric] if metric else ["eval_loss"]

    if len(metrics) == 1 and "," in metrics[0]:
        metrics = [m.strip() for m in metrics[0].split(",") if m.strip()]
    else:
        metrics = [m.strip() for m in metrics if m and m.strip()]

    if not metrics:
        raise ValueError("No metric provided; supply --metric or --metrics with a single value.")

    if len(metrics) != 1:
        raise ValueError(
            "analyze.py now expects exactly one metric per invocation; received "
            f"{len(metrics)} metrics ({metrics})."
        )

    selected_metric = metrics[0]
    combined_total_metric: Optional[str] = None

    if combine_networks:
        normalized_metric = normalize_combined_metric_name(selected_metric)
        combo = NETWORK_METRIC_COMBINATIONS.get(normalized_metric)
        if combo is None:
            available = ", ".join(sorted(NETWORK_METRIC_COMBINATIONS))
            raise ValueError(
                "combine_networks is only defined for the metrics "
                f"[{available}]; received '{selected_metric}'."
            )
        metrics = [normalized_metric]
        combined_total_metric = combo["total"]
    else:
        metrics = [selected_metric]

    df = fetch_and_process_data(
        wandb_entity,
        wandb_project,
        group,
        metrics,
        combine_networks,
        split_by,
    )
    if df.empty:
        return print(f"No data found for group: '{group}'")

    if combine_networks:
        metric_label = (
            resolve_metric_labels(combined_total_metric)[0]
            if combined_total_metric is not None
            else "Combined Network Metric"
        )
        title_suffix = metric_label
    else:
        readable_metrics = [resolve_metric_labels(m)[0] for m in metrics]
        title_suffix = ", ".join(readable_metrics)

    title = f"{group.replace('_', ' ').title()}: {title_suffix} (IQM)"
    if split_by:
        pretty_split = split_by.replace("_", " ")
        title += f" • Split by {pretty_split.title()}"
    save_suffix = (
        build_combined_metric_suffix(metrics[0])
        if combine_networks
        else "_".join([m.replace("/", "_") for m in metrics])
    )
    if split_by:
        save_suffix = f"{save_suffix}_split_by_{split_by}"
    chart = create_chart(df, metrics, title, combine_networks)

    save_path = Path(output_dir) / ext / f"{group}_{save_suffix}_iqm_smoothed.{ext}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"Chart saved to: {save_path}")
    return chart


if __name__ == "__main__":
    tyro.cli(main)
