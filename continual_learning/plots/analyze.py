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

from ablation_plot import (
    parse_run_name,
    coerce_numeric_value,
    ALT_FONT_FAMILY,
    build_algorithm_legend_domain,
    scaled_font_size,
)


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
    "srank_hidden": {
        "value": "value_srank_hidden",
        "actor": "actor_srank_hidden",
        "total": "total_srank_hidden",
    },
    "gradient_norm": {
        "value": "nn/vf_gradient_norm",
        "actor": "nn/policy_gradient_norm",
        "total": "nn/total_gradient_norm",
    },
    "parameter_norm": {
        "value": "nn/vf_parameter_norm",
        "actor": "nn/policy_parameter_norm",
        "total": "nn/total_parameter_norm",
    },
}


# HIDDEN_LAYER_SUFFIXES = [f"layer_{idx}_act" for idx in range(5)]


SRANK_LAYER_COMBINATIONS = {
    "value_srank_hidden": {
        "components": [
            f"nn/value_srank/{suffix}"
            for suffix in [f"layer_{idx}_act" for idx in range(5)]
        ]
        + ["nn/value_srank/output_act"],
        "output_metric": "value_srank_hidden",
        "label": "Value Network: Hidden Layer S-Rank (Mean)",
        "reducer": np.mean,
    },
    "actor_srank_hidden": {
        "components": [
            f"nn/actor_srank/main/{suffix}"
            for suffix in [f"layer_{idx}_act" for idx in range(4)]
        ]
        + ["nn/actor_srank/main/output_act"],
        "output_metric": "actor_srank_hidden",
        "label": "Actor Network: Hidden Layer S-Rank (Mean)",
        "reducer": np.mean,
    },
}


CUSTOM_METRIC_TITLES = {
    combo["output_metric"]: combo["label"]
    for combo in SRANK_LAYER_COMBINATIONS.values()
}
CUSTOM_METRIC_TITLES["total_srank_hidden"] = "Combined Networks: Hidden Layer S-Rank (Mean)"
CUSTOM_METRIC_TITLES["nn/total_dormant_neurons/total_ratio_normalized"] = (
    "Combined Networks: Dormant Neuron Ratio (Balanced)"
)
CUSTOM_METRIC_TITLES["nn/total_linearised_neurons/total_ratio_normalized"] = (
    "Combined Networks: Linearized Neuron Ratio (Balanced)"
)
CUSTOM_METRIC_TITLES["total_srank_hidden_normalized"] = (
    "Combined Networks: Hidden Layer S-Rank (Balanced)"
)
CUSTOM_METRIC_TITLES["nn/total_gradient_norm"] = "Combined Networks: Gradient Norm"
CUSTOM_METRIC_TITLES["nn/total_gradient_norm_normalized"] = (
    "Combined Networks: Gradient Norm (Balanced)"
)


METRIC_ALIASES = {
    "policy_srank_hidden": "actor_srank_hidden",
    "grad_norm": "gradient_norm",
}

ALGORITHM_DISPLAY_NAMES = {
    # Ant
    "redo": "ReDo",
    "regrama": "ReGraMa",
    "cbp": "CBP",
    "ccbp": "CPR",
    "shrink_and_perturb": "Shrink & Perturb",
    "soft_shrink_and_perturb": "Soft Shrink & Perturb",
    "adam": "Adam",
    "standard": "Adam",
    # Humanoid
    "ccbp_bigger_rollout_new_hparams": "CPR",
    "regrama_bigger_rollout_new_hparams": "ReGraMa",
    "cbp_bigger_rollout": "CBP",
    "redo_bigger_rollout": "ReDo",
    "shrink_and_perturb_bigger_rollout": "Shrink & Perturb",
    "soft_shrink_and_perturb_bigger_rollout": "Soft Shrink & Perturb",
    "shrink_and_perturb_br_adam_sb_lr_smaller_net": "Soft Shrink & Perturb",
    "shrink_and_perturb_smaller": "Soft Shrink & Perturb",
    "standard_bigger_rollout": "Adam"
}


def canonicalize_metric_name(metric: str) -> str:
    stripped = metric.strip()
    canonical = METRIC_ALIASES.get(stripped.lower())
    return canonical if canonical is not None else stripped


def combined_total_metric_key(base_metric: str, normalized: bool) -> str:
    combo = NETWORK_METRIC_COMBINATIONS.get(base_metric)
    if not combo:
        return base_metric
    total_key = combo["total"]
    return f"{total_key}_normalized" if normalized else total_key


def normalize_combined_metric_name(metric: str) -> str:
    """Canonicalize user-provided aliases for combined network metrics."""

    candidate = metric.strip()
    canonical = canonicalize_metric_name(candidate)
    lowered = canonical.lower()
    if lowered in {"dormant", "dormant_neurons"}:
        return "dormant_neurons"
    if lowered in {
        "linearised",
        "linearised_neurons",
        "linearized",
        "linearized_neurons",
    }:
        return "linearised_neurons"
    if lowered in {
        "srank_hidden",
        "actor_srank_hidden",
        "value_srank_hidden",
    }:
        return "srank_hidden"
    if lowered in {"gradient_norm", "grad_norm", "gradientnorm", "gradnorm"}:
        return "gradient_norm"
    return canonical


def build_combined_metric_suffix(metric: str, balanced: bool = False) -> str:
    """Return a descriptive filename stem for combined network plots."""

    normalized_metric = normalize_combined_metric_name(metric).lower()
    if normalized_metric == "dormant_neurons":
        base = "dormant"
    elif normalized_metric == "linearised_neurons":
        base = "linearised"
    else:
        base = re.sub(r"[^a-z0-9]+", "_", normalized_metric).strip("_") or "metric"
    suffix = "balanced_combined" if balanced else "combined"
    return f"{base}_{suffix}"


def slugify_title(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized.lower() or "plot"


def resolve_metric_labels(metric: str) -> Tuple[str, str]:
    """Return a human-readable title and y-axis label for a metric key."""

    # Remove _ci suffix if present (confidence interval notation)
    metric_clean = metric.replace("_ci", "").replace("_CI", "")
    normalized_variant = metric.endswith("_normalized")

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
    elif "total_gradient_norm" in metric:
        metric_name = "Combined Networks: Gradient Norm"
    elif "vf_gradient_norm" in metric:
        metric_name = "Value Network: Gradient Norm"
    elif "policy_gradient_norm" in metric:
        metric_name = "Actor Network: Gradient Norm"
    elif "mean_episodic_return" in metric:
        metric_name = "Mean Episode Return"
    elif "eval_accuracy" in metric_clean:
        metric_name = "Mean Evaluation Accuracy"
    else:
        # Use cleaned metric for display to avoid "Ci" suffix
        metric_name = (
            metric_clean.split("/")[-1].replace("_", " ").title()
            if "/" in metric_clean
            else metric_clean.replace("_", " ").title()
        )

    if "dormant_neurons" in metric or "linearised_neurons" in metric:
        if normalized_variant:
            y_label = "Balanced Ratio (Z-Score)"
        else:
            y_label = "Percentage"
    elif "srank" in metric:
        y_label = "Balanced S-Rank (Z-Score)" if normalized_variant else "S-Rank"
    elif "gradient_norm" in metric:
        y_label = "Balanced Gradient Norm (Z-Score)" if normalized_variant else "Gradient Norm"
    elif "mean_episodic_return" in metric:
        y_label = "Episode Return"
    elif "eval_accuracy" in metric:
        y_label = "Evaluation Accuracy"
    else:
        y_label = "Balanced Score (Z-Score)" if normalized_variant else metric_name

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
    # Strip -copy suffix (from wandb run copies)
    normalized = re.sub(r'-copy$', '', normalized)
    normalized = re.sub(r'[,_-]+$', '', normalized)
    return normalized or "unknown_algorithm"


def display_algorithm_name(name: str) -> str:
    label = name.strip()
    return ALGORITHM_DISPLAY_NAMES.get(label.lower(), label)


def extract_run_identity(run) -> Tuple[str, str]:
    run_name = getattr(run, "name", "") or ""
    # Strip -copy suffix early (from wandb run copies) before seed extraction
    run_name = re.sub(r'-copy$', '', run_name)
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


def fetch_runs(entity: str, project: str, group: str, name_pattern: Optional[str] = None):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
    finished = [r for r in runs if r.state == "finished"]

    # Filter by name pattern if provided
    if name_pattern:
        import fnmatch
        finished = [r for r in finished if fnmatch.fnmatch(r.name, name_pattern)]
        print(f"Fetched {len(runs)} runs, {len(finished)} finished matching pattern '{name_pattern}'")
    else:
        print(f"Fetched {len(runs)} runs, {len(finished)} finished")

    return finished


def combine_network_metrics(
    df: pd.DataFrame,
    base_metric: str,
    normalize_networks: bool = False,
) -> pd.DataFrame:
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

    total_metric_key = combined_total_metric_key(base_metric, normalize_networks)

    total_data = merged[["algorithm", "step", "n_seeds", "n_values"]].copy()

    if not normalize_networks:
        total_data["iqm"] = (merged["iqm_value"] + merged["iqm_actor"]) / 2
        total_data["q25"] = (merged["q25_value"] + merged["q25_actor"]) / 2
        total_data["q75"] = (merged["q75_value"] + merged["q75_actor"]) / 2
    else:
        def _norm_stats(series: pd.Series) -> Tuple[float, float]:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return 0.0, 1.0
            mean = float(numeric.mean())
            std = float(numeric.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            return mean, std

        value_mean, value_std = _norm_stats(value_data["iqm"])
        actor_mean, actor_std = _norm_stats(actor_data["iqm"])

        for prefix, mean, std in (
            ("value", value_mean, value_std),
            ("actor", actor_mean, actor_std),
        ):
            merged[f"iqm_{prefix}_scaled"] = (merged[f"iqm_{prefix}"] - mean) / std
            merged[f"q25_{prefix}_scaled"] = (merged[f"q25_{prefix}"] - mean) / std
            merged[f"q75_{prefix}_scaled"] = (merged[f"q75_{prefix}"] - mean) / std

        total_data["iqm"] = (
            merged["iqm_value_scaled"] + merged["iqm_actor_scaled"]
        ) / 2
        total_data["q25"] = (
            merged["q25_value_scaled"] + merged["q25_actor_scaled"]
        ) / 2
        total_data["q75"] = (
            merged["q75_value_scaled"] + merged["q75_actor_scaled"]
        ) / 2

    total_data["metric"] = total_metric_key

    combined_df = pd.concat([df, total_data], ignore_index=True)
    if df.attrs:
        combined_df.attrs.update(df.attrs)
    if normalize_networks:
        normalized_metrics = set(df.attrs.get("normalized_metrics", []))
        normalized_metrics.add(total_metric_key)
        combined_df.attrs["normalized_metrics"] = sorted(normalized_metrics)
    return combined_df


def fetch_and_process_data(
    entity: str,
    project: str,
    group: str,
    metrics: Union[str, List[str]],
    combine_networks: bool = False,
    normalize_networks: bool = False,
    split_by: Optional[str] = None,
    algorithm_name: Optional[str] = None,
    name_pattern: Optional[str] = None,
) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group, name_pattern)

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

    def register_layer_combination(metric_name: str) -> Optional[str]:
        alias_info = SRANK_LAYER_COMBINATIONS.get(metric_name)
        if alias_info is None:
            return None

        output_metric = alias_info.get("output_metric", metric_name)
        if metric_name in seen_layer_aliases:
            return output_metric

        alias_record = dict(alias_info)
        components = list(alias_record.get("components", []))
        alias_record["alias"] = metric_name
        alias_record["components"] = components
        layer_combinations.append(alias_record)
        seen_layer_aliases.add(metric_name)

        for component_metric in components:
            append_unique(metrics_to_track, component_metric)

        return output_metric

    for metric in metrics:
        alias_output_metric = register_layer_combination(metric)
        if alias_output_metric is not None:
            append_unique(requested_metrics, alias_output_metric)
            continue

        if combine_networks and metric in NETWORK_METRIC_COMBINATIONS:
            combo = NETWORK_METRIC_COMBINATIONS[metric]
            value_alias_output = register_layer_combination(combo["value"])
            actor_alias_output = register_layer_combination(combo["actor"])
            if value_alias_output is None:
                append_unique(metrics_to_track, combo["value"])
            if actor_alias_output is None:
                append_unique(metrics_to_track, combo["actor"])
            append_unique(
                requested_metrics,
                combined_total_metric_key(metric, normalize_networks),
            )
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
        # Use algorithm_name override if provided, otherwise extract from run
        if algorithm_name:
            display_name = algorithm_name
            # Still need to extract seed for grouping
            _, seed_id = extract_run_identity(run)
        else:
            algo_name, seed_id = extract_run_identity(run)
            display_name = display_algorithm_name(algo_name)

        split_label = get_split_label(run, split_by)
        if split_by and split_label:
            algo_label = f"{display_name}, {split_by}={split_label}"
        else:
            algo_label = display_name
            if split_by:
                missing_split_runs.append(getattr(run, "name", algorithm_name or algo_name))

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
                metric_key = metric.lower()
                if isinstance(numeric_value, (int, float, np.integer, np.floating)):
                    if "episodic_return" in metric_key or "episode_return" in metric_key:
                        numeric_value = max(float(numeric_value), -2000.0)
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
        for base_metric in NETWORK_METRIC_COMBINATIONS.keys():
            df = combine_network_metrics(df, base_metric, normalize_networks)

    if requested_metrics:
        requested_metric_set = set(requested_metrics)
        df = df[df["metric"].isin(requested_metric_set)]
        if df.empty:
            return df

    df = df.sort_values(["algorithm", "metric", "step"])

    # Print more detailed summary
    print("\nData summary:")
    performance_summary = []
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

            performance_summary.append(
                {
                    "metric": metric,
                    "algorithm": algo,
                    "avg_iqm": avg_iqm,
                    "avg_q25": avg_q25,
                    "avg_q75": avg_q75,
                    "peak_iqm": peak_iqm,
                    "peak_step": peak_step,
                    "peak_q25": peak_q25,
                    "peak_q75": peak_q75,
                    "final_iqm": final_iqm,
                    "final_step": final_step,
                    "final_q25": final_q25,
                    "final_q75": final_q75,
                    "n_points": len(algo_df),
                }
            )

    if performance_summary:
        df.attrs["performance_summary"] = performance_summary

    return df


def get_algorithm_render_order(algorithm: str) -> int:
    """Determine rendering order for algorithms. Higher values are drawn later (on top).

    CPR should always be drawn last to appear in the foreground.
    """
    algo_upper = algorithm.upper()
    if "CCBP" in algo_upper or "CPR" in algo_upper:
        return 1000  # Draw CPR last (foreground)
    return 0  # Draw other algorithms first (background)


def create_chart(
    df: pd.DataFrame,
    metrics: Union[str, List[str]],
    title: str = "",
    combine_networks: bool = False,
    show_iqr: bool = True,
    base_text_size: float = 25.0,
    chart_width: Optional[int] = None,
    chart_height: Optional[int] = None,
    line_width: float = 2.0,
    y_tick_count: Optional[int] = None,
    x_axis_max: float = 400.0,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    log_scale: bool = False,
) -> alt.Chart:
    """Create a smoothed Altair chart with IQM lines and optional shaded IQR.

    Note: log_scale requires all y-values to be positive. Negative or zero values
    may cause rendering issues or be excluded from the visualization.
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    # Sort dataframe to ensure CPR is drawn last (appears in foreground)
    # Higher render_order values are drawn later (on top)
    df = df.copy()
    df["_render_order"] = df["algorithm"].apply(get_algorithm_render_order)
    df = df.sort_values(["_render_order", "algorithm", "metric", "step"], ascending=[True, True, True, True])
    df = df.drop(columns=["_render_order"])

    axis_label_large = scaled_font_size(base_text_size, 1.5)
    axis_label_default = scaled_font_size(base_text_size, 1.0)
    # axis_label_small = scaled_font_size(base_text_size, 0.7)
    axis_title_large = scaled_font_size(base_text_size, 1.5)
    # axis_title_small = scaled_font_size(base_text_size, 0.8)
    legend_label_large = scaled_font_size(base_text_size, 1.1)
    # legend_label_small = scaled_font_size(base_text_size, 0.7)
    legend_title_large = scaled_font_size(base_text_size, 1.5)
    legend_symbol_large = scaled_font_size(base_text_size, 50)
    # legend_symbol_small = scaled_font_size(base_text_size, 7.5)
    legend_symbol_stroke_large = max(2, scaled_font_size(base_text_size, 0.5))
    legend_symbol_stroke_small = max(2, scaled_font_size(base_text_size, 0.35))
    legend_label_limit = max(220, int(round(base_text_size * 12)))
    chart_title_small = scaled_font_size(base_text_size, 0.7) # bar chart only
    chart_title_medium = scaled_font_size(base_text_size, 0.8)
    chart_title_large = scaled_font_size(base_text_size, 1.6)
    default_padding = {"right": 25}

    # If we have multiple metrics, create subplots
    if len(df["metric"].unique()) > 1:
        charts = []

        for metric in df["metric"].unique():
            metric_df = df[df["metric"] == metric]
            metric_name, y_label = resolve_metric_labels(metric)
            legend_domain = build_algorithm_legend_domain(metric_df["algorithm"])
            color_scale = (
                alt.Scale(domain=legend_domain)
                if legend_domain
                else alt.Undefined
            )

            base = alt.Chart(metric_df).encode(
                x=alt.X(
                    "step:Q",
                    title="Training Steps (Millions)",
                    scale=alt.Scale(domain=[0, x_axis_max], nice=True),
                    axis=alt.Axis(
                        labelFontSize=axis_label_large,
                        titleFontSize=axis_title_large,
                        labelFont=ALT_FONT_FAMILY,
                        titleFont=ALT_FONT_FAMILY,
                    ),
                ),
                color=alt.Color(
                    "algorithm:N",
                    title="Algorithm",
                    scale=color_scale,
                    legend=alt.Legend(
                        title=None,
                        symbolOpacity=1.0,
                        orient="bottom-left",
                        fillColor="rgba(255,255,255,1)",
                        strokeColor="gray",
                        padding=5,
                        cornerRadius=3,
                        labelFontSize=legend_label_large,
                        labelFontWeight="bold",
                        titleFontSize=legend_title_large,
                        titleFontWeight="bold",
                        symbolSize=legend_symbol_large,
                        symbolStrokeWidth=legend_symbol_stroke_small,
                        labelLimit=legend_label_limit,
                        labelPadding=8,
                        labelFont=ALT_FONT_FAMILY,
                        titleFont=ALT_FONT_FAMILY,
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
            axis_kwargs = {
                "labelFontSize": axis_label_default,
                "titleFontSize": axis_title_large,
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            }
            if y_tick_count is not None:
                axis_kwargs["tickCount"] = max(2, int(y_tick_count))
            axis_config = alt.Axis(**axis_kwargs)

            # Configure y-axis scale
            y_scale_kwargs = {}
            if log_scale:
                y_scale_kwargs["type"] = "log"
            if y_min is not None and y_max is not None:
                # Both bounds specified
                y_scale_kwargs["domain"] = [y_min, y_max]
            elif y_min is not None:
                # Only min specified
                y_scale_kwargs["domainMin"] = y_min
            elif y_max is not None:
                # Only max specified
                y_scale_kwargs["domainMax"] = y_max
            y_scale_config = alt.Scale(**y_scale_kwargs) if y_scale_kwargs else alt.Undefined

            if show_iqr:
                bands = smoothed_base.mark_area(opacity=0.25).encode(
                    y=alt.Y(
                        "smooth_q25:Q",
                        title=y_label,
                        axis=axis_config,
                        scale=y_scale_config,
                    ),
                    y2=alt.Y2("smooth_q75:Q", title=""),
                )

            lines = smoothed_base.mark_line(strokeWidth=line_width).encode(
                y=alt.Y(
                    "smooth_iqm:Q",
                    title="" if show_iqr else y_label,
                    axis=axis_config,
                    scale=y_scale_config,
                ),
                tooltip=[
                    alt.Tooltip("algorithm:N", title="Algorithm"),
                    alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
                    alt.Tooltip("smooth_iqm:Q", title="Smoothed IQM", format=".3f"),
                    alt.Tooltip("n_seeds:Q", title="Seeds"),
                    alt.Tooltip("n_values:Q", title="Total Values"),
                ],
            )

            chart = lines
            if show_iqr:
                chart = bands + chart

            chart = chart.properties(
                width=chart_width if chart_width is not None else 800,
                height=chart_height if chart_height is not None else 300,
                title=alt.TitleParams(
                    text=metric_name,
                    fontSize=chart_title_small,
                    fontWeight="bold",
                    font=ALT_FONT_FAMILY,
                ),
                padding=default_padding,
            )

            charts.append(chart)

        # Combine charts vertically
        combined_chart = alt.vconcat(*charts).resolve_scale(color="independent")

        return (
            combined_chart.properties(
                title=alt.TitleParams(
                    text=title if title else "Network Metrics Comparison",
                    fontSize=chart_title_medium,
                    fontWeight="bold",
                    font=ALT_FONT_FAMILY,
                ),
                padding=default_padding,
            )
            .configure_axis(
                grid=True,
                gridOpacity=0.8,
                labelFontSize=axis_label_default,
                titleFontSize=axis_title_large,
                labelFont=ALT_FONT_FAMILY,
                titleFont=ALT_FONT_FAMILY,
            )
            .configure_legend(labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)
            .interactive()
        )

    else:
        # Single metric - use original logic
        metric = df["metric"].iloc[0] if "metric" in df.columns else metrics[0]
        metric_name, y_label = resolve_metric_labels(metric)
        legend_domain = build_algorithm_legend_domain(df["algorithm"])
        color_scale = (
            alt.Scale(domain=legend_domain)
            if legend_domain
            else alt.Undefined
        )

        base = alt.Chart(df).encode(
            x=alt.X(
                "step:Q",
                title="Training Steps (Millions)",
                scale=alt.Scale(domain=[0, x_axis_max], nice=True),
                axis=alt.Axis(
                    labelFontSize=axis_label_large,
                    titleFontSize=axis_title_large,
                    labelFont=ALT_FONT_FAMILY,
                    titleFont=ALT_FONT_FAMILY,
                ),
            ),
            color=alt.Color(
                "algorithm:N",
                title="Algorithm",
                scale=color_scale,
                legend=alt.Legend(
                    title=None,
                    symbolOpacity=1.0,
                    orient="bottom-left",
                    fillColor="rgba(255,255,255,1)",
                    strokeColor="gray",
                    padding=5,
                    cornerRadius=3,
                    labelFontSize=legend_label_large,
                    labelFontWeight="bold",
                    symbolSize=legend_symbol_large,
                    symbolStrokeWidth=legend_symbol_stroke_large,
                    labelLimit=legend_label_limit,
                    labelPadding=8,
                    labelFont=ALT_FONT_FAMILY,
                    titleFont=ALT_FONT_FAMILY,
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
    axis_kwargs = {
        "labelFontSize": axis_label_large,
        "titleFontSize": axis_title_large,
        "labelFont": ALT_FONT_FAMILY,
        "titleFont": ALT_FONT_FAMILY,
    }
    if y_tick_count is not None:
        axis_kwargs["tickCount"] = max(2, int(y_tick_count))
    axis_config = alt.Axis(**axis_kwargs)

    # Configure y-axis scale
    y_scale_kwargs = {}
    if log_scale:
        y_scale_kwargs["type"] = "log"
    if y_min is not None and y_max is not None:
        # Both bounds specified
        y_scale_kwargs["domain"] = [y_min, y_max]
    elif y_min is not None:
        # Only min specified
        y_scale_kwargs["domainMin"] = y_min
    elif y_max is not None:
        # Only max specified
        y_scale_kwargs["domainMax"] = y_max
    y_scale_config = alt.Scale(**y_scale_kwargs) if y_scale_kwargs else alt.Undefined

    if show_iqr:
        bands = smoothed_base.mark_area(opacity=0.25).encode(
            y=alt.Y(
                "smooth_q25:Q",
                title=y_label,
                axis=axis_config,
                scale=y_scale_config,
            ),
            y2=alt.Y2("smooth_q75:Q", title=""),
        )

    # Create the line chart using the SMOOTHED iqm value
    lines = smoothed_base.mark_line(strokeWidth=line_width).encode(
        y=alt.Y(
            "smooth_iqm:Q",
            title="" if show_iqr else y_label,
            axis=axis_config,
            scale=y_scale_config,
        ),
        tooltip=[
            alt.Tooltip("algorithm:N", title="Algorithm"),
            alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
            alt.Tooltip("smooth_iqm:Q", title="Smoothed IQM", format=".2f"),
            alt.Tooltip("n_seeds:Q", title="Seeds"),
            alt.Tooltip("n_values:Q", title="Total Values"),
        ],
    )

    chart = lines
    if show_iqr:
        chart = bands + chart

    return (
        chart.properties(
            width=chart_width if chart_width is not None else 1000,
            height=chart_height if chart_height is not None else 400,
            title=alt.TitleParams(
                text=title
                if title
                else f"{metric_name} (IQM over Seeds)",
                fontSize=chart_title_large,
                fontWeight="bold",
                font=ALT_FONT_FAMILY,
            ),
            padding=default_padding,
        )
        .configure_axis(
            grid=True,
            gridOpacity=0.8,
            labelFontSize=axis_label_large,
            titleFontSize=axis_title_large,
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .configure_legend(labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)
        .interactive()
    )


def create_peak_final_bar_chart(
    summary_df: pd.DataFrame,
    overall_title: str,
    metric_label: str,
    base_text_size: float = 20.0,
    log_scale: bool = False,
    show_iqr: bool = True,
    y_min: Optional[float] = None,
    legend_inside: bool = False,
) -> alt.Chart:
    """
    Grouped bar chart (Final vs Peak IQM) with robust y-axis:
    - Explicit y-domain that includes bars and error bars (no 'nice' rounding).
    - Single shared y-scale applied to all layers (bars + rules + caps).
    - Do NOT set axis=None on any layer sharing the y-scale.
    - Extra padding + autosize(type='pad', contains='padding') to prevent clipping.
    - clip=True applied on each mark (not the LayerChart).

    Note: log_scale requires all y-values to be positive. Negative or zero values
    may cause rendering issues or be excluded from the visualization.
    """
    if summary_df.empty:
        raise ValueError("Performance summary is empty; cannot build bar chart.")

    df = summary_df.copy()
    if "metric" not in df.columns:
        df["metric"] = metric_label

    # Sort to ensure consistent ordering with line charts (CPR last)
    df["_render_order"] = df["algorithm"].apply(get_algorithm_render_order)
    df = df.sort_values(["_render_order", "algorithm"])
    df = df.drop(columns=["_render_order"])

    axis_label_size = scaled_font_size(base_text_size, 1.1)
    axis_title_size = scaled_font_size(base_text_size, 1.2)
    legend_label_size = scaled_font_size(base_text_size, 1.1)  # Match axis label size
    legend_title_size = scaled_font_size(base_text_size, 1.2)
    legend_symbol_size = scaled_font_size(base_text_size, 12.5)
    legend_label_limit = max(220, int(round(base_text_size * 12)))
    chart_title_size = scaled_font_size(base_text_size, 1.2)
    x_label_size = axis_label_size
    tick_size = scaled_font_size(base_text_size, 0.9)
    header_font_size = axis_label_size

    # --- reshape to long format (Final / Peak rows)
    rows = []
    for r in df.itertuples():
        final_iqm = getattr(r, "final_iqm", np.nan)
        peak_iqm  = getattr(r, "peak_iqm",  np.nan)
        sort_key  = final_iqm if pd.notna(final_iqm) else peak_iqm

        def add(stage, iqm, q25, q75, step, order):
            if pd.isna(iqm):
                return
            lo = q25 if pd.notna(q25) else iqm
            hi = q75 if pd.notna(q75) else iqm
            rows.append({
                "metric":    getattr(r, "metric"),
                "algorithm": getattr(r, "algorithm"),
                "stage":     stage,
                "iqm":       float(iqm),
                "lower":     float(lo),
                "upper":     float(hi),
                "step":      float(step) if pd.notna(step) else np.nan,
                "stage_order": order,
                "sort_key":  float(sort_key) if pd.notna(sort_key) else float(iqm),
            })

        add("Final", final_iqm,
            getattr(r, "final_q25", np.nan),
            getattr(r, "final_q75", np.nan),
            getattr(r, "final_step", np.nan), 0)

        add("Peak", peak_iqm,
            getattr(r, "peak_q25", np.nan),
            getattr(r, "peak_q75", np.nan),
            getattr(r, "peak_step", np.nan), 1)

    if not rows:
        raise ValueError("Unable to build bar chart; no valid peak/final IQM values found.")

    chart_df = pd.DataFrame(rows)

    # Each metric's bars should start from its observed minimum so that values "fill"
    # upward from the lowest performance (e.g., -2000) instead of diverging around 0.
    baseline_map = {}
    for metric_name, metric_rows in chart_df.groupby("metric"):
        candidates = pd.concat(
            [metric_rows["iqm"], metric_rows["lower"]],
            axis=0,
            ignore_index=True,
        ).dropna()
        baseline_map[metric_name] = float(candidates.min()) if not candidates.empty else 0.0

    if y_min is not None:
        # Use y_min as baseline for all bars
        chart_df["baseline"] = y_min
        baseline_floor = y_min
    else:
        chart_df["baseline"] = chart_df["metric"].map(baseline_map)
        baseline_floor = float(chart_df["baseline"].min()) if not chart_df.empty else 0.0
        baseline_floor = -2001

    # --- explicit, robust y-domain (single metric => global domain)
    metric_count = chart_df["metric"].nunique()
    y_domain = None
    if metric_count == 1:
        ymax = np.nanmax([chart_df["iqm"].max(), chart_df["upper"].max()])
        if not np.isfinite(ymax):
            ymax = baseline_floor + 1.0
        if ymax <= baseline_floor:
            pad = max(1.0, 0.05 * abs(baseline_floor) if baseline_floor else 0.05)
            ymax = baseline_floor + pad
        span = max(1e-9, ymax - baseline_floor)
        top_pad = max(1e-6, 0.04 * span)
        y_domain = [baseline_floor, float(ymax + top_pad)]

    # --- encodings
    algorithm_count = chart_df["algorithm"].nunique()
    base_width = max(320, algorithm_count * 85)

    axis_label = (metric_label or "").strip() or "IQM"
    if "IQM" not in axis_label.upper():
        axis_label = f"{axis_label} (IQM)"

    color_scale = alt.Scale(domain=["Final", "Peak"], range=["#1f77b4", "#ff7f0e"])
    if legend_inside:
        legend = alt.Legend(
            title=None,
            orient="top-right",
            direction="vertical",
            padding=10,
            fillColor="white",
            strokeColor="#ccc",
            cornerRadius=4,
            labelFontSize=legend_label_size,
            labelFontWeight="bold",
            symbolSize=legend_symbol_size,
            values=["Final", "Peak"],
            labelLimit=legend_label_limit,
            labelPadding=8,
            labelFont=ALT_FONT_FAMILY,
        )
    else:
        legend = alt.Legend(
            title=None,
            orient="top",
            direction="horizontal",
            padding=10,
            labelFontSize=legend_label_size,
            labelFontWeight="bold",
            symbolSize=legend_symbol_size,
            values=["Final", "Peak"],
            labelLimit=legend_label_limit,
            labelPadding=8,
            labelFont=ALT_FONT_FAMILY,
        )

    x = alt.X(
        "algorithm:N",
        sort=alt.SortField(field="sort_key", order="descending"),
        title=None,
        axis=alt.Axis(
            offset=12, labelAngle=-30, labelPadding=8,
            labelFontSize=x_label_size,
            labelBaseline="top", labelAlign="right", labelLimit=360,
            labelFont=ALT_FONT_FAMILY,
        ),
    )
    x_off  = alt.XOffset("stage:N", sort=["Final", "Peak"])
    color  = alt.Color("stage:N", title="Stage", scale=color_scale, legend=legend)
    order  = alt.Order("stage_order:Q")

    # Shared y-scale with explicit domain (no 'nice', no zero)
    if y_domain:
        if log_scale:
            y_scale = alt.Scale(type="log", nice=False, domain=y_domain)
        else:
            y_scale = alt.Scale(zero=False, nice=False, domain=y_domain)
    else:
        if log_scale:
            y_scale = alt.Scale(type="log", nice=True, domainMin=baseline_floor)
        else:
            y_scale = alt.Scale(zero=False, nice=True, domainMin=baseline_floor)

    y_axis = alt.Axis(
        title=axis_label,
        format=".2f",
        titlePadding=10,
        titleFontSize=axis_title_size,
        labelFontSize=axis_label_size,
        titleFont=ALT_FONT_FAMILY,
        labelFont=ALT_FONT_FAMILY,
        tickCount=6,
    )
    y = alt.Y("baseline:Q", scale=y_scale, axis=y_axis)
    y2 = alt.Y2("iqm:Q")

    base = alt.Chart(chart_df)  # no clip here

    # Bars define the axis; other layers reuse the scale
    bars = base.mark_bar(size=28, clip=True).encode(
        x=x,
        xOffset=x_off,
        y=y,
        y2=y2,
        color=color,
        order=order,
        tooltip=[
            alt.Tooltip("algorithm:N", title="Algorithm"),
            alt.Tooltip("stage:N",     title="Stage"),
            alt.Tooltip("iqm:Q",       title="IQM",       format=".3f"),
            alt.Tooltip("lower:Q",     title="IQR Lower", format=".3f"),
            alt.Tooltip("upper:Q",     title="IQR Upper", format=".3f"),
            alt.Tooltip("step:Q",      title="Step (M)",  format=".1f"),
        ],
    )

    rules = base.mark_rule(clip=True, color="black").encode(
        x=x, xOffset=x_off,
        y=alt.Y("lower:Q", scale=y_scale),
        y2=alt.Y2("upper:Q"),
        order=order,
    )

    lower_caps = base.mark_tick(thickness=2, size=tick_size, clip=True, color="black").encode(
        x=x, xOffset=x_off,
        y=alt.Y("lower:Q", scale=y_scale),
        order=order,
    )

    upper_caps = base.mark_tick(thickness=2, size=tick_size, clip=True, color="black").encode(
        x=x, xOffset=x_off,
        y=alt.Y("upper:Q", scale=y_scale),
        order=order,
    )

    # Conditionally add IQR error bars
    if show_iqr:
        chart = bars + rules + lower_caps + upper_caps
    else:
        chart = bars

    # Facet if multiple metrics; keep independent y per facet
    if metric_count > 1:
        chart = chart.encode(
            column=alt.Column(
                "metric:N",
                title=None,
                header=alt.Header(
                    labelFontSize=header_font_size,
                    titleFontSize=header_font_size,
                    labelFont=ALT_FONT_FAMILY,
                    titleFont=ALT_FONT_FAMILY,
                ),
            )
        ).resolve_scale(y="independent")

    return (
        chart.properties(
            width=base_width,
            height=420,
            padding={"left": 76, "right": 18, "top": 10, "bottom": 44},
            autosize=alt.AutoSizeParams(type="pad", contains="padding"),
            title=alt.TitleParams(
                text=overall_title or f"{metric_label}: Peak vs Final IQM",
                fontSize=chart_title_size,
                fontWeight="bold",
                font=ALT_FONT_FAMILY,
            ),
        )
        .configure_axis(
            grid=True,
            gridOpacity=0.8,
            labelFontSize=axis_label_size,
            titleFontSize=axis_title_size,
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .configure_legend(
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
            labelFontSize=legend_label_size,
            titleFontSize=legend_title_size,
        )
    )

def main(
    wandb_entity: str,
    wandb_project: str = "crl_experiments",
    group: str = "default_group",
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    combine_networks: bool = False,
    normalize_networks: bool = False,
    split_by: Optional[str] = None,
    algorithm_name: Optional[str] = None,
    name_pattern: Optional[str] = None,
    plot_title: Optional[str] = None,
    output_dir: str = "./plots",
    ext: str = "png",
    debug: bool = False,
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
    legend_inside: bool = False,
):
    if not metrics:
        metrics = [metric] if metric else ["eval_loss"]

    if len(metrics) == 1 and "," in metrics[0]:
        metrics = [m.strip() for m in metrics[0].split(",") if m.strip()]
    else:
        metrics = [m.strip() for m in metrics if m and m.strip()]

    metrics = [canonicalize_metric_name(m) for m in metrics]

    if not metrics:
        raise ValueError("No metric provided; supply --metric or --metrics with a single value.")

    if len(metrics) != 1:
        raise ValueError(
            "analyze.py now expects exactly one metric per invocation; received "
            f"{len(metrics)} metrics ({metrics})."
        )

    selected_metric = metrics[0]
    combined_total_metric: Optional[str] = None

    if normalize_networks and not combine_networks:
        raise ValueError("--normalize-networks requires --combine-networks.")

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
        combined_total_metric = combined_total_metric_key(
            normalized_metric, normalize_networks
        )
    else:
        metrics = [selected_metric]

    df = fetch_and_process_data(
        wandb_entity,
        wandb_project,
        group,
        metrics,
        combine_networks,
        normalize_networks,
        split_by,
        algorithm_name,
        name_pattern,
    )
    if df.empty:
        return print(f"No data found for group: '{group}'")

    summary_records = df.attrs.get("performance_summary")
    summary_df = pd.DataFrame(summary_records) if summary_records else pd.DataFrame()

    if combine_networks:
        metric_label = (
            resolve_metric_labels(combined_total_metric)[0]
            if combined_total_metric is not None
            else "Combined Network Metric"
        )
        title_suffix = metric_label
    else:
        readable_metrics = [resolve_metric_labels(m)[0] for m in metrics]
        metric_label = readable_metrics[0] if readable_metrics else metrics[0]
        title_suffix = ", ".join(readable_metrics)

    title = f"{group.replace('_', ' ').title()}: {title_suffix} (IQM)"
    if split_by:
        pretty_split = split_by.replace("_", " ")
        title += f" • Split by {pretty_split.title()}"
    save_suffix = (
        build_combined_metric_suffix(metrics[0], normalize_networks)
        if combine_networks
        else "_".join([m.replace("/", "_") for m in metrics])
    )
    if split_by:
        save_suffix = f"{save_suffix}_split_by_{split_by}"
    slugged_title = slugify_title(plot_title) if plot_title else None
    if plot_title:
        title = plot_title

    chart = create_chart(
        df,
        metrics,
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

    filename_stem = slugged_title if slugged_title else f"{group}_{save_suffix}_iqm_smoothed"
    save_path = Path(output_dir) / ext / f"{filename_stem}.{ext}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"Chart saved to: {save_path}")

    if bar_chart:
        if summary_df.empty:
            print("Warning: unable to generate peak vs. final bar chart (no summary data).")
        else:
            bar_title = plot_title if plot_title else f"{group.replace('_', ' ').title()}: Peak vs Final IQM"
            bar_chart_obj = create_peak_final_bar_chart(
                summary_df,
                bar_title,
                metric_label,
                base_text_size=base_text_size,
                log_scale=log_scale,
                show_iqr=show_iqr,
                y_min=y_min,
                legend_inside=legend_inside,
            )
            bar_filename_stem = (
                f"{slugged_title}_peak_vs_final_bar"
                if slugged_title
                else f"{group}_{save_suffix}_peak_vs_final_bar"
            )
            bar_save_path = Path(output_dir) / ext / f"{bar_filename_stem}.{ext}"
            bar_save_path.parent.mkdir(exist_ok=True, parents=True)
            bar_chart_obj.save(str(bar_save_path))
            print(f"Summary bar chart saved to: {bar_save_path}")

    return chart


if __name__ == "__main__":
    tyro.cli(main)
