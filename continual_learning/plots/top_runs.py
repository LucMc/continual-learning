#!/usr/bin/env python3
"""Query top runs from a W&B group by a specified metric."""
import re
import tyro
import wandb
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Any


def extract_config_name(run_name: str) -> str:
    """Strip seed suffix (e.g., _s0, _s1, _s2 or _0, _1, _2) from run name."""
    # Match patterns like _s0, _s1, _s2 or just _0, _1, _2 at end
    return re.sub(r'_s?\d+$', '', run_name)


def main(
    entity: str = "lucmc",
    project: str = "MT1_sweep",
    group: str = "mt1_ccbp_sweep",
    metric: str = "final/mean_return",
    direction: str = "max",
    n: int = 10,
    show_config: bool = True,
    no_aggregate: bool = False,
):
    """
    Find top N configs from a W&B group ranked by a metric, averaged across seeds.

    Args:
        entity: W&B entity/username
        project: W&B project name
        group: Run group name
        metric: Metric to rank by (from run.summary)
        direction: 'max' for highest values, 'min' for lowest
        n: Number of top configs to show
        show_config: Also print hyperparameter configs
        no_aggregate: If True, rank individual runs instead of aggregating by config
    """
    api = wandb.Api()

    # Fetch all runs in the group
    runs = list(api.runs(
        f"{entity}/{project}",
        filters={"group": group},
        per_page=500,
    ))

    print(f"Found {len(runs)} total runs in group '{group}'")

    # Filter to finished runs with the metric
    valid_runs: List[Tuple[Any, float]] = []
    for run in runs:
        if run.state != "finished":
            continue
        summary = run.summary
        if metric in summary and summary[metric] is not None:
            valid_runs.append((run, float(summary[metric])))

    print(f"Found {len(valid_runs)} finished runs with metric '{metric}'")

    if not valid_runs:
        print(f"\nNo runs found with metric '{metric}'. Available summary keys from first run:")
        if runs:
            print(list(runs[0].summary.keys())[:20])
        return

    if no_aggregate:
        # Original behavior: rank individual runs
        reverse = direction == "max"
        valid_runs.sort(key=lambda x: x[1], reverse=reverse)

        print(f"\n{'='*80}")
        print(f"Top {min(n, len(valid_runs))} individual runs by {metric} ({direction}):")
        print(f"{'='*80}\n")

        for i, (run, value) in enumerate(valid_runs[:n], 1):
            print(f"{i:2d}. {run.name}")
            print(f"    {metric}: {value:.4f}")
            print(f"    URL: {run.url}")
            print()
    else:
        # Aggregate by config (strip seed suffix)
        config_runs: dict[str, List[Tuple[Any, float]]] = defaultdict(list)
        for run, value in valid_runs:
            config_name = extract_config_name(run.name)
            config_runs[config_name].append((run, value))

        # Compute mean and std for each config
        config_stats: List[Tuple[str, float, float, int, List[Tuple[Any, float]]]] = []
        for config_name, runs_list in config_runs.items():
            values = [v for _, v in runs_list]
            mean_val = np.mean(values)
            std_val = np.std(values) if len(values) > 1 else 0.0
            config_stats.append((config_name, mean_val, std_val, len(runs_list), runs_list))

        # Sort by mean value
        reverse = direction == "max"
        config_stats.sort(key=lambda x: x[1], reverse=reverse)

        print(f"\n{'='*80}")
        print(f"Top {min(n, len(config_stats))} configs by {metric} ({direction}), averaged across seeds:")
        print(f"{'='*80}\n")

        for i, (config_name, mean_val, std_val, num_seeds, runs_list) in enumerate(config_stats[:n], 1):
            print(f"{i:2d}. {config_name}")
            print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}  (n={num_seeds} seeds)")

            # Show individual seed values
            seed_values = [(run.name.split('_')[-1], v) for run, v in runs_list]
            seed_str = ", ".join([f"{s}={v:.1f}" for s, v in sorted(seed_values)])
            print(f"    Seeds: {seed_str}")

            if show_config:
                # Get config from first run
                config = runs_list[0][0].config
                interesting_keys = [
                    'replacement_rate', 'threshold', 'decay_rate',
                    'update_frequency', 'every_n', 'update_every',
                    'learning_rate', 'lr', 'batch_size',
                    'reset_interval', 'maturity_threshold',
                    'cbp_replacement_rate', 'utility_threshold',
                ]
                shown = []
                for key in interesting_keys:
                    if key in config and key != 'seed':
                        shown.append(f"{key}={config[key]}")
                # Also show any keys with 'rate', 'threshold', 'lr', 'frequency', 'every' in name
                for key, val in config.items():
                    if any(x in key.lower() for x in ['rate', 'threshold', 'lr', 'lambda', 'frequency', 'every_n', 'update_every']):
                        if key not in interesting_keys and key != 'seed':
                            shown.append(f"{key}={val}")
                if shown:
                    print(f"    Config: {', '.join(shown[:8])}")

            # Show URL of best seed
            best_run = max(runs_list, key=lambda x: x[1]) if direction == "max" else min(runs_list, key=lambda x: x[1])
            print(f"    Best seed URL: {best_run[0].url}")
            print()

        # Summary statistics across all configs
        all_means = [mean_val for _, mean_val, _, _, _ in config_stats]
        print(f"{'='*80}")
        print(f"Summary statistics across {len(config_stats)} configs:")
        print(f"  Best mean:   {max(all_means):.4f}")
        print(f"  Worst mean:  {min(all_means):.4f}")
        print(f"  Mean of means: {np.mean(all_means):.4f}")
        print(f"  Median:      {np.median(all_means):.4f}")


if __name__ == "__main__":
    tyro.cli(main)
