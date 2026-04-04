#!/usr/bin/env python3
"""Compare sweep configs for a given algorithm against the adam baseline.

Fetches runs from the sweep project (cont-minatar-sweep) for the specified
algorithm, computes per-task and average performance, and prints a comparison
table with metrics normalised relative to adam (from cont-minatar).

Usage:
    python continual_learning/plots/sweep_comparison.py --algo cpr
    python continual_learning/plots/sweep_comparison.py --algo redo --tail-frac 0.3
    python continual_learning/plots/sweep_comparison.py --algo cbp --output-csv results.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tyro
import wandb

TASKS = ["space_invaders", "asterix", "seaquest"]
STEPS_PER_TASK = 1_500_000
TOTAL_STEPS = STEPS_PER_TASK * len(TASKS)

# Task step ranges (inclusive): task i runs from i*STEPS_PER_TASK to (i+1)*STEPS_PER_TASK
TASK_RANGES = {
    task: (i * STEPS_PER_TASK, (i + 1) * STEPS_PER_TASK)
    for i, task in enumerate(TASKS)
}


# ---- helpers ----------------------------------------------------------------

def compute_iqm(values: np.ndarray) -> float:
    if len(values) < 4:
        return float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return float(np.mean(values[mask])) if np.any(mask) else float(np.mean(values))


def _extract_config_tag(run_name: str, algo: str) -> str:
    """Extract the hyperparameter tag from a sweep run name.

    Expected format: sweep_{algo}_{param_tag}_s{seed}
    """
    prefix = f"sweep_{algo}_"
    if not run_name.startswith(prefix):
        return run_name
    rest = run_name[len(prefix):]
    # Remove trailing _s{seed}
    parts = rest.rsplit("_s", 1)
    return parts[0] if len(parts) == 2 else rest


def _extract_seed(run_name: str) -> str:
    parts = run_name.rsplit("_s", 1)
    return parts[1] if len(parts) == 2 else "0"


# ---- data fetching ----------------------------------------------------------

def fetch_adam_baseline(
    entity: str,
    project: str,
    tail_frac: float,
) -> dict[str, dict]:
    """Fetch adam continual runs and compute per-task performance.

    Returns dict[task] -> {"avg": float, "peak": float}
    """
    api = wandb.Api()
    runs = list(api.runs(
        f"{entity}/{project}",
        filters={"group": "minatar_discrete_sac"},
        per_page=300,
    ))
    adam_runs = [
        r for r in runs
        if r.state == "finished" and "adam" in r.name and "LOW" in r.name
    ]
    if not adam_runs:
        # Fall back: try any adam run
        adam_runs = [r for r in runs if r.state == "finished" and "adam" in r.name]

    print(f"Adam baseline: {len(adam_runs)} finished run(s) from {entity}/{project}")

    task_results: dict[str, dict] = {}
    for task in TASKS:
        start_step, end_step = TASK_RANGES[task]
        seed_avgs: list[float] = []
        seed_peaks: list[float] = []

        for r in adam_runs:
            h = r.history(
                keys=["charts/mean_episode_return", "charts/total_steps"],
                samples=10_000,
            )
            if h.empty:
                continue

            steps = h["charts/total_steps"].values
            vals = h["charts/mean_episode_return"].values

            mask = (~np.isnan(steps)) & (~np.isnan(vals))
            steps, vals = steps[mask], vals[mask]

            task_mask = (steps >= start_step) & (steps <= end_step)
            task_vals = vals[task_mask]

            if len(task_vals) == 0:
                continue

            tail_n = max(1, int(len(task_vals) * tail_frac))
            seed_avgs.append(float(np.mean(task_vals[-tail_n:])))
            seed_peaks.append(float(np.max(task_vals)))

        if seed_avgs:
            task_results[task] = {
                "avg": float(np.mean(seed_avgs)),
                "peak": float(np.mean(seed_peaks)),
            }
            print(f"  {task}: avg={task_results[task]['avg']:.2f}, peak={task_results[task]['peak']:.2f}")
        else:
            print(f"  WARNING: no adam data for {task}")

    return task_results


def fetch_sweep_runs(
    entity: str,
    project: str,
    algo: str,
    tail_frac: float,
) -> pd.DataFrame:
    """Fetch sweep runs for the given algorithm.

    Returns DataFrame with columns:
        config, seed, task, avg_return, peak_return
    """
    api = wandb.Api()
    group = f"minatar_sweep_{algo}"
    runs = list(api.runs(
        f"{entity}/{project}",
        filters={"group": group},
        per_page=500,
    ))
    finished = [r for r in runs if r.state == "finished"]
    print(f"\nSweep runs ({algo}): {len(finished)} finished (of {len(runs)} total in '{group}')")

    rows: list[dict] = []
    for r in finished:
        config_tag = _extract_config_tag(r.name, algo)
        seed = _extract_seed(r.name)

        h = r.history(
            keys=["charts/mean_episode_return", "charts/total_steps"],
            samples=10_000,
        )
        if h.empty:
            continue

        steps = h["charts/total_steps"].values
        vals = h["charts/mean_episode_return"].values

        mask = (~np.isnan(steps)) & (~np.isnan(vals))
        steps, vals = steps[mask], vals[mask]

        for task in TASKS:
            start_step, end_step = TASK_RANGES[task]
            task_mask = (steps >= start_step) & (steps <= end_step)
            task_vals = vals[task_mask]

            if len(task_vals) == 0:
                continue

            tail_n = max(1, int(len(task_vals) * tail_frac))
            rows.append({
                "config": config_tag,
                "seed": seed,
                "task": task,
                "avg_return": float(np.mean(task_vals[-tail_n:])),
                "peak_return": float(np.max(task_vals)),
            })

    return pd.DataFrame(rows)


# ---- aggregation & table ----------------------------------------------------

def build_comparison_table(
    sweep_df: pd.DataFrame,
    adam_baseline: dict[str, dict],
) -> pd.DataFrame:
    """Aggregate sweep results and compute relative-to-adam metrics.

    Returns one row per config with columns for each task's avg/peak and
    relative performance vs adam.
    """
    if sweep_df.empty:
        return pd.DataFrame()

    result_rows: list[dict] = []

    for config, cfg_df in sweep_df.groupby("config"):
        row: dict = {"config": config}
        n_seeds = cfg_df["seed"].nunique()
        row["n_seeds"] = n_seeds

        task_rel_avgs: list[float] = []
        task_rel_peaks: list[float] = []

        for task in TASKS:
            task_df = cfg_df[cfg_df["task"] == task]
            if task_df.empty:
                row[f"{task}_avg"] = np.nan
                row[f"{task}_peak"] = np.nan
                row[f"{task}_rel_avg"] = np.nan
                row[f"{task}_rel_peak"] = np.nan
                continue

            # Aggregate across seeds using IQM
            avg_val = compute_iqm(task_df["avg_return"].values)
            peak_val = compute_iqm(task_df["peak_return"].values)

            row[f"{task}_avg"] = avg_val
            row[f"{task}_peak"] = peak_val

            # Relative to adam
            if task in adam_baseline and adam_baseline[task]["avg"] != 0:
                rel_avg = (avg_val / adam_baseline[task]["avg"]) * 100
                rel_peak = (peak_val / adam_baseline[task]["peak"]) * 100
                row[f"{task}_rel_avg"] = rel_avg
                row[f"{task}_rel_peak"] = rel_peak
                task_rel_avgs.append(rel_avg)
                task_rel_peaks.append(rel_peak)
            else:
                row[f"{task}_rel_avg"] = np.nan
                row[f"{task}_rel_peak"] = np.nan

        # Cross-task averages
        if task_rel_avgs:
            row["mean_rel_avg"] = float(np.mean(task_rel_avgs))
            row["mean_rel_peak"] = float(np.mean(task_rel_peaks))
        else:
            row["mean_rel_avg"] = np.nan
            row["mean_rel_peak"] = np.nan

        result_rows.append(row)

    df = pd.DataFrame(result_rows)
    df = df.sort_values("mean_rel_avg", ascending=False).reset_index(drop=True)
    return df


def print_table(df: pd.DataFrame, algo: str, adam_baseline: dict[str, dict]) -> None:
    """Pretty-print the comparison table."""
    if df.empty:
        print("No data to display.")
        return

    print(f"\n{'='*120}")
    print(f"  SWEEP RESULTS: {algo.upper()}  (relative to adam baseline)")
    print(f"{'='*120}")

    # Print adam baseline first
    print(f"\n  Adam baseline (from cont-minatar):")
    for task in TASKS:
        if task in adam_baseline:
            b = adam_baseline[task]
            print(f"    {task:18s}  avg={b['avg']:8.2f}  peak={b['peak']:8.2f}")

    # Column headers
    print(f"\n{'─'*120}")
    header = f"{'Rank':>4}  {'Config':<50}  {'Seeds':>5}"
    for task in TASKS:
        short = task[:6]
        header += f"  {short+'_avg':>10}  {short+'_rel':>10}"
    header += f"  {'mean_rel':>10}"
    print(header)
    print(f"{'─'*120}")

    for idx, row in df.iterrows():
        line = f"{idx+1:>4}  {str(row['config']):<50}  {int(row['n_seeds']):>5}"
        for task in TASKS:
            avg = row.get(f"{task}_avg", np.nan)
            rel = row.get(f"{task}_rel_avg", np.nan)
            avg_s = f"{avg:8.2f}" if not np.isnan(avg) else "     N/A"
            rel_s = f"{rel:7.1f}%" if not np.isnan(rel) else "     N/A"
            line += f"  {avg_s:>10}  {rel_s:>10}"
        mean_rel = row.get("mean_rel_avg", np.nan)
        mean_s = f"{mean_rel:7.1f}%" if not np.isnan(mean_rel) else "     N/A"
        line += f"  {mean_s:>10}"
        print(line)

    print(f"{'─'*120}")

    # Top 5 summary
    print(f"\n  Top 5 configs by mean relative avg performance:")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        mean_rel = row.get("mean_rel_avg", np.nan)
        print(f"    {i+1}. {row['config']}")
        for task in TASKS:
            avg = row.get(f"{task}_avg", np.nan)
            rel = row.get(f"{task}_rel_avg", np.nan)
            peak = row.get(f"{task}_peak", np.nan)
            rel_peak = row.get(f"{task}_rel_peak", np.nan)
            print(f"         {task:18s}  avg={avg:8.2f} ({rel:6.1f}% of adam)"
                  f"  peak={peak:8.2f} ({rel_peak:6.1f}% of adam)")
        print(f"         {'Mean relative':18s}  avg={mean_rel:6.1f}%"
              f"  peak={row.get('mean_rel_peak', np.nan):6.1f}%")
        print()


# ---- CLI --------------------------------------------------------------------

def main(
    algo: str,
    wandb_entity: str = "lucmc",
    sweep_project: str = "cont-minatar-sweep",
    baseline_project: str = "cont-minatar",
    tail_frac: float = 0.2,
    output_csv: Optional[str] = None,
):
    """Compare sweep configs for an algorithm against the adam continual baseline.

    Args:
        algo: Algorithm name (redo, regrama, cbp, cpr, shrink_and_perturb).
        wandb_entity: W&B entity/username.
        sweep_project: W&B project containing sweep runs.
        baseline_project: W&B project containing adam baseline runs.
        tail_frac: Fraction of tail steps to average for "avg" metric.
        output_csv: Optional path to save results as CSV.
    """
    print(f"Fetching adam baseline from {wandb_entity}/{baseline_project}...")
    adam_baseline = fetch_adam_baseline(wandb_entity, baseline_project, tail_frac)
    if not adam_baseline:
        print("ERROR: could not compute adam baseline. Exiting.")
        return

    print(f"\nFetching sweep runs for {algo} from {wandb_entity}/{sweep_project}...")
    sweep_df = fetch_sweep_runs(wandb_entity, sweep_project, algo, tail_frac)
    if sweep_df.empty:
        print(f"ERROR: no sweep data found for {algo}. Exiting.")
        return

    print(f"\nSweep data: {len(sweep_df)} task-level measurements")
    print(f"Configs: {sweep_df['config'].nunique()}")
    print(f"Seeds: {sweep_df['seed'].unique().tolist()}")

    table = build_comparison_table(sweep_df, adam_baseline)
    print_table(table, algo, adam_baseline)

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        table.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    tyro.cli(main)
