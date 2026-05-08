#!/usr/bin/env python3
"""Compare sweep configs for a given algorithm against the adam baseline.

Fetches runs from the sweep project (cont-minatar-sweep) for the specified
algorithm, computes per-task and average performance, and prints a comparison
table with metrics normalised relative to adam (from cont-minatar).

Usage:
    python continual_learning/plots/sweep_comparison.py --algo cpr
    python continual_learning/plots/sweep_comparison.py --algo redo --tail-frac 0.3
    python continual_learning/plots/sweep_comparison.py --algo cbp --output-csv results.csv

    # Compare best config from each algorithm side-by-side:
    python continual_learning/plots/sweep_comparison.py --all
    python continual_learning/plots/sweep_comparison.py --all --top-k 3
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import tyro
import wandb

TASKS = ["space_invaders", "asterix", "seaquest"]
STEPS_PER_TASK = 1_500_000
TOTAL_STEPS = STEPS_PER_TASK * len(TASKS)
BOUNDARY_FRAC = 0.05  # fraction of task steps for boundary window
ALL_ALGOS = ["cbp", "cpr", "redo", "regrama", "shrink_and_perturb"]

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


def _is_recent(run, max_age_days: Optional[float]) -> bool:
    if max_age_days is None:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
    return created >= cutoff


# ---- data fetching ----------------------------------------------------------

def fetch_adam_baseline(
    entity: str,
    project: str,
    tail_frac: float,
    max_age_days: Optional[float] = None,
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
        and _is_recent(r, max_age_days)
    ]
    if not adam_runs:
        # Fall back: try any adam run (still respecting age filter)
        adam_runs = [
            r for r in runs
            if r.state == "finished" and "adam" in r.name
            and _is_recent(r, max_age_days)
        ]

    print(f"Adam baseline: {len(adam_runs)} finished run(s) from {entity}/{project}"
          f"{f' (last {max_age_days:.0f}d)' if max_age_days else ''}")

    task_results: dict[str, dict] = {}
    for task in TASKS:
        start_step, end_step = TASK_RANGES[task]
        seed_avgs: list[float] = []
        seed_peaks: list[float] = []
        seed_means: list[float] = []
        seed_boundary_starts: list[float] = []
        seed_boundary_ends: list[float] = []

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
            task_steps = steps[task_mask]
            task_vals = vals[task_mask]

            if len(task_vals) == 0:
                continue

            tail_n = max(1, int(len(task_vals) * tail_frac))
            seed_avgs.append(float(np.mean(task_vals[-tail_n:])))
            seed_peaks.append(float(np.max(task_vals)))
            seed_means.append(float(np.mean(task_vals)))

            # Boundary values (small window at start/end of task)
            boundary_window = STEPS_PER_TASK * BOUNDARY_FRAC
            b_start_mask = task_steps <= (start_step + boundary_window)
            b_end_mask = task_steps >= (end_step - boundary_window)
            if np.any(b_start_mask):
                seed_boundary_starts.append(float(np.mean(task_vals[b_start_mask])))
            if np.any(b_end_mask):
                seed_boundary_ends.append(float(np.mean(task_vals[b_end_mask])))

        if seed_avgs:
            task_results[task] = {
                "avg": float(np.mean(seed_avgs)),
                "peak": float(np.mean(seed_peaks)),
                "mean": float(np.mean(seed_means)),
                "boundary_start": float(np.mean(seed_boundary_starts)) if seed_boundary_starts else np.nan,
                "boundary_end": float(np.mean(seed_boundary_ends)) if seed_boundary_ends else np.nan,
            }
            b = task_results[task]
            print(f"  {task}: avg={b['avg']:.2f}, peak={b['peak']:.2f}, "
                  f"mean={b['mean']:.2f}, boundary=[{b['boundary_start']:.2f} -> {b['boundary_end']:.2f}]")
        else:
            print(f"  WARNING: no adam data for {task}")

    return task_results


def fetch_sweep_runs(
    entity: str,
    project: str,
    algo: str,
    tail_frac: float,
    max_age_days: Optional[float] = None,
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
    finished = [r for r in runs if r.state == "finished" and _is_recent(r, max_age_days)]
    print(f"\nSweep runs ({algo}): {len(finished)} finished"
          f" (of {len(runs)} total in '{group}')"
          f"{f', last {max_age_days:.0f}d' if max_age_days else ''}")

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
            task_steps = steps[task_mask]
            task_vals = vals[task_mask]

            if len(task_vals) == 0:
                continue

            tail_n = max(1, int(len(task_vals) * tail_frac))

            # Boundary values
            boundary_window = STEPS_PER_TASK * BOUNDARY_FRAC
            b_start_mask = task_steps <= (start_step + boundary_window)
            b_end_mask = task_steps >= (end_step - boundary_window)
            b_start = float(np.mean(task_vals[b_start_mask])) if np.any(b_start_mask) else np.nan
            b_end = float(np.mean(task_vals[b_end_mask])) if np.any(b_end_mask) else np.nan

            rows.append({
                "config": config_tag,
                "seed": seed,
                "task": task,
                "avg_return": float(np.mean(task_vals[-tail_n:])),
                "peak_return": float(np.max(task_vals)),
                "mean_return": float(np.mean(task_vals)),
                "boundary_start": b_start,
                "boundary_end": b_end,
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
                row[f"{task}_mean"] = np.nan
                row[f"{task}_boundary_start"] = np.nan
                row[f"{task}_boundary_end"] = np.nan
                continue

            # Aggregate across seeds using IQM
            avg_val = compute_iqm(task_df["avg_return"].values)
            peak_val = compute_iqm(task_df["peak_return"].values)

            # Normal mean across seeds for mean_return and boundary values
            mean_val = float(np.mean(task_df["mean_return"].values))
            boundary_start_val = float(np.nanmean(task_df["boundary_start"].values))
            boundary_end_val = float(np.nanmean(task_df["boundary_end"].values))

            row[f"{task}_avg"] = avg_val
            row[f"{task}_peak"] = peak_val
            row[f"{task}_mean"] = mean_val
            row[f"{task}_boundary_start"] = boundary_start_val
            row[f"{task}_boundary_end"] = boundary_end_val

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
            task_mean = row.get(f"{task}_mean", np.nan)
            b_start = row.get(f"{task}_boundary_start", np.nan)
            b_end = row.get(f"{task}_boundary_end", np.nan)
            print(f"         {task:18s}  avg={avg:8.2f} ({rel:6.1f}% of adam)"
                  f"  peak={peak:8.2f} ({rel_peak:6.1f}% of adam)")
            print(f"         {'':18s}  mean={task_mean:8.2f}"
                  f"  boundary=[{b_start:8.2f} -> {b_end:8.2f}]")
        print(f"         {'Mean relative':18s}  avg={mean_rel:6.1f}%"
              f"  peak={row.get('mean_rel_peak', np.nan):6.1f}%")
        print()

    # Boundary reward summary table
    print(f"\n{'─'*120}")
    print(f"  TASK BOUNDARY REWARDS (mean episodic return at start/end of each task)")
    print(f"{'─'*120}")
    # Adam baseline boundaries
    print(f"  {'Adam baseline':50s}", end="")
    for task in TASKS:
        if task in adam_baseline:
            b = adam_baseline[task]
            bs = b.get("boundary_start", np.nan)
            be = b.get("boundary_end", np.nan)
            print(f"  {task[:10]:>10}: [{bs:7.1f} -> {be:7.1f}]", end="")
    print()
    # Top configs
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        label = f"  {i+1}. {str(row['config'])[:47]}"
        print(f"{label:50s}", end="")
        for task in TASKS:
            b_start = row.get(f"{task}_boundary_start", np.nan)
            b_end = row.get(f"{task}_boundary_end", np.nan)
            bs_s = f"{b_start:7.1f}" if not np.isnan(b_start) else "    N/A"
            be_s = f"{b_end:7.1f}" if not np.isnan(b_end) else "    N/A"
            print(f"  {task[:10]:>10}: [{bs_s} -> {be_s}]", end="")
        print()
    print(f"{'─'*120}")


# ---- cross-algorithm comparison ---------------------------------------------

def compare_all_algos(
    wandb_entity: str,
    sweep_project: str,
    adam_baseline: dict[str, dict],
    tail_frac: float,
    top_k: int = 1,
    max_age_days: Optional[float] = None,
    min_seeds: int = 1,
) -> pd.DataFrame:
    """Fetch sweeps for all algorithms, pick the top-k config from each, and
    return a combined comparison table."""
    combined_rows: list[pd.DataFrame] = []

    for algo in ALL_ALGOS:
        print(f"\n{'─'*60}")
        print(f"Fetching sweep runs for {algo}...")
        sweep_df = fetch_sweep_runs(wandb_entity, sweep_project, algo, tail_frac, max_age_days)
        if sweep_df.empty:
            print(f"  WARNING: no sweep data for {algo}, skipping")
            continue

        table = build_comparison_table(sweep_df, adam_baseline)
        if table.empty:
            continue

        if min_seeds > 1:
            before = len(table)
            table = table[table["n_seeds"] >= min_seeds].reset_index(drop=True)
            print(f"  filtered by n_seeds>={min_seeds}: {len(table)}/{before} configs kept")
            if table.empty:
                continue

        top = table.head(top_k).copy()
        top.insert(0, "algorithm", algo)
        combined_rows.append(top)

    if not combined_rows:
        return pd.DataFrame()

    combined = pd.concat(combined_rows, ignore_index=True)
    combined = combined.sort_values("mean_rel_avg", ascending=False).reset_index(drop=True)
    return combined


def print_all_table(
    df: pd.DataFrame,
    adam_baseline: dict[str, dict],
    top_k: int,
) -> None:
    """Pretty-print the cross-algorithm comparison table."""
    if df.empty:
        print("No data to display.")
        return

    label = "best config" if top_k == 1 else f"top {top_k} configs"
    print(f"\n{'='*130}")
    print(f"  CROSS-ALGORITHM COMPARISON  ({label} per algorithm, relative to adam)")
    print(f"{'='*130}")

    print(f"\n  Adam baseline:")
    for task in TASKS:
        if task in adam_baseline:
            b = adam_baseline[task]
            print(f"    {task:18s}  avg={b['avg']:8.2f}  peak={b['peak']:8.2f}")

    print(f"\n{'─'*130}")
    header = f"{'Rank':>4}  {'Algorithm':<20}  {'Config':<35}  {'Seeds':>5}"
    for task in TASKS:
        short = task[:6]
        header += f"  {short+'_rel':>10}"
    header += f"  {'mean_rel':>10}"
    print(header)
    print(f"{'─'*130}")

    for idx, row in df.iterrows():
        algo = row["algorithm"]
        config = str(row["config"])
        if len(config) > 35:
            config = config[:32] + "..."
        n_seeds = int(row["n_seeds"])

        line = f"{idx+1:>4}  {algo:<20}  {config:<35}  {n_seeds:>5}"
        for task in TASKS:
            rel = row.get(f"{task}_rel_avg", np.nan)
            rel_s = f"{rel:7.1f}%" if not np.isnan(rel) else "     N/A"
            line += f"  {rel_s:>10}"
        mean_rel = row.get("mean_rel_avg", np.nan)
        mean_s = f"{mean_rel:7.1f}%" if not np.isnan(mean_rel) else "     N/A"
        line += f"  {mean_s:>10}"
        print(line)

    print(f"{'─'*130}")

    # Detailed breakdown of top entries
    print(f"\n  Detailed breakdown:")
    for i, (_, row) in enumerate(df.iterrows()):
        algo = row["algorithm"]
        mean_rel = row.get("mean_rel_avg", np.nan)
        print(f"    {i+1}. {algo} — {row['config']}")
        for task in TASKS:
            avg = row.get(f"{task}_avg", np.nan)
            rel = row.get(f"{task}_rel_avg", np.nan)
            peak = row.get(f"{task}_peak", np.nan)
            rel_peak = row.get(f"{task}_rel_peak", np.nan)
            task_mean = row.get(f"{task}_mean", np.nan)
            b_start = row.get(f"{task}_boundary_start", np.nan)
            b_end = row.get(f"{task}_boundary_end", np.nan)
            print(f"         {task:18s}  avg={avg:8.2f} ({rel:6.1f}% of adam)"
                  f"  peak={peak:8.2f} ({rel_peak:6.1f}% of adam)")
            print(f"         {'':18s}  mean={task_mean:8.2f}"
                  f"  boundary=[{b_start:8.2f} -> {b_end:8.2f}]")
        print(f"         {'Mean relative':18s}  avg={mean_rel:6.1f}%"
              f"  peak={row.get('mean_rel_peak', np.nan):6.1f}%")
        print()

    # Cross-algorithm boundary comparison
    print(f"\n{'─'*130}")
    print(f"  TASK BOUNDARY REWARDS (mean episodic return at start/end of each task)")
    print(f"{'─'*130}")
    print(f"  {'Adam baseline':50s}", end="")
    for task in TASKS:
        if task in adam_baseline:
            b = adam_baseline[task]
            bs = b.get("boundary_start", np.nan)
            be = b.get("boundary_end", np.nan)
            print(f"  {task[:10]:>10}: [{bs:7.1f} -> {be:7.1f}]", end="")
    print()
    for i, (_, row) in enumerate(df.iterrows()):
        algo = row["algorithm"]
        label = f"  {i+1}. {algo} — {str(row['config'])[:35]}"
        print(f"{label:50s}", end="")
        for task in TASKS:
            b_start = row.get(f"{task}_boundary_start", np.nan)
            b_end = row.get(f"{task}_boundary_end", np.nan)
            bs_s = f"{b_start:7.1f}" if not np.isnan(b_start) else "    N/A"
            be_s = f"{b_end:7.1f}" if not np.isnan(b_end) else "    N/A"
            print(f"  {task[:10]:>10}: [{bs_s} -> {be_s}]", end="")
        print()
    print(f"{'─'*130}")


# ---- time-series fetching & plotting ----------------------------------------

def fetch_timeseries_for_configs(
    entity: str,
    project: str,
    algo_configs: list[tuple[str, str]],
    max_age_days: Optional[float] = None,
) -> pd.DataFrame:
    """Fetch raw time-series for specific algo/config pairs.

    Returns DataFrame with columns: algorithm, seed, step, mean_return
    """
    api = wandb.Api()
    rows: list[dict] = []

    for algo, config_tag in algo_configs:
        group = f"minatar_sweep_{algo}"
        runs = list(api.runs(
            f"{entity}/{project}",
            filters={"group": group},
            per_page=500,
        ))
        finished = [r for r in runs if r.state == "finished" and _is_recent(r, max_age_days)]
        matching = [r for r in finished if _extract_config_tag(r.name, algo) == config_tag]
        print(f"  Timeseries: {algo}/{config_tag}: {len(matching)} run(s)")

        for r in matching:
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

            for s, v in zip(steps, vals):
                rows.append({
                    "algorithm": algo,
                    "config": config_tag,
                    "seed": seed,
                    "step": int(s),
                    "mean_return": float(v),
                })

    return pd.DataFrame(rows)


def fetch_adam_timeseries(
    entity: str,
    project: str,
    max_age_days: Optional[float] = None,
) -> pd.DataFrame:
    """Fetch raw time-series for adam baseline."""
    api = wandb.Api()
    runs = list(api.runs(
        f"{entity}/{project}",
        filters={"group": "minatar_discrete_sac"},
        per_page=300,
    ))
    adam_runs = [
        r for r in runs
        if r.state == "finished" and "adam" in r.name and "LOW" in r.name
        and _is_recent(r, max_age_days)
    ]
    if not adam_runs:
        adam_runs = [
            r for r in runs
            if r.state == "finished" and "adam" in r.name
            and _is_recent(r, max_age_days)
        ]

    print(f"  Timeseries: adam baseline: {len(adam_runs)} run(s)")

    rows: list[dict] = []
    for r in adam_runs:
        seed = r.name.rsplit("_", 1)[-1]
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

        for s, v in zip(steps, vals):
            rows.append({
                "algorithm": "adam",
                "config": "baseline",
                "seed": seed,
                "step": int(s),
                "mean_return": float(v),
            })

    return pd.DataFrame(rows)


def plot_best_configs(
    timeseries_df: pd.DataFrame,
    output_dir: str = "./plots",
    bin_size: int = 10_000,
) -> None:
    """Plot mean episodic return over training steps for best configs.

    Renders three horizontal panels (one per task), each with its own y-axis
    so asterix/seaquest detail isn't compressed by space_invaders.
    """
    if timeseries_df.empty:
        print("No data to plot.")
        return

    df = timeseries_df.copy()
    df["binned_step"] = (df["step"] / bin_size).round().astype(int) * bin_size
    df["step_M"] = df["binned_step"] / 1_000_000
    # Tag each row with the active task based on its step
    df["task"] = (df["binned_step"] // STEPS_PER_TASK).clip(upper=len(TASKS) - 1)
    df["task"] = df["task"].apply(lambda i: TASKS[int(i)])

    # Aggregate across seeds per (algorithm, task, step)
    agg = df.groupby(["algorithm", "task", "step_M"]).agg(
        mean_return=("mean_return", "mean"),
        q25=(
            "mean_return",
            lambda x: float(np.percentile(x, 25)) if len(x) > 1 else float(x.mean()),
        ),
        q75=(
            "mean_return",
            lambda x: float(np.percentile(x, 75)) if len(x) > 1 else float(x.mean()),
        ),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    task_titles = {
        "space_invaders": "Space Invaders",
        "asterix": "Asterix",
        "seaquest": "Seaquest",
    }

    panels: list[alt.Chart] = []
    for i, task in enumerate(TASKS):
        task_df = agg[agg["task"] == task]
        if task_df.empty:
            continue

        x_lo = i * STEPS_PER_TASK / 1_000_000
        x_hi = (i + 1) * STEPS_PER_TASK / 1_000_000

        base = alt.Chart(task_df).encode(
            x=alt.X(
                "step_M:Q",
                title="Training Steps (Millions)",
                scale=alt.Scale(domain=[x_lo, x_hi], nice=False),
                axis=alt.Axis(labelFontSize=11, titleFontSize=13),
            ),
            color=alt.Color(
                "algorithm:N",
                title="Algorithm",
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    labelFontSize=11,
                    symbolSize=150,
                )
                if i == 0
                else None,
            ),
        )

        smoothed = base.transform_window(
            frame=[-10, 10],
            groupby=["algorithm"],
            smooth_mean="mean(mean_return)",
            smooth_q25="mean(q25)",
            smooth_q75="mean(q75)",
        )

        bands = smoothed.mark_area(opacity=0.18).encode(
            y=alt.Y(
                "smooth_q25:Q",
                title="Mean Episode Return" if i == 0 else None,
            ),
            y2=alt.Y2("smooth_q75:Q"),
        )

        lines = smoothed.mark_line(strokeWidth=2).encode(
            y=alt.Y("smooth_mean:Q"),
            tooltip=[
                alt.Tooltip("algorithm:N", title="Algorithm"),
                alt.Tooltip("step_M:Q", title="Step (M)", format=".2f"),
                alt.Tooltip("smooth_mean:Q", title="Mean Return", format=".2f"),
                alt.Tooltip("n_seeds:Q", title="Seeds"),
            ],
        )

        panel = alt.layer(bands, lines).properties(
            width=340,
            height=320,
            title=alt.TitleParams(
                text=task_titles.get(task, task),
                fontSize=13,
                fontWeight="bold",
            ),
        )
        panels.append(panel)

    if not panels:
        print("No data to plot.")
        return

    chart = (
        alt.hconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                text="Best Sweep Config per Algorithm vs Adam Baseline",
                fontSize=15,
                fontWeight="bold",
            )
        )
        .configure_axis(grid=True, gridOpacity=0.3)
    )

    out_dir = Path(output_dir)
    for ext in ("html", "png"):
        save_path = out_dir / ext / f"sweep_best_configs.{ext}"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        chart.save(str(save_path))
        print(f"  Saved: {save_path}")


# ---- CLI --------------------------------------------------------------------

def main(
    algo: str = "",
    all: bool = False,
    top_k: int = 1,
    plot: bool = False,
    max_age_days: Optional[float] = 4,
    wandb_entity: str = "lucmc",
    sweep_project: str = "cont-minatar-sweep",
    baseline_project: str = "cont-minatar",
    tail_frac: float = 0.2,
    min_seeds: int = 1,
    output_csv: Optional[str] = None,
    output_dir: str = "./plots",
):
    """Compare sweep configs for an algorithm against the adam continual baseline.

    Args:
        algo: Algorithm name (redo, regrama, cbp, cpr, shrink_and_perturb).
              Ignored when --all is set.
        all: Compare best config from every algorithm side-by-side.
        top_k: Number of top configs per algorithm when using --all (default: 1).
        plot: Plot the best config from each algorithm (requires --all).
        max_age_days: Only include runs created within this many days. Set to
                      None to include all runs regardless of age.
        wandb_entity: W&B entity/username.
        sweep_project: W&B project containing sweep runs.
        baseline_project: W&B project containing adam baseline runs.
        tail_frac: Fraction of tail steps to average for "avg" metric.
        output_csv: Optional path to save results as CSV.
        output_dir: Directory to save plots (default: ./plots).
    """
    if not all and not algo:
        print("ERROR: provide --algo <name> or use --all")
        return

    print(f"Fetching adam baseline from {wandb_entity}/{baseline_project}...")
    adam_baseline = fetch_adam_baseline(wandb_entity, baseline_project, tail_frac, max_age_days)
    if not adam_baseline:
        print("ERROR: could not compute adam baseline. Exiting.")
        return

    if all:
        table = compare_all_algos(
            wandb_entity,
            sweep_project,
            adam_baseline,
            tail_frac,
            top_k,
            max_age_days,
            min_seeds=min_seeds,
        )
        print_all_table(table, adam_baseline, top_k)

        if plot and not table.empty:
            print(f"\nFetching time-series for best configs...")
            algo_configs = [
                (row["algorithm"], row["config"])
                for _, row in table.drop_duplicates("algorithm").iterrows()
            ]
            ts_sweep = fetch_timeseries_for_configs(
                wandb_entity, sweep_project, algo_configs, max_age_days,
            )
            ts_adam = fetch_adam_timeseries(
                wandb_entity, baseline_project, max_age_days,
            )
            ts_all = pd.concat([ts_adam, ts_sweep], ignore_index=True)
            print(f"\nPlotting {len(algo_configs)} algorithms + adam baseline...")
            plot_best_configs(ts_all, output_dir=output_dir)
    else:
        print(f"\nFetching sweep runs for {algo} from {wandb_entity}/{sweep_project}...")
        sweep_df = fetch_sweep_runs(wandb_entity, sweep_project, algo, tail_frac, max_age_days)
        if sweep_df.empty:
            print(f"ERROR: no sweep data found for {algo}. Exiting.")
            return

        print(f"\nSweep data: {len(sweep_df)} task-level measurements")
        print(f"Configs: {sweep_df['config'].nunique()}")
        print(f"Seeds: {sweep_df['seed'].unique().tolist()}")

        table = build_comparison_table(sweep_df, adam_baseline)
        if min_seeds > 1 and not table.empty:
            before = len(table)
            table = table[table["n_seeds"] >= min_seeds].reset_index(drop=True)
            print(f"\nFiltered by n_seeds>={min_seeds}: {len(table)}/{before} configs kept")
        print_table(table, algo, adam_baseline)

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        table.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    tyro.cli(main)
