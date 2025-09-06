#!/usr/bin/env python3
"""
Plot binned bar chart of policy collapses over training time.

Definition: A policy collapse occurs when the mean episodic return drops by at least
`collapse_threshold` from its running peak. By default, we count at most one collapse
per peak plateau; if the run later achieves a new peak and then drops by the threshold
again, that is counted as another collapse.

This script mirrors how runs are fetched and parsed in analyze.py: it reads all runs
in a given W&B group, extracts the metric history with steps, detects collapse events,
bins them by step, and plots collapses over time as a bar chart.
"""

from __future__ import annotations

import tyro
import numpy as np
import altair as alt
from pathlib import Path
import pandas as pd
from collections import defaultdict
import wandb


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    filters = {"group": group}
    runs = api.runs(f"{entity}/{project}", filters=filters, per_page=300)
    all_runs = list(runs)

    print(f"Total runs fetched: {len(all_runs)}")
    run_states: dict[str, int] = defaultdict(int)
    for run in all_runs:
        run_states[run.state] += 1
    print(f"Run states: {dict(run_states)}")

    return [x for x in all_runs if x.state == "finished"]


def parse_algo_and_seed(run_name: str) -> tuple[str, str]:
    parts = run_name.split("_")
    if len(parts) >= 2:
        algo_name = "_".join(parts[:-1])
        seed_id = parts[-1]
    else:
        algo_name = run_name
        seed_id = "0"
    return algo_name, seed_id


def detect_collapses_for_run(
    run,
    metric: str,
    threshold: float,
    bin_size: int,
    allow_multiple: bool,
):
    """Yield collapse events as dicts with keys: algorithm, seed, run_name, step_bin, step_m.

    Collapse detection logic:
    - Maintain a running peak of the metric.
    - When the value first drops below (peak - threshold) for a given peak plateau,
      count one collapse. If allow_multiple is True, we wait for a new higher peak to
      be established before allowing another collapse to be counted.
    - Steps are sourced from the W&B history column "_step" and binned to `bin_size`.
    """
    hist = run.history(keys=[metric, "_step"])  # DataFrame
    if hist.empty:
        return []

    # Drop rows with NaN in either column and ensure sorted by step
    hist = hist[["_step", metric]].dropna().sort_values("_step")
    if hist.empty:
        return []

    algo_name, seed_id = parse_algo_and_seed(run.name)

    peak = -np.inf
    collapsed_after_peak = False  # one collapse per peak plateau
    events = []

    for step, value in hist[["_step", metric]].itertuples(index=False):
        try:
            step = int(step)
            value = float(value)
        except Exception:
            continue

        if value > peak:
            peak = value
            collapsed_after_peak = False
            continue

        if not collapsed_after_peak and (peak - value) >= threshold:
            # Record one collapse event for this peak plateau
            binned_step = int(round(step / bin_size) * bin_size)
            step_m = binned_step / 1_000_000.0
            events.append(
                {
                    "algorithm": algo_name,
                    "seed": seed_id,
                    "run_name": run.name,
                    "step": step_m,
                }
            )
            collapsed_after_peak = True

            # If multiple collapses per run are not desired, break after first
            if not allow_multiple:
                break

    return events


def build_events_df(
    entity: str,
    project: str,
    group: str,
    metric: str,
    collapse_threshold: float,
    bin_size: int,
    allow_multiple: bool,
) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group)
    all_events = []

    total_runs = 0
    with_metric = 0
    for run in runs:
        total_runs += 1
        try:
            hist = run.history(keys=[metric])
            if hist.empty:
                continue
            with_metric += 1
        except Exception:
            continue

        events = detect_collapses_for_run(
            run, metric=metric, threshold=collapse_threshold, bin_size=bin_size, allow_multiple=allow_multiple
        )
        if events:
            all_events.extend(events)

    print(f"Runs processed: {total_runs}; runs with metric '{metric}': {with_metric}")
    print(f"Total collapse events: {len(all_events)}")

    if not all_events:
        return pd.DataFrame(columns=["algorithm", "seed", "run_name", "step"])  # empty

    df = pd.DataFrame(all_events)
    df = df.sort_values(["algorithm", "run_name", "step"])  # step already in millions
    return df


def create_collapse_chart(
    df: pd.DataFrame,
    title: str,
    bin_size: int,
    stack_by_algorithm: bool = True
) -> alt.Chart:
    bin_m = bin_size / 1_000_000.0
    df = df.copy()
    # Your df["step"] is a bin *center* (rounded to a multiple). Build edges around it.
    df['bin_start'] = df['step'] - bin_m / 2.0
    df['bin_end']   = df['step'] + bin_m / 2.0

    base = alt.Chart(df).encode(
        x=alt.X('bin_start:Q', title='Training Steps (Millions)', scale=alt.Scale(nice=False)),
        x2='bin_end:Q',
    )

    if stack_by_algorithm:
        base = base.encode(
            color=alt.Color(
                'algorithm:N',
                title='Algorithm',
                legend=alt.Legend(
                    title=None,
                    orient='bottom-left',
                    fillColor='rgba(255,255,255,1)',
                    strokeColor='gray',
                    padding=5,
                    cornerRadius=3,
                ),
            )
        )

    tooltip_fields = [
        alt.Tooltip('bin_start:Q', title='Bin start (M)', format='.2f'),
        alt.Tooltip('bin_end:Q', title='Bin end (M)', format='.2f'),
        alt.Tooltip('count():Q', title='# Collapses'),
    ]
    if stack_by_algorithm:
        tooltip_fields.append(alt.Tooltip('algorithm:N', title='Algorithm'))

    chart = (
        base.mark_bar()
            .encode(y=alt.Y('count():Q', title='Policy Collapses'),
                    tooltip=tooltip_fields)
            .properties(width=1000, height=400, title=title)
            .configure_axis(grid=True, gridOpacity=0.3)
            .configure_bar(binSpacing=0)
            .interactive()
    )
    return chart


def main(
    wandb_entity: str,
    wandb_project: str = "crl_experiments",
    group: str | None = "default_group",
    metric: str = "charts/mean_episodic_return",
    collapse_threshold: float = 3000.0,
    bin_size: int = 10_000_000,
    allow_multiple: bool = True,
    stack_by_algorithm: bool = True,
    output_dir: str = "./plots",
    save_html: bool = False,
    debug: bool = False,
):
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    print(
        f"Fetching collapse events for group='{group}', metric='{metric}', threshold={collapse_threshold}, bin_size={bin_size}"
    )
    df_events = build_events_df(
        wandb_entity,
        wandb_project,
        group,
        metric,
        collapse_threshold,
        bin_size,
        allow_multiple,
    )

    if df_events.empty:
        print(
            f"No collapse events found for group='{group}' and metric='{metric}'.\n"
            "Check that runs exist and the metric is logged."
        )
        return

    title = f"{group.replace('_', ' ').title()}: Policy Collapses Over Time"
    chart = create_collapse_chart(df_events, title=title, bin_size=bin_size, stack_by_algorithm=stack_by_algorithm)

    ext = "html" if save_html else "svg"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    outfile = Path(output_dir) / f"{group}_policy_collapses_binned.{ext}"
    chart.save(str(outfile))
    print(f"\n✅ Chart saved to: {outfile}")


if __name__ == "__main__":
    tyro.cli(main)
