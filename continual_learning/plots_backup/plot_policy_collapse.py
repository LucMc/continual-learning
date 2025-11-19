#!/usr/bin/env python3
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
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
    finished = [r for r in runs if r.state == "finished"]
    print(f"Fetched {len(runs)} runs, {len(finished)} finished")
    return finished


def parse_algo_and_seed(run_name: str) -> tuple[str, str]:
    parts = run_name.split("_")
    return ("_".join(parts[:-1]), parts[-1]) if len(parts) >= 2 else (run_name, "0")


def detect_collapses_for_run(run, metric: str, threshold: float, bin_size: int, allow_multiple: bool):
    hist = run.history(keys=[metric, "_step"])
    if hist.empty: return []

    hist = hist[["_step", metric]].dropna().sort_values("_step")
    if hist.empty: return []

    algo_name, seed_id = parse_algo_and_seed(run.name)
    peak, collapsed_after_peak, events = -np.inf, False, []

    for step, value in hist[["_step", metric]].itertuples(index=False):
        try: step, value = int(step), float(value)
        except: continue

        if value > peak:
            peak, collapsed_after_peak = value, False
            continue

        if not collapsed_after_peak and (peak - value) >= threshold:
            binned_step = int(round(step / bin_size) * bin_size)
            events.append({"algorithm": algo_name, "seed": seed_id, "run_name": run.name, "step": binned_step / 1_000_000.0})
            collapsed_after_peak = True
            if not allow_multiple: break

    return events


def build_events_df(entity: str, project: str, group: str, metric: str, collapse_threshold: float,
                   bin_size: int, allow_multiple: bool) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group)
    all_events = []

    for run in runs:
        try:
            if run.history(keys=[metric]).empty: continue
            events = detect_collapses_for_run(run, metric, collapse_threshold, bin_size, allow_multiple)
            all_events.extend(events)
        except: continue

    print(f"Total collapse events: {len(all_events)}")
    return pd.DataFrame(all_events).sort_values(["algorithm", "run_name", "step"]) if all_events else pd.DataFrame(columns=["algorithm", "seed", "run_name", "step"])


def create_collapse_chart(df: pd.DataFrame, title: str, bin_size: int, stack_by_algorithm: bool = True) -> alt.Chart:
    bin_m = bin_size / 1_000_000.0
    df = df.copy()
    df['bin_start'] = df['step'] - bin_m / 2.0
    df['bin_end'] = df['step'] + bin_m / 2.0

    base = alt.Chart(df).encode(x=alt.X('bin_start:Q', title='Training Steps (Millions)'), x2='bin_end:Q')

    if stack_by_algorithm:
        base = base.encode(color=alt.Color('algorithm:N', title='Algorithm'))

    return base.mark_bar().encode(
        y=alt.Y('count():Q', title='Policy Collapses'),
        tooltip=[alt.Tooltip('bin_start:Q', title='Bin start (M)', format='.2f'),
                alt.Tooltip('count():Q', title='# Collapses')]
    ).properties(width=1000, height=400, title=title).configure_axis(grid=True).interactive()


def main(wandb_entity: str, wandb_project: str = "crl_experiments", group: str = "default_group",
         metric: str = "charts/mean_episodic_return", collapse_threshold: float = 3000.0,
         bin_size: int = 10_000_000, allow_multiple: bool = True, stack_by_algorithm: bool = True,
         output_dir: str = "./plots", save_html: bool = False, debug: bool = False):

    df_events = build_events_df(wandb_entity, wandb_project, group, metric, collapse_threshold, bin_size, allow_multiple)

    if df_events.empty:
        return print(f"No collapse events found for group='{group}'")

    title = f"{group.replace('_', ' ').title()}: Policy Collapses Over Time"
    chart = create_collapse_chart(df_events, title, bin_size, stack_by_algorithm)

    ext = "html" if save_html else "svg"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    outfile = Path(output_dir) / f"{group}_policy_collapses_binned.{ext}"
    chart.save(str(outfile))
    print(f"Chart saved to: {outfile}")


if __name__ == "__main__":
    tyro.cli(main)
