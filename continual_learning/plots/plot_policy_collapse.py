#!/usr/bin/env python3
from __future__ import annotations
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import pandas as pd
import wandb

from analyze import ALGORITHM_DISPLAY_NAMES, display_algorithm_name, normalize_algorithm_name
from ablation_plot import build_algorithm_legend_domain

ALGORITHM_ALIAS_LOOKUP = {key.lower(): value for key, value in ALGORITHM_DISPLAY_NAMES.items()}


ALT_THEME_NAME = "times_new_roman_theme"
ALT_FONT_FAMILY = "Times New Roman"


def _times_new_roman_theme() -> dict:
    return {
        "config": {
            "title": {"font": ALT_FONT_FAMILY},
            "axis": {
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            },
            "header": {
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            },
            "legend": {
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            },
            "mark": {"font": ALT_FONT_FAMILY},
            "text": {"font": ALT_FONT_FAMILY},
        }
    }


def _enable_times_new_roman_theme() -> None:
    try:
        alt.themes.register(ALT_THEME_NAME, _times_new_roman_theme)
    except ValueError:
        pass
    alt.themes.enable(ALT_THEME_NAME)


_enable_times_new_roman_theme()


def resolve_algorithm_display(name: str) -> str:
    if isinstance(name, float) and np.isnan(name):
        name = ""
    elif not isinstance(name, str):
        name = "" if name is None else str(name)

    normalized = normalize_algorithm_name(name)
    lowered = normalized.lower()
    if lowered in ALGORITHM_ALIAS_LOOKUP:
        return ALGORITHM_ALIAS_LOOKUP[lowered]

    parts = lowered.split("_")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:end])
        if candidate in ALGORITHM_ALIAS_LOOKUP:
            return ALGORITHM_ALIAS_LOOKUP[candidate]

    return display_algorithm_name(normalized)


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
    finished = [r for r in runs if r.state == "finished"]
    print(f"Fetched {len(runs)} runs, {len(finished)} finished")
    return finished


def parse_algo_and_seed(run_name: str) -> tuple[str, str]:
    if not run_name:
        return ("unknown_algorithm", "0")

    parts = run_name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        algorithm_raw, seed_value = "_".join(parts[:-1]), parts[-1]
    else:
        algorithm_raw, seed_value = run_name, "0"

    algorithm_normalized = normalize_algorithm_name(algorithm_raw)
    return algorithm_normalized, seed_value


def detect_collapses_for_run(
    run,
    metric: str,
    threshold: float,
    bin_size: int,
    min_consecutive_below: int,
    display_name: str | None = None,
):
    hist = run.history(keys=[metric, "_step"])
    if hist.empty: return []

    hist = hist[["_step", metric]].dropna().sort_values("_step")
    if hist.empty: return []

    algo_name, seed_id = parse_algo_and_seed(run.name)
    display_algo = display_name if display_name is not None else resolve_algorithm_display(algo_name)
    peak, collapse_recorded, below_count, events = -np.inf, False, 0, []

    for step, value in hist[["_step", metric]].itertuples(index=False):
        try: step, value = int(step), float(value)
        except: continue

        if value > peak:
            peak, collapse_recorded, below_count = value, False, 0
            continue

        if value <= peak - threshold:
            below_count += 1
        else:
            below_count = 0

        if not collapse_recorded and below_count >= min_consecutive_below:
            binned_step = int(round(step / bin_size) * bin_size)
            print(
                f"Detected policy collapse for {run.name} (algo={algo_name}, seed={seed_id}) "
                f"after {below_count} consecutive drops at step={step} ({step / 1_000_000.0:.2f}M); "
                f"peak={peak:.2f}, value={value:.2f}"
            )
            events.append({
                "algorithm": algo_name,
                "algorithm_display": display_algo,
                "seed": seed_id,
                "run_name": run.name,
                "step": binned_step / 1_000_000.0,
            })
            collapse_recorded = True
            break

    return events


def build_events_df(entity: str, project: str, group: str, metric: str, collapse_threshold: float,
                   bin_size: int, min_consecutive_below: int) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    runs = fetch_runs(entity, project, group)
    all_events = []
    algorithms: set[str] = set()
    runs_per_algorithm: dict[str, int] = {}

    for run in runs:
        algo_name, _ = parse_algo_and_seed(run.name)
        display_algo = resolve_algorithm_display(algo_name)
        algorithms.add(display_algo)
        runs_per_algorithm[display_algo] = runs_per_algorithm.get(display_algo, 0) + 1
        try:
            if run.history(keys=[metric]).empty: continue
            events = detect_collapses_for_run(
                run,
                metric,
                collapse_threshold,
                bin_size,
                min_consecutive_below,
                display_name=display_algo,
            )
            all_events.extend(events)
        except: continue

    print(f"Total collapse events: {len(all_events)}")
    print(f"Runs per algorithm: {runs_per_algorithm}")
    events_df = (
        pd.DataFrame(all_events).sort_values(["algorithm", "run_name", "step"])
        if all_events
        else pd.DataFrame(columns=["algorithm", "algorithm_display", "seed", "run_name", "step"])
    )
    if not events_df.empty and "algorithm_display" not in events_df.columns:
        events_df["algorithm_display"] = events_df["algorithm"].map(resolve_algorithm_display)

    return events_df, build_algorithm_legend_domain(algorithms), runs_per_algorithm


def create_collapse_chart(df: pd.DataFrame, title: str, bin_size: int, stack_by_algorithm: bool = True) -> alt.Chart:
    bin_m = bin_size / 1_000_000.0
    df = df.copy()
    df['bin_start'] = df['step'] - bin_m / 2.0
    df['bin_end'] = df['step'] + bin_m / 2.0

    base = alt.Chart(df).encode(x=alt.X('bin_start:Q', title='Training Steps (Millions)'), x2='bin_end:Q')

    if stack_by_algorithm:
        legend_domain = build_algorithm_legend_domain(df['algorithm_display'])
        color_scale = alt.Scale(domain=legend_domain) if legend_domain else alt.Undefined
        base = base.encode(
            color=alt.Color(
                'algorithm_display:N',
                title='Algorithm',
                scale=color_scale,
            )
        )

    return (
        base.mark_bar()
        .encode(
            y=alt.Y('count():Q', title='Collapse Frequency'),
            tooltip=[
                alt.Tooltip('bin_start:Q', title='Bin start (M)', format='.2f'),
                alt.Tooltip('count():Q', title='# Collapses'),
            ],
        )
        .properties(width=1000, height=400, title=title)
        .configure_title(fontSize=24, font=ALT_FONT_FAMILY)
        .configure_axis(
            grid=True,
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .configure_axisX(labelFontSize=18, titleFontSize=16)
        .configure_axisY(labelFontSize=18, titleFontSize=18)
        .configure_legend(labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)
        .interactive()
    )


def create_overall_collapse_chart(
    df: pd.DataFrame,
    title: str,
    algorithms: list[str],
    runs_per_algorithm: dict[str, int] | None = None,
    show_percentage: bool = False,
) -> alt.Chart:
    grouped = df.groupby('algorithm_display', as_index=False).agg(collapse_count=('run_name', 'nunique'))
    if grouped.empty and algorithms:
        grouped = pd.DataFrame({
            'algorithm_display': algorithms,
            'collapse_count': [0] * len(algorithms),
        })
    else:
        grouped = grouped.sort_values('collapse_count', ascending=False)
        if algorithms:
            missing_algorithms = [algo for algo in algorithms if algo not in grouped['algorithm_display'].values]
            if missing_algorithms:
                grouped = pd.concat([
                    grouped,
                    pd.DataFrame({
                        'algorithm_display': missing_algorithms,
                        'collapse_count': [0] * len(missing_algorithms),
                    }),
                ], ignore_index=True)

    if grouped.empty:
        return alt.Chart(grouped).mark_bar()

    # Calculate percentage if requested
    if show_percentage and runs_per_algorithm:
        grouped['total_runs'] = grouped['algorithm_display'].map(
            lambda x: runs_per_algorithm.get(x, 1)
        )
        grouped['collapse_percentage'] = (grouped['collapse_count'] / grouped['total_runs']) * 100
        grouped = grouped.sort_values('collapse_percentage', ascending=False)
        y_field = 'collapse_percentage'
        y_title = 'Collapse Rate (%)'
        tooltip_field = alt.Tooltip('collapse_percentage:Q', title='Collapse %', format='.1f')
        max_value = float(grouped['collapse_percentage'].max())
        # For percentage, use nice round numbers
        if max_value <= 25:
            domain_max = 25
            axis_values = [0, 5, 10, 15, 20, 25]
        elif max_value <= 50:
            domain_max = 50
            axis_values = [0, 10, 20, 30, 40, 50]
        elif max_value <= 75:
            domain_max = 75
            axis_values = [0, 15, 30, 45, 60, 75]
        else:
            domain_max = 100
            axis_values = [0, 20, 40, 60, 80, 100]
    else:
        y_field = 'collapse_count'
        y_title = 'Collapse Frequency'
        tooltip_field = alt.Tooltip('collapse_count:Q', title='Collapses')
        max_value = int(grouped['collapse_count'].max())
        if max_value <= 5:
            axis_values = list(range(0, max_value + 1))
        else:
            tick_step = max(1, int(np.ceil(max_value / 5)))
            axis_values = list(range(0, max_value + tick_step, tick_step))
        domain_max = axis_values[-1] if axis_values else max_value

    if domain_max <= 0:
        domain_max = 1 if not show_percentage else 100

    category_order = grouped['algorithm_display'].tolist()

    # Use consistent color scheme based on algorithm legend order
    legend_domain = build_algorithm_legend_domain(grouped['algorithm_display'])
    color_scale = alt.Scale(domain=legend_domain) if legend_domain else alt.Undefined

    tooltips = [
        alt.Tooltip('algorithm_display:N', title='Algorithm'),
        tooltip_field,
    ]
    if show_percentage and runs_per_algorithm:
        tooltips.extend([
            alt.Tooltip('collapse_count:Q', title='Collapses'),
            alt.Tooltip('total_runs:Q', title='Total Runs'),
        ])

    return (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X('algorithm_display:N', title=None, sort=category_order),
            y=alt.Y(
                f'{y_field}:Q',
                title=y_title,
                scale=alt.Scale(domain=(0, domain_max)),
                axis=alt.Axis(values=axis_values),
            ),
            color=alt.Color('algorithm_display:N', legend=None, scale=color_scale),
            tooltip=tooltips,
        )
        .properties(width=750, height=500, title=title)
        .configure_title(fontSize=28, font=ALT_FONT_FAMILY)
        .configure_axis(
            grid=True,
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .configure_axisX(labelFontSize=28, titleFontSize=30, labelLimit=0)
        .configure_axisY(labelFontSize=28, titleFontSize=30)
        .configure_legend(labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)
    )


def main(wandb_entity: str, wandb_project: str = "crl_experiments", group: str = "default_group",
         metric: str = "charts/mean_episodic_return", collapse_threshold: float = 3000.0,
         bin_size: int = 10_000_000, stack_by_algorithm: bool = True,
         min_consecutive_below: int = 3,
         output_dir: str = "./plots", save_html: bool = False, debug: bool = False,
         overall: bool = False, output_name: str | None = None,
         percentage: bool = False):

    if debug:
        print("Debug mode enabled: using synthetic collapse events instead of querying Weights & Biases.")
        df_events = pd.DataFrame([
            {"algorithm": "alg_a", "seed": "0", "run_name": "alg_a_0", "step": 12.0},
            {"algorithm": "alg_a", "seed": "1", "run_name": "alg_a_1", "step": 18.0},
            {"algorithm": "alg_b", "seed": "0", "run_name": "alg_b_0", "step": 7.0},
        ])
        df_events['algorithm_display'] = df_events['algorithm'].map(resolve_algorithm_display)
        algorithms = sorted(df_events['algorithm_display'].unique())
        runs_per_algorithm = {"alg_a": 5, "alg_b": 5}
    else:
        df_events, algorithms, runs_per_algorithm = build_events_df(
            wandb_entity,
            wandb_project,
            group,
            metric,
            collapse_threshold,
            bin_size,
            min_consecutive_below,
        )

    if df_events.empty:
        if overall and algorithms:
            print(f"No collapse events found for group='{group}', but plotting zero counts for all algorithms.")
        else:
            return print(f"No collapse events found for group='{group}'")

    if overall:
        if percentage:
            print("Generating overall collapse percentage chart (percentage of runs collapsed).")
            title = f"Collapse Rate"
        else:
            print("Generating overall collapse frequency chart (one collapse per run).")
            title = f"Collapse Frequency"
        chart = create_overall_collapse_chart(
            df_events,
            title,
            algorithms if algorithms else [],
            runs_per_algorithm=runs_per_algorithm,
            show_percentage=percentage,
        )
    else:
        print("Generating collapse timeline chart (binned by training steps).")
        title = f"{group.replace('_', ' ').title()}: Policy Collapses Over Time"
        chart = create_collapse_chart(df_events, title, bin_size, stack_by_algorithm)

    ext = "html" if save_html else "png"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    if percentage:
        suffix = "overall_percentage" if overall else "binned"
    else:
        suffix = "overall" if overall else "binned"
    name_prefix = output_name if output_name else group
    outfile = Path(output_dir) / f"{name_prefix}_policy_collapses_{suffix}.{ext}"
    chart.save(str(outfile))
    print(f"Chart saved to: {outfile}")


if __name__ == "__main__":
    tyro.cli(main)
