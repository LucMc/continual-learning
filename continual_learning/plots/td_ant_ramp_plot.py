#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


DEFAULT_ENTITY = "lucmc"
DEFAULT_PROJECT = "TD Ant ramp"
DEFAULT_GROUP = "ramp25_obs13_act12_info-none"
DEFAULT_METRIC = "charts/mean_episodic_return"
DEFAULT_OUTPUT_DIR = Path("plots/td_ant_ramp")
EXCLUDED_METHODS = {"cbp", "cbpl", "cbplrr", "cpr", "cprH", "cprL", "cprLuf", "cprVLuf"}
CPR_METHODS = {"cprVL"}

METHOD_ORDER = [
    "cprVL",
    "cbplrrN",
    "regrama",
    "redo",
    "shrink_and_perturb",
    "adam",
]
METHOD_LABELS = {
    "adam": "Adam",
    "cbplrr": "CBP-LRR",
    "cbplrrN": "CBP",
    "cpr": "CPR",
    "cprH": "CPR-H",
    "cprL": "CPR-L",
    "cprVL": "CPR",
    "redo": "ReDo",
    "regrama": "ReGraMa",
    "shrink_and_perturb": "Shrink & Perturb",
}
METHOD_COLORS = {
    "adam": "#EECA3B",
    "cbplrr": "#F58518",
    "cbplrrN": "#F58518",
    "cpr": "#4C78A8",
    "cprH": "#7F7F7F",
    "cprL": "#9467BD",
    "cprVL": "#4C78A8",
    "redo": "#72B7B2",
    "regrama": "#E45756",
    "shrink_and_perturb": "#54A24B",
}


def parse_method_and_seed(run_name: str, group: str) -> tuple[str, int]:
    pattern = rf"^td_ant_(?P<method>.+)_{re.escape(group)}_s(?P<seed>\d+)$"
    match = re.match(pattern, run_name)
    if match:
        return match.group("method"), int(match.group("seed"))

    fallback = re.match(r"^td_ant_(?P<method>.+)_s(?P<seed>\d+)$", run_name)
    if fallback:
        return fallback.group("method"), int(fallback.group("seed"))

    raise ValueError(f"Could not parse method and seed from run name: {run_name}")


def fetch_raw_history(
    entity: str,
    project: str,
    group: str,
    metric: str,
    samples: int,
) -> pd.DataFrame:
    api = wandb.Api()
    runs = list(
        api.runs(
            f"{entity}/{project}",
            filters={"group": group},
            per_page=200,
        )
    )
    finished_runs = [run for run in runs if run.state == "finished"]
    if not finished_runs:
        raise RuntimeError(f"No finished runs found for {entity}/{project}, group={group!r}")

    rows: list[pd.DataFrame] = []
    for run in sorted(finished_runs, key=lambda r: r.name):
        method, seed = parse_method_and_seed(run.name, group)
        if method in EXCLUDED_METHODS:
            print(f"Skipping {run.name}: excluded method {method}")
            continue

        history = run.history(keys=[metric], samples=samples)
        if history.empty or metric not in history.columns:
            print(f"Skipping {run.name}: no history for {metric}")
            continue

        run_df = (
            history[["_step", metric]]
            .rename(columns={"_step": "step", metric: "mean_episodic_return"})
            .dropna()
            .copy()
        )
        run_df["method"] = method
        run_df["method_label"] = METHOD_LABELS.get(method, method.replace("_", " ").title())
        run_df["seed"] = seed
        run_df["run_name"] = run.name
        rows.append(run_df)
        print(
            f"{run.name}: {len(run_df)} points, "
            f"steps {int(run_df['step'].min())}..{int(run_df['step'].max())}"
        )

    if not rows:
        raise RuntimeError(f"No usable {metric} history found in finished runs")

    return pd.concat(rows, ignore_index=True)


def filter_excluded_methods(raw_df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = raw_df[~raw_df["method"].isin(EXCLUDED_METHODS)].copy()
    filtered_df["method_label"] = filtered_df["method"].map(METHOD_LABELS).fillna(
        filtered_df["method_label"]
    )
    return filtered_df


def ordered_methods(methods: set[str]) -> list[str]:
    return [
        method
        for method in METHOD_ORDER
        if method in methods
    ] + [
        method
        for method in sorted(methods)
        if method not in METHOD_ORDER
    ]


def aggregate_by_method(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = filter_excluded_methods(raw_df)
    agg = (
        raw_df.groupby(["method", "method_label", "step"], as_index=False)
        .agg(
            mean_return=("mean_episodic_return", "mean"),
            std_return=("mean_episodic_return", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "step"])
    )
    agg["std_return"] = agg["std_return"].fillna(0.0)
    agg["sem_return"] = agg["std_return"] / np.sqrt(agg["n_seeds"])
    agg["step_millions"] = agg["step"] / 1_000_000
    return agg


def plot_aggregate(
    agg_df: pd.DataFrame,
    output_dir: Path,
    task_steps: int,
    smooth_window: int,
    filename_suffix: str = "",
    x_max_millions: float | None = None,
    title_suffix: str = "",
) -> tuple[Path, Path]:
    agg_df = filter_excluded_methods(agg_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"td_ant_ramp_mean_episodic_return{filename_suffix}.png"
    svg_path = output_dir / f"td_ant_ramp_mean_episodic_return{filename_suffix}.svg"

    fig, ax = plt.subplots(figsize=(12, 6.5))

    max_step_millions = float(agg_df["step_millions"].max())
    x_limit_millions = (
        min(max_step_millions, x_max_millions)
        if x_max_millions is not None
        else max_step_millions
    )
    if task_steps > 0:
        task_step_millions = task_steps / 1_000_000
        for boundary in np.arange(task_step_millions, x_limit_millions, task_step_millions):
            ax.axvline(boundary, color="#CCCCCC", linewidth=0.6, alpha=0.35, zorder=0)

    plotted_methods = ordered_methods(set(agg_df["method"]))

    for method in plotted_methods:
        method_df = agg_df[agg_df["method"] == method].sort_values("step")
        if x_max_millions is not None:
            method_df = method_df[method_df["step_millions"] <= x_limit_millions]
        if method_df.empty:
            continue

        x = method_df["step_millions"].to_numpy()
        y = method_df["mean_return"].to_numpy()
        sem = method_df["sem_return"].to_numpy()

        if smooth_window > 1:
            y = (
                pd.Series(y)
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            sem = (
                pd.Series(sem)
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

        label = METHOD_LABELS.get(method, method.replace("_", " ").title())
        color = METHOD_COLORS.get(method)
        line_zorder = 5 if method in CPR_METHODS else 3
        ax.fill_between(
            x,
            y - sem,
            y + sem,
            color=color,
            alpha=0.20,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x, y, label=label, color=color, linewidth=1.6, zorder=line_zorder)

    ax.set_title(f"Mean Episodic Return TD Ant Ramp{title_suffix}", fontsize=18, pad=12)
    ax.set_xlabel("Training Steps (Millions)", fontsize=14)
    ax.set_ylabel("Episode Return", fontsize=14)
    ax.set_xlim(left=0, right=x_limit_millions)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.75)
    ax.grid(True, axis="x", color="#EEEEEE", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, prop={"weight": "bold", "size": 12})
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    subtitle = "Line: mean across seeds; band: +/- standard error"
    if smooth_window > 1:
        subtitle += f"; {smooth_window}-point centered rolling smoothing"
    fig.text(0.125, 0.02, subtitle, fontsize=10, color="#555555")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def build_interval_comparison(
    raw_df: pd.DataFrame,
    interval_millions: float,
) -> pd.DataFrame:
    raw_df = filter_excluded_methods(raw_df)
    interval_steps = int(interval_millions * 1_000_000)
    if interval_steps <= 0:
        raise ValueError("interval_millions must be positive")

    interval_df = raw_df.copy()
    interval_df["interval_index"] = np.ceil(interval_df["step"] / interval_steps).astype(int)
    interval_df["interval_index"] = interval_df["interval_index"].clip(lower=1)
    interval_df["interval_start_millions"] = (
        interval_df["interval_index"] - 1
    ) * interval_millions
    interval_df["interval_end_millions"] = (
        interval_df["interval_index"] * interval_millions
    )

    per_seed = (
        interval_df.groupby(
            [
                "method",
                "method_label",
                "seed",
                "interval_start_millions",
                "interval_end_millions",
            ],
            as_index=False,
        )["mean_episodic_return"]
        .mean()
        .rename(columns={"mean_episodic_return": "seed_average_return"})
    )

    comparison = (
        per_seed.groupby(
            [
                "interval_start_millions",
                "interval_end_millions",
                "method",
                "method_label",
            ],
            as_index=False,
        )
        .agg(
            average_return=("seed_average_return", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["interval_end_millions", "method"])
    )
    return comparison


def _format_interval(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def render_interval_comparison_markdown(comparison_df: pd.DataFrame) -> str:
    methods = ordered_methods(set(comparison_df["method"]))
    method_labels = {
        row.method: row.method_label
        for row in comparison_df[["method", "method_label"]].drop_duplicates().itertuples()
    }

    rows = ["| Interval (M steps) | " + " | ".join(method_labels[m] for m in methods) + " |"]
    rows.append("| --- | " + " | ".join("---:" for _ in methods) + " |")

    intervals = (
        comparison_df[["interval_start_millions", "interval_end_millions"]]
        .drop_duplicates()
        .sort_values(["interval_end_millions", "interval_start_millions"])
    )
    for interval in intervals.itertuples(index=False):
        interval_df = comparison_df[
            (comparison_df["interval_start_millions"] == interval.interval_start_millions)
            & (comparison_df["interval_end_millions"] == interval.interval_end_millions)
        ]
        values = {
            row.method: row.average_return
            for row in interval_df[["method", "average_return"]].itertuples()
        }
        best_value = max(values.values())
        interval_label = (
            f"{_format_interval(interval.interval_start_millions)}-"
            f"{_format_interval(interval.interval_end_millions)}"
        )

        cells = []
        for method in methods:
            value = values.get(method)
            if value is None:
                cells.append("")
                continue

            formatted = f"{value:,.0f}"
            if np.isclose(value, best_value):
                formatted = f"**{formatted}**"
            cells.append(formatted)

        rows.append("| " + interval_label + " | " + " | ".join(cells) + " |")

    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TD Ant ramp mean episodic return by method over training steps."
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--task-steps", type=int, default=40_000_000)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--crop-step-millions", type=float, default=500.0)
    parser.add_argument("--table-interval-millions", type=float, default=200.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    raw_df = fetch_raw_history(
        entity=args.entity,
        project=args.project,
        group=args.group,
        metric=args.metric,
        samples=args.samples,
    )
    raw_df = filter_excluded_methods(raw_df)
    agg_df = aggregate_by_method(raw_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "td_ant_ramp_mean_episodic_return_raw.csv"
    agg_path = args.output_dir / "td_ant_ramp_mean_episodic_return_aggregate.csv"
    interval_suffix = f"{args.table_interval_millions:g}".replace(".", "p")
    comparison_path = args.output_dir / f"td_ant_ramp_{interval_suffix}m_interval_comparison.csv"
    comparison_md_path = args.output_dir / f"td_ant_ramp_{interval_suffix}m_interval_comparison.md"
    raw_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    comparison_df = build_interval_comparison(raw_df, args.table_interval_millions)
    comparison_df.to_csv(comparison_path, index=False)
    comparison_markdown = render_interval_comparison_markdown(comparison_df)
    comparison_md_path.write_text(comparison_markdown)
    full_png_path, full_svg_path = plot_aggregate(
        agg_df=agg_df,
        output_dir=args.output_dir,
        task_steps=args.task_steps,
        smooth_window=args.smooth_window,
    )
    crop_png_path, crop_svg_path = plot_aggregate(
        agg_df=agg_df,
        output_dir=args.output_dir,
        task_steps=args.task_steps,
        smooth_window=args.smooth_window,
        filename_suffix=f"_{int(args.crop_step_millions)}m",
        x_max_millions=args.crop_step_millions,
        title_suffix=f" (0-{args.crop_step_millions:g}M steps)",
    )

    print("\nSeeds per method:")
    for method, method_df in raw_df.groupby("method"):
        seeds = [int(seed) for seed in sorted(method_df["seed"].unique())]
        label = METHOD_LABELS.get(method, method)
        print(f"  {label}: {len(seeds)} seeds {seeds}")

    print(f"\nAverage return by {args.table_interval_millions:g}M-step interval:")
    print(comparison_markdown)

    print("\nSaved:")
    print(f"  raw CSV:        {raw_path}")
    print(f"  aggregate CSV:  {agg_path}")
    print(f"  comparison CSV: {comparison_path}")
    print(f"  comparison MD:  {comparison_md_path}")
    print(f"  full PNG:       {full_png_path}")
    print(f"  full SVG:       {full_svg_path}")
    print(f"  cropped PNG:    {crop_png_path}")
    print(f"  cropped SVG:    {crop_svg_path}")


if __name__ == "__main__":
    main()
