#!/usr/bin/env python3
"""Generate Continual World-style plots for continual MetaWorld experiments.

Produces:
  - Average success rate over time (main CL visualization)
  - Per-task success grid showing forgetting
  - Current-task success rate
  - Summary bar chart
  - Sweep top-K config plots (for CCBP / ReGraMa hyperparameter sweeps)

Usage:
    # Full CL runs - all plot types
    python -m continual_learning.plots.plot_continual \
      --wandb-entity lucmc --wandb-project metaworld_sac --plot-type all

    # Just the average success plot
    python -m continual_learning.plots.plot_continual \
      --wandb-entity lucmc --wandb-project metaworld_sac --plot-type average

    # Sweep top-K
    python -m continual_learning.plots.plot_continual \
      --wandb-entity lucmc --wandb-project crl_experiments \
      --group metaworld_sac_ccbp_sweep --plot-type sweep --top-k 5
"""
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
import re
from collections import defaultdict
from typing import Optional, Literal, List, Dict

try:
    from ablation_plot import (
        ALT_FONT_FAMILY,
        scaled_font_size,
        coerce_numeric_value,
        compute_iqm,
        parse_run_name,
        build_algorithm_legend_domain,
        summarize_group_performance,
        slugify_title,
        _enable_times_new_roman_theme,
        aggregate_data,
        create_ablation_chart,
        fetch_ablation_data,
    )
except ImportError:
    from .ablation_plot import (
        ALT_FONT_FAMILY,
        scaled_font_size,
        coerce_numeric_value,
        compute_iqm,
        parse_run_name,
        build_algorithm_legend_domain,
        summarize_group_performance,
        slugify_title,
        _enable_times_new_roman_theme,
        aggregate_data,
        create_ablation_chart,
        fetch_ablation_data,
    )

_enable_times_new_roman_theme()

# ---------------------------------------------------------------------------
# Algorithm name mapping
# ---------------------------------------------------------------------------
ALGO_DISPLAY: Dict[str, str] = {
    "adam": "Adam",
    "standard": "Adam",
    "redo": "ReDo",
    "regrama": "ReGraMa",
    "cbp": "CBP",
    "ccbp": "CPR",
    "ccbp3": "CPR",
    "ccbph": "CPR-H",
    "ccbpl": "CPR-L",
    "cpr": "CPR",
    "shrink_and_perturb": "Shrink & Perturb",
    "soft_shrink_and_perturb": "Soft Shrink & Perturb",
    "sp": "Shrink & Perturb",
    "bro": "BRO",
    "sac": "SAC",
}

EXCLUDED_ALGORITHMS = {"ccbp2"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_algo(name: str) -> str:
    """Lowercase, strip seed suffixes."""
    n = name.strip().lower()
    n = re.sub(r"_s\d+$", "", n)
    n = re.sub(r"[,_]seed=[^,_]+", "", n)
    n = re.sub(r"_\d+$", "", n)
    return re.sub(r"[,_-]+$", "", n)


def _display_algo(name: str) -> str:
    norm = _normalize_algo(name)
    return ALGO_DISPLAY.get(norm, name.replace("_", " ").title())


def _extract_algo_from_run(run) -> str:
    """Extract algorithm/optimizer from run name.

    Continual MetaWorld names look like: sac_{optimizer}_{seed}
    """
    run_name = getattr(run, "name", "") or ""
    run_lower = run_name.lower()

    # Check multi-word algorithms first
    for mw in ("shrink_and_perturb", "soft_shrink_and_perturb"):
        if mw in run_lower:
            return mw

    # Pattern: sac_<algo>_<seed> or sac2m_<algo>_<seed>
    match = re.match(r"^sac(?:2m)?_([a-zA-Z_]+?)_(\d+)$", run_name)
    if match:
        return match.group(1)

    # Fallback: second segment (skip sac/sac2m prefix)
    parts = run_name.split("_")
    if len(parts) >= 3 and parts[0] in ("sac", "sac2m"):
        return parts[1]
    if len(parts) >= 2:
        return parts[1]

    return run_name


def _extract_seed_from_run(run) -> str:
    config = getattr(run, "config", {}) or {}
    if "seed" in config:
        return str(config["seed"])
    run_name = getattr(run, "name", "") or ""
    m = re.search(r"_(\d+)$", run_name)
    return m.group(1) if m else "0"


def _task_boundaries(steps_per_task: int, num_tasks: int) -> List[float]:
    """Return task boundary positions in millions of steps."""
    return [i * steps_per_task / 1e6 for i in range(1, num_tasks)]


def _boundary_rules(steps_per_task: int, num_tasks: int) -> alt.Chart:
    """Create vertical dashed rules at task boundaries."""
    boundaries = _task_boundaries(steps_per_task, num_tasks)
    rules_df = pd.DataFrame({"boundary": boundaries})
    return alt.Chart(rules_df).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.6
    ).encode(x="boundary:Q")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def _get_steps_per_task(run) -> Optional[int]:
    """Extract steps_per_task from run config (nested under 'training')."""
    cfg = getattr(run, "config", {}) or {}
    training = cfg.get("training", {})
    val = training.get("steps_per_task", cfg.get("steps_per_task", None))
    return int(val) if val is not None else None


def fetch_continual_data(
    entity: str,
    project: str,
    metric: str,
    group: Optional[str] = None,
    include_failed: bool = False,
    run_prefix: Optional[str] = None,
    filter_steps_per_task: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch W&B runs and return DataFrame[algorithm, seed, step, value]."""
    api = wandb.Api()
    filters = {"group": group} if group else None
    runs = list(api.runs(f"{entity}/{project}", filters=filters, per_page=500))

    if include_failed:
        valid = [r for r in runs if r.state in ("finished", "failed", "crashed", "running")]
    else:
        valid = [r for r in runs if r.state == "finished"]

    if run_prefix:
        valid = [r for r in valid if (getattr(r, "name", "") or "").startswith(run_prefix)]

    if filter_steps_per_task is not None:
        valid = [r for r in valid if _get_steps_per_task(r) == filter_steps_per_task]

    print(f"Fetched {len(runs)} runs, {len(valid)} valid (group={group}, prefix={run_prefix}, spt={filter_steps_per_task})")

    records: List[dict] = []
    for run in valid:
        algo_raw = _extract_algo_from_run(run)
        if _normalize_algo(algo_raw) in EXCLUDED_ALGORITHMS:
            continue
        algo = _display_algo(algo_raw)
        seed = _extract_seed_from_run(run)

        history = run.history(keys=[metric, "_step"], samples=5000)
        if history.empty:
            continue

        count = 0
        for _, row in history.iterrows():
            step = row.get("_step")
            val = coerce_numeric_value(row.get(metric))
            if pd.notna(step) and val is not None:
                records.append({"algorithm": algo, "seed": seed, "step": int(step), "value": float(val)})
                count += 1

        if count > 0:
            print(f"  {run.name}: {count} pts (algo={algo}, seed={seed})")

    return pd.DataFrame(records)


def fetch_per_task_data(
    entity: str,
    project: str,
    group: Optional[str] = None,
    include_failed: bool = False,
    run_prefix: Optional[str] = None,
    filter_steps_per_task: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch per-task eval success rates. Returns DataFrame[algorithm, seed, step, task, value]."""
    api = wandb.Api()
    filters = {"group": group} if group else None
    runs = list(api.runs(f"{entity}/{project}", filters=filters, per_page=500))

    if include_failed:
        valid = [r for r in runs if r.state in ("finished", "failed", "crashed", "running")]
    else:
        valid = [r for r in runs if r.state == "finished"]

    if run_prefix:
        valid = [r for r in valid if (getattr(r, "name", "") or "").startswith(run_prefix)]

    if filter_steps_per_task is not None:
        valid = [r for r in valid if _get_steps_per_task(r) == filter_steps_per_task]

    print(f"Fetching per-task data: {len(valid)} valid runs")

    # Detect task keys by scanning runs until we find the full set
    task_keys: List[str] = []
    for run in valid[:20]:
        hist = run.history(samples=2)
        if hist.empty:
            continue
        for col in hist.columns:
            if re.match(r"eval/.+/success_rate", col) and col not in task_keys:
                task_keys.append(col)
        # Stop once we've found a reasonable number of tasks
        if len(task_keys) >= 9:
            break

    if not task_keys:
        print("  No per-task eval keys found")
        return pd.DataFrame()

    task_names = [k.split("/")[1] for k in task_keys]
    print(f"  Found {len(task_keys)} task keys: {task_names}")

    records: List[dict] = []
    for run in valid:
        algo_raw = _extract_algo_from_run(run)
        if _normalize_algo(algo_raw) in EXCLUDED_ALGORITHMS:
            continue
        algo = _display_algo(algo_raw)
        seed = _extract_seed_from_run(run)

        history = run.history(keys=task_keys + ["_step"], samples=5000)
        if history.empty:
            continue

        for _, row in history.iterrows():
            step = row.get("_step")
            if pd.isna(step):
                continue
            for key, task_name in zip(task_keys, task_names):
                val = coerce_numeric_value(row.get(key))
                if val is not None:
                    records.append({
                        "algorithm": algo,
                        "seed": seed,
                        "step": int(step),
                        "task": task_name,
                        "value": float(val),
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_continual(df: pd.DataFrame, bin_size: int = 10_000) -> pd.DataFrame:
    """Bin steps, compute IQM/Q25/Q75 per algorithm per bin."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["binned_step"] = (df["step"] / bin_size).round() * bin_size

    agg: List[dict] = []
    for (algo, step), grp in df.groupby(["algorithm", "binned_step"]):
        vals = grp["value"].dropna().to_numpy(dtype=np.float64)
        if len(vals) == 0:
            continue
        iqm_val = compute_iqm(vals)
        q25, q75 = (np.percentile(vals, [25, 75]) if len(vals) > 1 else (iqm_val, iqm_val))
        agg.append({
            "algorithm": algo,
            "step": step / 1e6,
            "iqm": iqm_val,
            "q25": q25,
            "q75": q75,
            "n_runs": len(vals),
        })

    result = pd.DataFrame(agg)
    if not result.empty:
        result = result.sort_values(["algorithm", "step"])
    return result


def aggregate_per_task(df: pd.DataFrame, bin_size: int = 10_000) -> pd.DataFrame:
    """Same as aggregate_continual but grouped by (algorithm, task, binned_step)."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["binned_step"] = (df["step"] / bin_size).round() * bin_size

    agg: List[dict] = []
    for (algo, task, step), grp in df.groupby(["algorithm", "task", "binned_step"]):
        vals = grp["value"].dropna().to_numpy(dtype=np.float64)
        if len(vals) == 0:
            continue
        iqm_val = compute_iqm(vals)
        q25, q75 = (np.percentile(vals, [25, 75]) if len(vals) > 1 else (iqm_val, iqm_val))
        agg.append({
            "algorithm": algo,
            "task": task,
            "step": step / 1e6,
            "iqm": iqm_val,
            "q25": q25,
            "q75": q75,
            "n_runs": len(vals),
        })

    result = pd.DataFrame(agg)
    if not result.empty:
        result = result.sort_values(["algorithm", "task", "step"])
    return result


def compute_class_incremental_from_avg(
    avg_df: pd.DataFrame,
    steps_per_task: int = 500_000,
    num_tasks: int = 10,
) -> pd.DataFrame:
    """Compute true class-incremental success rate by rescaling eval/average_success_rate.

    eval/average_success_rate divides by tasks seen so far. To get true class-incremental
    (dividing by ALL tasks), we rescale:

        class_incr(s) = avg_success(s) × tasks_seen(s) / num_tasks

    where tasks_seen(s) = min(floor(s / steps_per_task) + 1, num_tasks).

    This gives the expected staircase: perfect task 1 → 1/num_tasks, etc.
    """
    if avg_df.empty:
        return pd.DataFrame()

    df = avg_df.copy()
    # Compute tasks seen at each step
    df["tasks_seen"] = np.minimum(
        np.floor(df["step"] / steps_per_task).astype(int) + 1,
        num_tasks,
    )
    # Rescale: avg_success * tasks_seen / num_tasks
    df["value"] = df["value"] * df["tasks_seen"] / num_tasks

    return aggregate_continual(df[["algorithm", "seed", "step", "value"]])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _make_color_encoding(
    df: pd.DataFrame,
    base_text_size: float,
) -> alt.Color:
    """Build a shared color encoding for algorithm lines."""
    legend_label_size = scaled_font_size(base_text_size, 1.1)
    legend_title_size = scaled_font_size(base_text_size, 1.5)
    legend_symbol_size = scaled_font_size(base_text_size, 50)
    legend_stroke = max(2, scaled_font_size(base_text_size, 0.5))
    legend_label_limit = max(220, int(round(base_text_size * 12)))

    legend_domain = build_algorithm_legend_domain(df["algorithm"].unique())

    return alt.Color(
        "algorithm:N",
        legend=alt.Legend(
            title=None,
            symbolOpacity=1.0,
            symbolType="stroke",
            symbolStrokeWidth=legend_stroke,
            symbolSize=legend_symbol_size,
            fillColor="rgba(255,255,255,1)",
            strokeColor="gray",
            padding=5,
            cornerRadius=3,
            orient="bottom-left",
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
            labelFontSize=legend_label_size,
            titleFontSize=legend_title_size,
            labelFontWeight="bold",
            labelPadding=8,
            labelLimit=legend_label_limit,
        ),
        scale=alt.Scale(domain=legend_domain),
    )


def _make_line_band_chart(
    agg_df: pd.DataFrame,
    steps_per_task: int,
    num_tasks: int,
    title: str,
    y_title: str,
    base_text_size: float = 25.0,
    chart_width: int = 900,
    chart_height: int = 400,
    line_width: float = 4.0,
    show_iqr: bool = True,
    y_domain: Optional[List[float]] = None,
) -> alt.Chart:
    """Line+band chart with task boundary rules. Shared by average and current-task charts."""
    axis_label = scaled_font_size(base_text_size, 1.5)
    axis_title = scaled_font_size(base_text_size, 1.5)
    chart_title_size = scaled_font_size(base_text_size, 1.6)

    axis_kw = {
        "labelFontSize": axis_label,
        "titleFontSize": axis_title,
        "labelFont": ALT_FONT_FAMILY,
        "titleFont": ALT_FONT_FAMILY,
    }

    max_step = float(agg_df["step"].max()) if not agg_df.empty else 5.0
    x_scale = alt.Scale(domain=[0, max_step], nice=False)

    color_enc = _make_color_encoding(agg_df, base_text_size)

    base = alt.Chart(agg_df).encode(
        x=alt.X("step:Q", title="Training Steps (Millions)",
                 scale=x_scale, axis=alt.Axis(**axis_kw)),
    )

    smoothed = base.transform_window(
        frame=[-5, 5], groupby=["algorithm"],
        smooth_iqm="mean(iqm)", smooth_q25="mean(q25)", smooth_q75="mean(q75)",
    )

    y_axis = alt.Axis(
        **axis_kw,
        tickCount=6,
    )

    y_scale = alt.Scale(domain=y_domain) if y_domain else alt.Scale()

    lines = smoothed.mark_line(strokeWidth=line_width).encode(
        color=color_enc,
        y=alt.Y("smooth_iqm:Q", title=y_title,
                 axis=y_axis, scale=y_scale),
        tooltip=[
            alt.Tooltip("algorithm:N", title="Method"),
            alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
            alt.Tooltip("smooth_iqm:Q", title="IQM", format=".3f"),
        ],
    )

    layers = lines
    if show_iqr:
        bands = smoothed.mark_area(opacity=0.2).encode(
            color=color_enc,
            y=alt.Y("smooth_q25:Q", title=y_title,
                     axis=y_axis, scale=y_scale),
            y2=alt.Y2("smooth_q75:Q"),
        )
        layers = bands + lines

    rules = _boundary_rules(steps_per_task, num_tasks)
    layers = layers + rules

    return (
        layers.properties(
            width=chart_width,
            height=chart_height,
            title=alt.TitleParams(text=title, font=ALT_FONT_FAMILY,
                                  fontSize=chart_title_size, fontWeight="bold"),
            padding={"right": 25},
        )
        .configure_axis(grid=True, gridOpacity=0.3,
                         labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)
        .configure_legend(labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY,
                          labelFontWeight="bold")
        .interactive()
    )


def create_average_success_chart(
    agg_df: pd.DataFrame,
    steps_per_task: int = 500_000,
    num_tasks: int = 10,
    title: str = "Average Success Rate",
    base_text_size: float = 25.0,
    chart_width: int = 900,
    chart_height: int = 400,
    line_width: float = 4.0,
    show_iqr: bool = True,
) -> alt.Chart:
    """Line+band chart of eval/average_success_rate with task boundary rules."""
    return _make_line_band_chart(
        agg_df, steps_per_task, num_tasks, title,
        y_title="Average Success Rate (Tasks Seen)",
        base_text_size=base_text_size,
        chart_width=chart_width, chart_height=chart_height,
        line_width=line_width, show_iqr=show_iqr,
        y_domain=[0, 1],
    )


def create_per_task_grid(
    agg_df: pd.DataFrame,
    steps_per_task: int = 500_000,
    num_tasks: int = 10,
    title: str = "Per-Task Success Rate",
    base_text_size: float = 14.0,
    panel_width: int = 350,
    panel_height: int = 200,
    line_width: float = 2.5,
    show_iqr: bool = True,
    columns: int = 5,
) -> alt.Chart:
    """2-row x 5-column grid of per-task success panels."""
    tasks = sorted(agg_df["task"].unique())
    if not tasks:
        raise ValueError("No tasks in data")

    max_step = float(agg_df["step"].max()) if not agg_df.empty else 5.0
    legend_domain = build_algorithm_legend_domain(agg_df["algorithm"].unique())
    color_scale = alt.Scale(domain=legend_domain)

    axis_label = scaled_font_size(base_text_size, 1.2)
    axis_title_sz = scaled_font_size(base_text_size, 1.3)
    panel_title_sz = scaled_font_size(base_text_size, 1.4)

    boundaries = _task_boundaries(steps_per_task, num_tasks)
    rules_df = pd.DataFrame({"boundary": boundaries})

    charts: List[alt.Chart] = []
    for task in tasks:
        task_df = agg_df[agg_df["task"] == task]
        if task_df.empty:
            continue

        base = alt.Chart(task_df).encode(
            x=alt.X("step:Q", title="Steps (M)",
                     scale=alt.Scale(domain=[0, max_step], nice=False),
                     axis=alt.Axis(labelFontSize=axis_label, titleFontSize=axis_title_sz,
                                   labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)),
        )

        smoothed = base.transform_window(
            frame=[-5, 5], groupby=["algorithm"],
            smooth_iqm="mean(iqm)", smooth_q25="mean(q25)", smooth_q75="mean(q75)",
        )

        y_kw = dict(
            title="Success Rate",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(labelFontSize=axis_label, titleFontSize=axis_title_sz,
                          labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY,
                          tickCount=5),
        )

        color_enc = alt.Color("algorithm:N", scale=color_scale, legend=None)

        panel_lines = smoothed.mark_line(strokeWidth=line_width).encode(
            color=color_enc,
            y=alt.Y("smooth_iqm:Q", **y_kw),
        )

        panel_layers = panel_lines
        if show_iqr:
            panel_bands = smoothed.mark_area(opacity=0.15).encode(
                color=color_enc,
                y=alt.Y("smooth_q25:Q", **y_kw),
                y2=alt.Y2("smooth_q75:Q"),
            )
            panel_layers = panel_bands + panel_lines

        panel_rules = alt.Chart(rules_df).mark_rule(
            strokeDash=[3, 3], color="gray", opacity=0.5,
        ).encode(x="boundary:Q")

        panel_layers = panel_layers + panel_rules

        display_name = task.replace("-", " ").replace("_", " ").title()
        panel = panel_layers.properties(
            width=panel_width, height=panel_height,
            title=alt.TitleParams(text=display_name, font=ALT_FONT_FAMILY,
                                  fontSize=panel_title_sz, fontWeight="bold"),
        )
        charts.append(panel)

    # Arrange into grid
    rows: List[alt.Chart] = []
    for i in range(0, len(charts), columns):
        row = charts[i : i + columns]
        rows.append(alt.hconcat(*row) if len(row) > 1 else row[0])

    grid = alt.vconcat(*rows) if len(rows) > 1 else rows[0]

    # Add a shared color legend by overlaying a dummy chart
    legend_chart = (
        alt.Chart(agg_df)
        .mark_point(opacity=0)
        .encode(
            color=alt.Color(
                "algorithm:N",
                scale=color_scale,
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    labelFont=ALT_FONT_FAMILY,
                    labelFontSize=scaled_font_size(base_text_size, 1.1),
                    labelFontWeight="bold",
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    symbolSize=scaled_font_size(base_text_size, 40),
                    direction="horizontal",
                ),
            )
        )
        .properties(width=panel_width * columns, height=0)
    )

    return alt.vconcat(grid, legend_chart).resolve_scale(color="shared")


def create_current_task_chart(
    agg_df: pd.DataFrame,
    steps_per_task: int = 500_000,
    num_tasks: int = 10,
    title: str = "Current Task Success Rate",
    base_text_size: float = 25.0,
    chart_width: int = 900,
    chart_height: int = 400,
    line_width: float = 4.0,
    show_iqr: bool = True,
) -> alt.Chart:
    """charts/success_rate over time — staircase pattern."""
    return _make_line_band_chart(
        agg_df, steps_per_task, num_tasks, title,
        y_title="Current Task Success Rate",
        base_text_size=base_text_size,
        chart_width=chart_width, chart_height=chart_height,
        line_width=line_width, show_iqr=show_iqr,
        y_domain=[0, 1],
    )


def create_summary_chart(
    agg_df: pd.DataFrame,
    title: str = "Average Success Rate (AUC)",
    base_text_size: float = 22.0,
    chart_width: int = 600,
    chart_height: int = 400,
) -> alt.Chart:
    """Bar chart of time-averaged eval/average_success_rate per algorithm with IQR."""
    # Compute area under curve (time-averaged) per algorithm
    final_rows: List[dict] = []
    for algo, grp in agg_df.groupby("algorithm"):
        grp = grp.sort_values("step")
        if len(grp) < 2:
            continue
        steps = grp["step"].values
        duration = steps[-1] - steps[0]
        if duration <= 0:
            continue
        # Time-averaged IQM, Q25, Q75
        auc_iqm = float(np.trapezoid(grp["iqm"].values, steps) / duration)
        auc_q25 = float(np.trapezoid(grp["q25"].values, steps) / duration)
        auc_q75 = float(np.trapezoid(grp["q75"].values, steps) / duration)
        final_rows.append({
            "algorithm": algo,
            "iqm": auc_iqm,
            "q25": auc_q25,
            "q75": auc_q75,
        })

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        raise ValueError("No final data")

    legend_domain = build_algorithm_legend_domain(final_df["algorithm"])
    axis_label = scaled_font_size(base_text_size, 1.2)
    axis_title_sz = scaled_font_size(base_text_size, 1.3)
    chart_title_sz = scaled_font_size(base_text_size, 1.4)

    base = alt.Chart(final_df)

    bars = base.mark_bar(size=45).encode(
        x=alt.X("algorithm:N", sort=legend_domain, title=None,
                 axis=alt.Axis(labelAngle=-30, labelFontSize=axis_label,
                               labelFont=ALT_FONT_FAMILY, labelLimit=200)),
        y=alt.Y("iqm:Q", title="Time-Averaged Success Rate",
                 scale=alt.Scale(domain=[0, min(1.0, final_df["q75"].max() * 1.15)]),
                 axis=alt.Axis(titleFontSize=axis_title_sz, labelFontSize=axis_label,
                               titleFont=ALT_FONT_FAMILY, labelFont=ALT_FONT_FAMILY,
                               format=".2f", tickCount=6)),
        color=alt.Color("algorithm:N", sort=legend_domain, legend=None),
        tooltip=[
            alt.Tooltip("algorithm:N", title="Method"),
            alt.Tooltip("iqm:Q", title="IQM", format=".3f"),
            alt.Tooltip("q25:Q", title="Q25", format=".3f"),
            alt.Tooltip("q75:Q", title="Q75", format=".3f"),
        ],
    )

    rules = base.mark_rule(color="black", strokeWidth=1.5).encode(
        x=alt.X("algorithm:N", sort=legend_domain),
        y=alt.Y("q25:Q"),
        y2=alt.Y2("q75:Q"),
    )

    tick_kw = dict(thickness=2, size=12, color="black")
    lower_caps = base.mark_tick(**tick_kw).encode(
        x=alt.X("algorithm:N", sort=legend_domain), y="q25:Q")
    upper_caps = base.mark_tick(**tick_kw).encode(
        x=alt.X("algorithm:N", sort=legend_domain), y="q75:Q")

    return (
        (bars + rules + lower_caps + upper_caps)
        .properties(
            width=chart_width, height=chart_height,
            title=alt.TitleParams(text=title, font=ALT_FONT_FAMILY,
                                  fontSize=chart_title_sz, fontWeight="bold"),
        )
        .configure_axis(grid=True, gridOpacity=0.3,
                         labelFont=ALT_FONT_FAMILY, titleFont=ALT_FONT_FAMILY)
    )


# ---------------------------------------------------------------------------
# Sweep top-K (reuses ablation_plot infrastructure)
# ---------------------------------------------------------------------------
def run_sweep_plot(
    entity: str,
    project: str,
    group: str,
    metric: str,
    top_k: int,
    output_dir: Path,
    ext: str,
    base_text_size: float,
    chart_width: int,
    chart_height: int,
    line_width: float,
    ranking_criteria: str = "final",
    plot_title: Optional[str] = None,
) -> None:
    """Generate sweep top-K plot using ablation_plot infrastructure."""
    df = fetch_ablation_data(entity, project, group, metric,
                             show_metric_in_legend=True, include_failed=True)
    if df.empty:
        print(f"No data for sweep group: {group}")
        return

    agg_df = aggregate_data(df, "config")
    if agg_df.empty:
        print("No aggregated sweep data")
        return

    # Rank and filter top-K
    ranking_map = {
        "final": agg_df.groupby("group")["iqm"].last(),
        "peak": agg_df.groupby("group")["iqm"].max(),
        "average": agg_df.groupby("group")["iqm"].mean(),
        "auc": pd.Series({
            g: np.trapezoid(gdf["iqm"].values, gdf["step"].values)
            for g, gdf in agg_df.groupby("group") if len(gdf) > 1
        }),
    }
    ranking = ranking_map[ranking_criteria]
    top_groups = ranking.sort_values(ascending=False).head(top_k).index
    agg_df = agg_df[agg_df["group"].isin(top_groups)]

    if agg_df.empty:
        return

    title = plot_title or f"{group.replace('_', ' ').title()} — Top {top_k} by {ranking_criteria}"
    summarize_group_performance(agg_df, metric)

    chart = create_ablation_chart(
        agg_df, metric, title,
        use_short_labels=True,
        show_iqr=True,
        base_text_size=base_text_size,
        chart_width=chart_width,
        chart_height=chart_height,
        line_width=line_width,
    )

    slug = slugify_title(title)
    save_path = output_dir / f"{slug}.{ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(save_path))
    print(f"Saved sweep chart: {save_path}")


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------
def main(
    wandb_entity: str,
    wandb_project: str = "metaworld_sac",
    group: Optional[str] = None,
    plot_type: Literal["all", "average", "per_task", "current", "summary", "forgetting", "sweep", "class_incremental"] = "all",
    metric_average: str = "eval/average_success_rate",
    metric_current: str = "charts/success_rate",
    steps_per_task: int = 500_000,
    num_tasks: int = 10,
    top_k: int = 5,
    ranking_criteria: Literal["final", "peak", "average", "auc"] = "final",
    output_dir: str = "./plots/continual",
    ext: str = "png",
    show_iqr: bool = True,
    base_text_size: float = 22.0,
    chart_width: int = 900,
    chart_height: int = 400,
    line_width: float = 4.0,
    include_failed: bool = False,
    plot_title: Optional[str] = None,
    run_prefix: Optional[str] = None,
    filter_steps_per_task: Optional[int] = None,
):
    """Generate continual MetaWorld visualizations.

    Args:
        wandb_entity: W&B entity/username
        wandb_project: W&B project name
        group: W&B group filter (required for sweep mode)
        plot_type: Which plots to generate
        metric_average: Metric key for average success rate
        metric_current: Metric key for current-task success rate
        steps_per_task: Steps per task (for boundary lines and class-incremental computation)
        num_tasks: Number of tasks
        top_k: Number of top configs for sweep mode
        ranking_criteria: How to rank configs in sweep mode
        output_dir: Output directory for PNGs
        ext: File extension
        show_iqr: Show IQR bands
        base_text_size: Base text size
        chart_width: Chart width in pixels
        chart_height: Chart height in pixels
        line_width: Line width
        include_failed: Include failed runs
        plot_title: Override plot title
        run_prefix: Filter runs by name prefix (e.g. 'sac_' or 'sac2m_')
        filter_steps_per_task: Only include runs whose config steps_per_task matches this value
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = slugify_title(group or wandb_project)

    # Build a steps suffix for filenames (e.g. "_500k", "_2m") when filtering by cohort
    if steps_per_task >= 1_000_000:
        steps_suffix = f"_{steps_per_task // 1_000_000}m"
    elif steps_per_task >= 1_000:
        steps_suffix = f"_{steps_per_task // 1_000}k"
    else:
        steps_suffix = f"_{steps_per_task}"
    # Only embed suffix in filename when run_prefix is set (cohort-specific plots)
    file_suffix = steps_suffix if run_prefix else ""

    # ------------------------------------------------------------------
    # Sweep mode
    # ------------------------------------------------------------------
    if plot_type == "sweep":
        if not group:
            print("Error: --group is required for sweep mode")
            return

        # Plot both average and current-task metrics
        for metric, suffix in [(metric_average, "avg_success"), (metric_current, "current_success")]:
            run_sweep_plot(
                wandb_entity, wandb_project, group, metric,
                top_k=top_k, output_dir=out, ext=ext,
                base_text_size=base_text_size,
                chart_width=chart_width, chart_height=chart_height,
                line_width=line_width, ranking_criteria=ranking_criteria,
                plot_title=plot_title,
            )
        return

    # ------------------------------------------------------------------
    # Continual-learning plots
    # ------------------------------------------------------------------
    do_avg = plot_type in ("all", "average", "class_incremental")
    do_per_task = plot_type in ("all", "per_task", "class_incremental")
    do_current = plot_type in ("all", "current")
    do_summary = plot_type in ("all", "summary", "class_incremental")
    do_forgetting = plot_type in ("all", "forgetting")
    do_class_incr = plot_type in ("all", "class_incremental")

    # Fetch average success data (used by average, summary, class_incremental)
    avg_df = pd.DataFrame()
    avg_agg = pd.DataFrame()
    if do_avg or do_summary:
        print(f"\nFetching {metric_average} ...")
        avg_df = fetch_continual_data(wandb_entity, wandb_project, metric_average,
                                      group=group, include_failed=include_failed,
                                      run_prefix=run_prefix,
                                      filter_steps_per_task=filter_steps_per_task)
        if not avg_df.empty:
            avg_agg = aggregate_continual(avg_df)

    # Fetch current-task data
    cur_agg = pd.DataFrame()
    if do_current:
        print(f"\nFetching {metric_current} ...")
        cur_df = fetch_continual_data(wandb_entity, wandb_project, metric_current,
                                      group=group, include_failed=include_failed,
                                      run_prefix=run_prefix,
                                      filter_steps_per_task=filter_steps_per_task)
        if not cur_df.empty:
            cur_agg = aggregate_continual(cur_df)

    # Fetch forgetting data
    fgt_agg = pd.DataFrame()
    if do_forgetting:
        metric_forgetting = "eval/average_forgetting"
        print(f"\nFetching {metric_forgetting} ...")
        fgt_df = fetch_continual_data(wandb_entity, wandb_project, metric_forgetting,
                                      group=group, include_failed=include_failed,
                                      run_prefix=run_prefix,
                                      filter_steps_per_task=filter_steps_per_task)
        if not fgt_df.empty:
            fgt_agg = aggregate_continual(fgt_df)

    # Fetch per-task data
    pt_df = pd.DataFrame()
    pt_agg = pd.DataFrame()
    if do_per_task and not do_class_incr:
        # Skip per-task fetch for class_incremental-only (uses avg_df instead)
        print("\nFetching per-task data ...")
        pt_df = fetch_per_task_data(wandb_entity, wandb_project,
                                    group=group, include_failed=include_failed,
                                    run_prefix=run_prefix,
                                    filter_steps_per_task=filter_steps_per_task)
        if not pt_df.empty:
            pt_agg = aggregate_per_task(pt_df)

    # Compute class-incremental by rescaling eval/average_success_rate
    ci_agg = pd.DataFrame()
    if do_class_incr and not avg_df.empty:
        print("\nComputing class-incremental success rate ...")
        ci_agg = compute_class_incremental_from_avg(avg_df, steps_per_task=steps_per_task, num_tasks=num_tasks)

    # ---- Create plots ----

    if do_class_incr and not ci_agg.empty:
        print("\nCreating class-incremental success chart ...")
        title = plot_title or "Class-Incremental Success Rate (Continual MetaWorld)"
        chart = _make_line_band_chart(
            ci_agg, steps_per_task, num_tasks, title,
            y_title="Class-Incremental Success Rate",
            base_text_size=base_text_size,
            chart_width=chart_width, chart_height=chart_height,
            line_width=line_width, show_iqr=show_iqr,
            y_domain=[0, 1],
        )
        path = out / f"{slug}_class_incremental{file_suffix}.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")

    if do_avg and not avg_agg.empty:
        print("\nCreating average success chart ...")
        title = plot_title or "Average Success Rate (Continual MetaWorld)"
        chart = create_average_success_chart(
            avg_agg, steps_per_task, num_tasks, title,
            base_text_size=base_text_size,
            chart_width=chart_width, chart_height=chart_height,
            line_width=line_width, show_iqr=show_iqr,
        )
        path = out / f"{slug}_average_success.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")

        # Text summary
        # Reuse ablation_plot summarizer by renaming column
        summary_df = avg_agg.rename(columns={"algorithm": "group"})
        summarize_group_performance(summary_df, metric_average)

    if do_current and not cur_agg.empty:
        print("\nCreating current-task chart ...")
        title = plot_title or "Current Task Success Rate (Continual MetaWorld)"
        chart = create_current_task_chart(
            cur_agg, steps_per_task, num_tasks, title,
            base_text_size=base_text_size,
            chart_width=chart_width, chart_height=chart_height,
            line_width=line_width, show_iqr=show_iqr,
        )
        path = out / f"{slug}_current_task.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")

    if do_per_task and not pt_agg.empty:
        print("\nCreating per-task grid ...")
        title = plot_title or "Per-Task Success Rate (Continual MetaWorld)"
        chart = create_per_task_grid(
            pt_agg, steps_per_task, num_tasks, title,
            base_text_size=14.0,
            panel_width=350, panel_height=200,
            line_width=2.5, show_iqr=show_iqr,
        )
        path = out / f"{slug}_per_task_grid.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")

    if do_summary and not avg_agg.empty:
        print("\nCreating summary bar chart ...")
        title = plot_title or "Time-Averaged Success Rate (Continual MetaWorld)"
        chart = create_summary_chart(
            avg_agg, title,
            base_text_size=base_text_size,
            chart_width=600, chart_height=chart_height,
        )
        path = out / f"{slug}_summary.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")

    if do_forgetting and not fgt_agg.empty:
        print("\nCreating forgetting chart ...")
        title = plot_title or "Average Forgetting (Continual MetaWorld)"
        chart = _make_line_band_chart(
            fgt_agg, steps_per_task, num_tasks, title,
            y_title="Average Forgetting",
            base_text_size=base_text_size,
            chart_width=chart_width, chart_height=chart_height,
            line_width=line_width, show_iqr=show_iqr,
        )
        path = out / f"{slug}_forgetting.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")

    print(f"\nDone! Charts saved to {out}")


if __name__ == "__main__":
    tyro.cli(main)
