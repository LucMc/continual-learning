#!/usr/bin/env python3
"""Generate plots for MT1 (MetaWorld single-task) results.

Creates 10 plots (one per task) showing mean episodic return averaged over seeds for each method.

Usage:
    python -m continual_learning.plots.plot_mt1 --wandb-entity lucmc --wandb-project "MT1 results"

    # To list available groups/tasks without plotting:
    python -m continual_learning.plots.plot_mt1 --wandb-entity lucmc --wandb-project "MT1 results" --list-groups

    # To plot specific tasks:
    python -m continual_learning.plots.plot_mt1 --wandb-entity lucmc --wandb-project "MT1 results" --tasks door-open,drawer-close
"""
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
import re
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
try:
    from ablation_plot import (
        ALT_FONT_FAMILY,
        scaled_font_size,
        coerce_numeric_value,
        _enable_times_new_roman_theme,
    )
except ImportError:
    from .ablation_plot import (
        ALT_FONT_FAMILY,
        scaled_font_size,
        coerce_numeric_value,
        _enable_times_new_roman_theme,
    )

_enable_times_new_roman_theme()

# Standard MT1 tasks (MetaWorld v2)
MT1_TASKS = [
    "door-open",
    "door-close",
    "drawer-open",
    "drawer-close",
    "button-press-topdown",
    "peg-insert-side",
    "window-open",
    "window-close",
    "reach",
    "push",
]

# Algorithm display names for consistent legend ordering
ALGORITHM_DISPLAY_NAMES = {
    "adam": "Adam",
    "standard": "Adam",
    "cbp": "CBP",
    "ccbp": "CCBP",
    "ccbp2": "CPR2",
    "cpr": "CPR",
    "redo": "ReDo",
    "regrama": "ReGraMa",
    "shrink_and_perturb": "Shrink & Perturb",
    "sp": "Shrink & Perturb",
    "bro": "BRO",
    "sac": "SAC",
}

ALGORITHM_LEGEND_ORDER = [
    "CPR2",
    "CCBP",
    "CBP",
    "ReGraMa",
    "ReDo",
    "Shrink & Perturb",
    "BRO",
    "SAC",
    "Adam",
]


def normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm names by removing seed suffixes and common variations."""
    if not name:
        return "unknown"

    normalized = name.strip().lower()

    # Remove seed suffix patterns like _s0, _seed=0, etc.
    normalized = re.sub(r'_s\d+$', '', normalized)
    normalized = re.sub(r'[,_]seed=[^,_]+', '', normalized)
    normalized = re.sub(r'_\d+$', '', normalized)  # Remove trailing numbers

    # Clean up trailing separators
    normalized = re.sub(r'[,_-]+$', '', normalized)

    return normalized


def display_algorithm_name(name: str) -> str:
    """Convert algorithm name to display format."""
    normalized = normalize_algorithm_name(name)
    return ALGORITHM_DISPLAY_NAMES.get(normalized, name.replace('_', ' ').title())


def algorithm_sort_key(algo: str) -> Tuple[int, str]:
    """Sort key for consistent algorithm ordering in legends."""
    try:
        idx = ALGORITHM_LEGEND_ORDER.index(algo)
    except ValueError:
        idx = len(ALGORITHM_LEGEND_ORDER)
    return (idx, algo)


def compute_iqm(values: np.ndarray) -> float:
    """Compute interquartile mean."""
    if len(values) < 4:
        return float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return float(np.mean(values[mask])) if np.any(mask) else float(np.mean(values))


def extract_task_from_run(run) -> Optional[str]:
    """Extract task name from run config or name."""
    config = getattr(run, "config", {}) or {}

    # Try common config keys for task name
    for key in ["task", "task_name", "env", "env_name", "environment"]:
        if key in config:
            return str(config[key])

    # Try to extract from run name or group
    run_name = getattr(run, "name", "") or ""
    group = getattr(run, "group", "") or ""

    # Check if any MT1 task name appears in run name or group
    for task in MT1_TASKS:
        task_pattern = task.replace("-", "[-_]?")
        if re.search(task_pattern, run_name, re.IGNORECASE) or \
           re.search(task_pattern, group, re.IGNORECASE):
            return task

    return None


def extract_algorithm_from_run(run) -> str:
    """Extract algorithm/optimizer name from run config or name.

    For MT1 runs, the format is typically: sac_{task}_{optimizer}_{seed}
    We want to extract the optimizer (adam, ccbp, cbp, redo, etc.)
    """
    config = getattr(run, "config", {}) or {}

    # First try the 'optimizer' config key (specific to MT1 runs)
    if "optimizer" in config:
        return str(config["optimizer"])

    # Try common config keys for algorithm/method
    for key in ["algorithm", "algo", "method", "agent"]:
        if key in config:
            return str(config[key])

    # Try to extract from run name
    # MT1 format: sac_{task}_{optimizer}_{seed}
    run_name = getattr(run, "name", "") or ""

    # Pattern: sac_taskname-v3_optimizer_seed
    match = re.match(r'^sac_[^_]+-v\d+_([^_]+)_\d+$', run_name)
    if match:
        return match.group(1)

    # Fallback: try splitting and getting second-to-last part
    parts = run_name.split("_")
    if len(parts) >= 3:
        # Check if last part is a number (seed)
        if parts[-1].isdigit():
            return parts[-2]

    if parts:
        return parts[0]

    return run_name


def extract_seed_from_run(run) -> str:
    """Extract seed from run config or name."""
    config = getattr(run, "config", {}) or {}

    if "seed" in config:
        return str(config["seed"])

    run_name = getattr(run, "name", "") or ""

    # Try to find seed in name
    match = re.search(r'seed[=_]?(\d+)', run_name, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try trailing number pattern
    match = re.search(r'_s?(\d+)$', run_name)
    if match:
        return match.group(1)

    # Use run ID as fallback
    return getattr(run, "id", "0")


def list_available_groups(entity: str, project: str) -> List[str]:
    """List all available groups in the wandb project."""
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", per_page=500))

    groups = set()
    tasks = set()
    algorithms = set()

    for run in runs:
        if run.group:
            groups.add(run.group)

        task = extract_task_from_run(run)
        if task:
            tasks.add(task)

        algo = extract_algorithm_from_run(run)
        if algo:
            algorithms.add(normalize_algorithm_name(algo))

    print(f"\nFound {len(runs)} total runs in {entity}/{project}")
    print(f"\nGroups ({len(groups)}):")
    for g in sorted(groups):
        print(f"  - {g}")

    print(f"\nDetected tasks ({len(tasks)}):")
    for t in sorted(tasks):
        print(f"  - {t}")

    print(f"\nDetected algorithms ({len(algorithms)}):")
    for a in sorted(algorithms):
        print(f"  - {a} -> {display_algorithm_name(a)}")

    return sorted(groups)


def fetch_mt1_data(
    entity: str,
    project: str,
    metric: str = "charts/mean_episodic_return",
    tasks: Optional[List[str]] = None,
    group_filter: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch MT1 data from wandb, organized by task.

    Returns:
        Dictionary mapping task names to DataFrames with columns:
        [algorithm, seed, step, value]
    """
    api = wandb.Api()

    filters = {}
    if group_filter:
        filters["group"] = group_filter

    runs = list(api.runs(f"{entity}/{project}", filters=filters if filters else None, per_page=500))
    finished_runs = [r for r in runs if r.state == "finished"]

    print(f"Fetched {len(runs)} runs, {len(finished_runs)} finished")

    # Organize data by task
    task_data: Dict[str, List[Dict]] = defaultdict(list)

    for run in finished_runs:
        task = extract_task_from_run(run)
        if task is None:
            # Try using group as task
            task = run.group if run.group else "unknown"

        # Filter tasks if specified
        if tasks and task not in tasks:
            continue

        algo = extract_algorithm_from_run(run)
        seed = extract_seed_from_run(run)
        display_algo = display_algorithm_name(algo)

        # Fetch history
        history = run.history(keys=[metric, "_step"], samples=5000)
        if history.empty:
            print(f"  Warning: No data for {run.name}")
            continue

        data_points = 0
        for _, row in history.iterrows():
            step = row.get("_step")
            value = coerce_numeric_value(row.get(metric))

            if pd.notna(step) and value is not None:
                task_data[task].append({
                    "algorithm": display_algo,
                    "seed": seed,
                    "step": int(step),
                    "value": float(value),
                    "run_name": run.name,
                })
                data_points += 1

        if data_points > 0:
            print(f"  {run.name}: {data_points} points (task={task}, algo={display_algo}, seed={seed})")

    # Convert to DataFrames
    result = {}
    for task, records in task_data.items():
        if records:
            result[task] = pd.DataFrame(records)

    return result


def aggregate_by_algorithm(df: pd.DataFrame, bin_size: int = 10000) -> pd.DataFrame:
    """Aggregate data by algorithm, computing IQM over seeds at each step."""
    if df.empty:
        return pd.DataFrame()

    # Bin steps
    df = df.copy()
    df["binned_step"] = (df["step"] / bin_size).round() * bin_size

    aggregated = []
    for (algo, step), group in df.groupby(["algorithm", "binned_step"]):
        raw_values = group["value"].dropna().values
        values = np.asarray(raw_values, dtype=np.float64)
        if len(values) == 0:
            continue

        seeds = group["seed"].nunique()
        iqm_val = compute_iqm(values)

        if len(values) > 1:
            q25, q75 = np.percentile(values, [25, 75])
        else:
            q25, q75 = iqm_val, iqm_val

        aggregated.append({
            "algorithm": algo,
            "step": step / 1_000_000,  # Convert to millions
            "iqm": iqm_val,
            "q25": q25,
            "q75": q75,
            "n_seeds": seeds,
            "n_values": len(values),
        })

    result = pd.DataFrame(aggregated)
    if not result.empty:
        result = result.sort_values(["algorithm", "step"])

    return result


def create_task_chart(
    df: pd.DataFrame,
    task_name: str,
    show_iqr: bool = True,
    base_text_size: float = 20.0,
    chart_width: int = 800,
    chart_height: int = 400,
    line_width: float = 4.0,
    x_axis_max: Optional[float] = None,
    y_axis_label: str = "Mean Episodic Return",
) -> alt.Chart:
    """Create a chart for a single MT1 task."""

    if df.empty:
        raise ValueError(f"No data for task: {task_name}")

    axis_label_size = scaled_font_size(base_text_size, 1.3)
    axis_title_size = scaled_font_size(base_text_size, 1.4)
    legend_label_size = scaled_font_size(base_text_size, 1.0)
    legend_symbol_size = scaled_font_size(base_text_size, 40)
    legend_stroke_width = max(2, scaled_font_size(base_text_size, 0.4))
    chart_title_size = scaled_font_size(base_text_size, 1.5)
    legend_label_limit = max(200, int(base_text_size * 10))

    # Determine x-axis range
    if x_axis_max is None:
        x_axis_max = float(df["step"].max()) * 1.02

    # Sort algorithms for consistent ordering
    algorithms = sorted(df["algorithm"].unique(), key=algorithm_sort_key)

    # Build color scale
    color_scale = alt.Scale(domain=algorithms)

    base = alt.Chart(df).encode(
        x=alt.X(
            "step:Q",
            title="Training Steps (Millions)",
            scale=alt.Scale(domain=[0, x_axis_max], nice=False),
            axis=alt.Axis(
                labelFontSize=axis_label_size,
                titleFontSize=axis_title_size,
                labelFont=ALT_FONT_FAMILY,
                titleFont=ALT_FONT_FAMILY,
            ),
        ),
        color=alt.Color(
            "algorithm:N",
            title="Method",
            scale=color_scale,
            legend=alt.Legend(
                title=None,
                symbolOpacity=1.0,
                orient="bottom-left",
                fillColor="rgba(255,255,255,0.9)",
                strokeColor="gray",
                padding=5,
                cornerRadius=3,
                labelFontSize=legend_label_size,
                labelFontWeight="bold",
                symbolSize=legend_symbol_size,
                symbolStrokeWidth=legend_stroke_width,
                labelLimit=legend_label_limit,
                labelPadding=6,
                labelFont=ALT_FONT_FAMILY,
            ),
        ),
    )

    # Apply smoothing
    smoothed = base.transform_window(
        frame=[-5, 5],
        groupby=["algorithm"],
        smooth_iqm="mean(iqm)",
        smooth_q25="mean(q25)",
        smooth_q75="mean(q75)",
    )

    y_axis = alt.Axis(
        labelFontSize=axis_label_size,
        titleFontSize=axis_title_size,
        labelFont=ALT_FONT_FAMILY,
        titleFont=ALT_FONT_FAMILY,
        tickCount=6,
    )

    # Create lines
    lines = smoothed.mark_line(strokeWidth=line_width).encode(
        y=alt.Y(
            "smooth_iqm:Q",
            title="" if show_iqr else y_axis_label,
            axis=y_axis,
        ),
        tooltip=[
            alt.Tooltip("algorithm:N", title="Method"),
            alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
            alt.Tooltip("smooth_iqm:Q", title="IQM Return", format=".1f"),
            alt.Tooltip("n_seeds:Q", title="Seeds"),
        ],
    )

    # Create bands (IQR) and combine with lines
    if show_iqr:
        bands = smoothed.mark_area(opacity=0.2).encode(
            y=alt.Y(
                "smooth_q25:Q",
                title=y_axis_label,
                axis=y_axis,
            ),
            y2=alt.Y2("smooth_q75:Q"),
        )
        chart = bands + lines
    else:
        chart = lines

    # Format task name for title
    title = task_name.replace("-", " ").replace("_", " ").title()

    return (
        chart.properties(
            width=chart_width,
            height=chart_height,
            title=alt.TitleParams(
                text=f"MT1: {title}",
                fontSize=chart_title_size,
                fontWeight="bold",
                font=ALT_FONT_FAMILY,
            ),
            padding={"right": 20},
        )
        .configure_axis(
            grid=True,
            gridOpacity=0.3,
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .configure_legend(
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .interactive()
    )


def create_combined_chart(
    task_dfs: Dict[str, pd.DataFrame],
    columns: int = 2,
    show_iqr: bool = True,
    base_text_size: float = 16.0,
    chart_width: int = 400,
    chart_height: int = 300,
    line_width: float = 3.0,
    x_axis_max: Optional[float] = None,
    y_axis_label: str = "Mean Episodic Return",
) -> alt.Chart:
    """Create a combined chart with all tasks in a grid layout."""

    charts = []

    # Use provided x_axis_max or find global max for consistency
    if x_axis_max is None:
        all_steps = []
        for df in task_dfs.values():
            if not df.empty:
                all_steps.extend(df["step"].tolist())
        x_axis_max = max(all_steps) * 1.02 if all_steps else 10.0

    for task_name, df in sorted(task_dfs.items()):
        if df.empty:
            continue

        try:
            chart = create_task_chart(
                df,
                task_name,
                show_iqr=show_iqr,
                base_text_size=base_text_size,
                chart_width=chart_width,
                chart_height=chart_height,
                line_width=line_width,
                x_axis_max=x_axis_max,
                y_axis_label=y_axis_label,
            )
            charts.append(chart)
        except Exception as e:
            print(f"Warning: Could not create chart for {task_name}: {e}")

    if not charts:
        raise ValueError("No charts could be created")

    # Arrange in grid
    rows = []
    for i in range(0, len(charts), columns):
        row_charts = charts[i:i + columns]
        if len(row_charts) == 1:
            rows.append(row_charts[0])
        else:
            rows.append(alt.hconcat(*row_charts))

    if len(rows) == 1:
        return rows[0]

    return alt.vconcat(*rows).resolve_scale(color="shared")


def main(
    wandb_entity: str,
    wandb_project: str = "MT1 results",
    metric: str = "charts/mean_episodic_return",
    tasks: Optional[List[str]] = None,
    group_filter: Optional[str] = None,
    output_dir: str = "./plots/mt1",
    ext: str = "png",
    show_iqr: bool = True,
    base_text_size: float = 20.0,
    chart_width: int = 800,
    chart_height: int = 400,
    line_width: float = 4.0,
    x_axis_max: float = 1.0,
    combined: bool = False,
    combined_columns: int = 2,
    list_groups: bool = False,
):
    """Generate MT1 plots from wandb data.

    Args:
        wandb_entity: Wandb entity/username
        wandb_project: Wandb project name
        metric: Metric to plot
        tasks: Specific tasks to plot (None for all)
        group_filter: Filter runs by group name
        output_dir: Output directory for plots
        ext: File extension (png, svg, pdf)
        show_iqr: Show interquartile range bands
        base_text_size: Base font size for charts
        chart_width: Chart width in pixels
        chart_height: Chart height in pixels
        line_width: Line width for plots
        x_axis_max: Maximum x-axis value in millions of steps (default: 1.0 = 1M steps)
        combined: Create a single combined plot with all tasks
        combined_columns: Number of columns in combined plot
        list_groups: Just list available groups without plotting
    """

    if list_groups:
        list_available_groups(wandb_entity, wandb_project)
        return

    # Derive y-axis label from metric
    metric_labels = {
        "charts/mean_episode_return": "Mean Episodic Return",
        "charts/mean_episodic_return": "Mean Episodic Return",
        "charts/success_rate": "Success Rate",
        "eval/mean_return": "Eval Mean Return",
        "eval/success_rate": "Eval Success Rate",
    }
    y_axis_label = metric_labels.get(metric, metric.split("/")[-1].replace("_", " ").title())

    # Parse tasks if provided as comma-separated string
    if tasks and len(tasks) == 1 and "," in tasks[0]:
        tasks = [t.strip() for t in tasks[0].split(",")]

    print(f"\nFetching data from {wandb_entity}/{wandb_project}")
    if tasks:
        print(f"Filtering to tasks: {tasks}")
    if group_filter:
        print(f"Filtering to group: {group_filter}")

    # Fetch data
    raw_data = fetch_mt1_data(
        wandb_entity,
        wandb_project,
        metric=metric,
        tasks=tasks,
        group_filter=group_filter,
    )

    if not raw_data:
        print("No data found!")
        return

    print(f"\nFound data for {len(raw_data)} tasks: {list(raw_data.keys())}")

    # Aggregate data
    aggregated_data = {}
    for task, df in raw_data.items():
        agg_df = aggregate_by_algorithm(df)
        if not agg_df.empty:
            aggregated_data[task] = agg_df

            # Print summary
            print(f"\n{task}:")
            for algo in sorted(agg_df["algorithm"].unique(), key=algorithm_sort_key):
                algo_df = agg_df[agg_df["algorithm"] == algo]
                max_seeds = algo_df["n_seeds"].max()
                final_iqm = algo_df.iloc[-1]["iqm"] if len(algo_df) > 0 else 0
                peak_iqm = algo_df["iqm"].max()
                print(f"  {algo}: {max_seeds} seeds, final={final_iqm:.1f}, peak={peak_iqm:.1f}")

    if not aggregated_data:
        print("No aggregated data!")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if combined:
        # Create single combined chart
        print("\nCreating combined chart...")
        chart = create_combined_chart(
            aggregated_data,
            columns=combined_columns,
            show_iqr=show_iqr,
            base_text_size=base_text_size * 0.8,
            chart_width=chart_width // 2,
            chart_height=chart_height // 2,
            line_width=line_width * 0.75,
            x_axis_max=x_axis_max,
            y_axis_label=y_axis_label,
        )

        save_path = output_path / f"mt1_combined.{ext}"
        chart.save(str(save_path))
        print(f"Saved: {save_path}")
    else:
        # Create individual charts for each task
        for task, df in sorted(aggregated_data.items()):
            print(f"\nCreating chart for {task}...")

            try:
                chart = create_task_chart(
                    df,
                    task,
                    show_iqr=show_iqr,
                    base_text_size=base_text_size,
                    chart_width=chart_width,
                    chart_height=chart_height,
                    line_width=line_width,
                    x_axis_max=x_axis_max,
                    y_axis_label=y_axis_label,
                )

                # Sanitize task name for filename
                safe_task = task.replace(" ", "_").replace("/", "_")
                save_path = output_path / f"mt1_{safe_task}.{ext}"
                chart.save(str(save_path))
                print(f"Saved: {save_path}")

            except Exception as e:
                print(f"Error creating chart for {task}: {e}")

    print(f"\nDone! Charts saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
