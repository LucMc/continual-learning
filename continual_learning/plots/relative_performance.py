#!/usr/bin/env python3
"""Plot % performance relative to single-task adam ceiling for MinAtar continual learning.

For each of the 3 MinAtar tasks (space_invaders, asterix, seaquest), computes the
single-task adam ceiling from LOW runs, then plots continual run eval performance
as a percentage of that ceiling over training steps.

Usage:
    python continual_learning/plots/relative_performance.py \
        --wandb-entity lucmc --wandb-project cont-minatar

    # Include older non-LOW continual runs for testing:
    python continual_learning/plots/relative_performance.py \
        --wandb-entity lucmc --wandb-project cont-minatar --include-legacy
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import tyro
import wandb

TASKS = ["space_invaders", "asterix", "seaquest"]
STEPS_PER_TASK = 1_500_000
TASK_BOUNDARIES = [STEPS_PER_TASK * i for i in range(1, len(TASKS))]


def compute_iqm(values: np.ndarray) -> float:
    if len(values) < 4:
        return float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return float(np.mean(values[mask])) if np.any(mask) else float(np.mean(values))


# ---------------------------------------------------------------------------
# 1. Single-task adam ceiling
# ---------------------------------------------------------------------------


def compute_ceilings(
    entity: str,
    project: str,
    tail_frac: float = 0.2,
) -> dict[str, float]:
    """Compute single-task adam ceiling for each MinAtar task.

    Uses the last ``tail_frac`` of ``charts/mean_episode_return`` from finished
    adam LOW runs, averaged across seeds.
    """
    api = wandb.Api()
    ceilings: dict[str, float] = {}

    for task in TASKS:
        group = f"minatar_single_{task}"
        runs = list(
            api.runs(f"{entity}/{project}", filters={"group": group}, per_page=100)
        )
        # Keep finished adam LOW runs
        adam_low = [
            r
            for r in runs
            if r.state == "finished" and "adam" in r.name and "LOW" in r.name
        ]
        if not adam_low:
            print(f"  WARNING: no finished adam LOW runs for {task}, skipping")
            continue

        seed_means: list[float] = []
        for r in adam_low:
            h = r.history(keys=["charts/mean_episode_return"], samples=5000)
            vals = h["charts/mean_episode_return"].dropna().values
            if len(vals) == 0:
                continue
            tail_n = max(1, int(len(vals) * tail_frac))
            seed_means.append(float(np.mean(vals[-tail_n:])))

        if seed_means:
            ceiling = float(np.mean(seed_means))
            ceilings[task] = ceiling
            print(
                f"  {task}: ceiling = {ceiling:.2f}  "
                f"(from {len(seed_means)} run(s), tail {tail_frac:.0%})"
            )
        else:
            print(f"  WARNING: no valid data for {task}")

    return ceilings


# ---------------------------------------------------------------------------
# 2. Fetch continual runs
# ---------------------------------------------------------------------------


def parse_run_name(name: str) -> tuple[str, str]:
    """Extract (optimizer, seed) from a continual run name.

    Expected formats:
        discrete_sac_minatar_{opt}_{net}_LOW_{seed}   (current)
        discrete_sac_minatar_{opt}_{net}_{seed}        (legacy)
    """
    parts = name.split("_")
    seed = parts[-1]
    # Walk backwards: skip seed, skip optional LOW, skip network (cnn/mlp)
    idx = len(parts) - 2
    if parts[idx] == "LOW":
        idx -= 1
    # parts[idx] should be the network type
    if parts[idx] in ("cnn", "mlp"):
        idx -= 1
    # Everything from index 3 (after "discrete_sac_minatar") to idx+1 is the optimizer
    opt = "_".join(parts[3 : idx + 1])
    return opt, seed


def fetch_continual_data(
    entity: str,
    project: str,
    continual_group: str,
    include_legacy: bool,
) -> pd.DataFrame:
    """Fetch per-task eval returns from continual runs.

    Returns a DataFrame with columns:
        optimizer, seed, step, task, eval_return
    """
    api = wandb.Api()
    runs = list(
        api.runs(
            f"{entity}/{project}", filters={"group": continual_group}, per_page=300
        )
    )
    finished = [r for r in runs if r.state == "finished"]
    if not include_legacy:
        finished = [r for r in finished if "LOW" in r.name]

    print(
        f"\nContinual runs: {len(finished)} finished"
        f" (of {len(runs)} total in group '{continual_group}')"
    )

    eval_keys = [f"eval/{t}/mean_return" for t in TASKS]
    step_key = "charts/total_steps"

    rows: list[dict] = []
    for r in finished:
        opt, seed = parse_run_name(r.name)
        print(f"  {r.name}  ->  optimizer={opt}, seed={seed}")

        h = r.history(keys=eval_keys + [step_key], samples=10_000)
        if h.empty:
            continue

        for _, row in h.iterrows():
            step = row.get(step_key)
            if pd.isna(step):
                continue
            for task in TASKS:
                val = row.get(f"eval/{task}/mean_return")
                if pd.notna(val):
                    rows.append(
                        {
                            "optimizer": opt,
                            "seed": seed,
                            "step": int(step),
                            "task": task,
                            "eval_return": float(val),
                        }
                    )

    return pd.DataFrame(rows)


def fetch_current_task_data(
    entity: str,
    project: str,
    continual_group: str,
    include_legacy: bool,
) -> pd.DataFrame:
    """Fetch charts/mean_episode_return time-series tagged with the active task.

    The active task is inferred from charts/total_steps and STEPS_PER_TASK.
    Returns a DataFrame with columns:
        optimizer, seed, step, task, current_return
    """
    api = wandb.Api()
    runs = list(
        api.runs(
            f"{entity}/{project}", filters={"group": continual_group}, per_page=300
        )
    )
    finished = [r for r in runs if r.state == "finished"]
    if not include_legacy:
        finished = [r for r in finished if "LOW" in r.name]

    rows: list[dict] = []
    for r in finished:
        opt, seed = parse_run_name(r.name)
        h = r.history(
            keys=["charts/mean_episode_return", "charts/total_steps"], samples=10_000
        )
        if h.empty:
            continue

        steps = h["charts/total_steps"].values
        rew = h["charts/mean_episode_return"].values
        mask = (~np.isnan(steps)) & (~np.isnan(rew))
        steps, rew = steps[mask], rew[mask]

        for s, v in zip(steps, rew):
            task_idx = min(int(s // STEPS_PER_TASK), len(TASKS) - 1)
            rows.append(
                {
                    "optimizer": opt,
                    "seed": seed,
                    "step": int(s),
                    "task": TASKS[task_idx],
                    "current_return": float(v),
                }
            )

    return pd.DataFrame(rows)


def compute_current_task_relative_df(
    current_df: pd.DataFrame,
    ceilings: dict[str, float],
    bin_size: int = 10_000,
) -> pd.DataFrame:
    """Convert current-task returns to % of that task's single-task adam ceiling.

    Returns aggregated DataFrame with columns:
        optimizer, task, step, iqm, q25, q75, n_seeds
    """
    if current_df.empty:
        return pd.DataFrame()

    df = current_df.copy()
    df["relative_pct"] = df.apply(
        lambda r: (r["current_return"] / ceilings[r["task"]]) * 100
        if r["task"] in ceilings and ceilings[r["task"]] > 0
        else np.nan,
        axis=1,
    )
    df = df.dropna(subset=["relative_pct"])
    df["binned_step"] = (df["step"] / bin_size).round().astype(int) * bin_size

    # Aggregate per (optimizer, seed, binned_step) — one task active per step,
    # so this also gives the "current task" series.
    per_seed = (
        df.groupby(["optimizer", "seed", "binned_step", "task"])["relative_pct"]
        .mean()
        .reset_index()
    )

    agg_rows: list[dict] = []
    for (opt, bstep), grp in per_seed.groupby(["optimizer", "binned_step"]):
        vals = grp["relative_pct"].values
        if len(vals) == 0:
            continue
        iqm = compute_iqm(vals)
        if len(vals) > 1:
            q25, q75 = np.percentile(vals, [25, 75])
        else:
            q25, q75 = iqm, iqm
        agg_rows.append(
            {
                "optimizer": opt,
                "task": "current",
                "step": bstep / 1_000_000,
                "iqm": iqm,
                "q25": q25,
                "q75": q75,
                "n_seeds": grp["seed"].nunique(),
            }
        )

    return pd.DataFrame(agg_rows).sort_values(["optimizer", "step"])


def aggregate_per_task_raw(
    current_df: pd.DataFrame,
    bin_size: int = 10_000,
) -> pd.DataFrame:
    """Aggregate raw current-task returns across seeds, per (optimizer, task).

    Uses the actual step (not relative within-task) so panels can be plotted
    side-by-side with each task occupying its real window on the x-axis.

    Returns DataFrame with columns:
        optimizer, task, step, iqm, q25, q75, n_seeds
    """
    if current_df.empty:
        return pd.DataFrame()

    df = current_df.copy()
    df["binned_step"] = (df["step"] / bin_size).round().astype(int) * bin_size

    per_seed = (
        df.groupby(["optimizer", "seed", "task", "binned_step"])["current_return"]
        .mean()
        .reset_index()
    )

    agg_rows: list[dict] = []
    for (opt, task, bstep), grp in per_seed.groupby(
        ["optimizer", "task", "binned_step"]
    ):
        vals = grp["current_return"].values
        if len(vals) == 0:
            continue
        iqm = compute_iqm(vals)
        if len(vals) > 1:
            q25, q75 = np.percentile(vals, [25, 75])
        else:
            q25, q75 = iqm, iqm
        agg_rows.append(
            {
                "optimizer": opt,
                "task": task,
                "step": bstep / 1_000_000,
                "iqm": iqm,
                "q25": q25,
                "q75": q75,
                "n_seeds": grp["seed"].nunique(),
            }
        )

    return pd.DataFrame(agg_rows).sort_values(["task", "optimizer", "step"])


def create_per_task_chart(
    df: pd.DataFrame,
    ceilings: dict[str, float],
    title: str = "",
) -> alt.Chart:
    """Three horizontal panels (one per task), each with its own y-axis scaled
    to that task's data range. Each panel covers the task's training window
    on the x-axis and shows the single-task adam ceiling as a dashed line.
    """
    task_labels = {
        "space_invaders": "Space Invaders",
        "asterix": "Asterix",
        "seaquest": "Seaquest",
    }

    panels: list[alt.Chart] = []
    for i, task in enumerate(TASKS):
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue

        x_lo = i * STEPS_PER_TASK / 1_000_000
        x_hi = (i + 1) * STEPS_PER_TASK / 1_000_000

        base = alt.Chart(task_df).encode(
            x=alt.X(
                "step:Q",
                title="Training Steps (Millions)",
                scale=alt.Scale(domain=[x_lo, x_hi], nice=False),
                axis=alt.Axis(labelFontSize=11, titleFontSize=13),
            ),
            color=alt.Color(
                "optimizer:N",
                title="Optimizer",
                legend=alt.Legend(
                    title=None,
                    symbolOpacity=1.0,
                    orient="bottom",
                    labelFontSize=11,
                    symbolSize=140,
                )
                if i == 0
                else None,
            ),
        )

        smoothed = base.transform_window(
            frame=[-10, 10],
            groupby=["optimizer"],
            smooth_iqm="mean(iqm)",
            smooth_q25="mean(q25)",
            smooth_q75="mean(q75)",
        )

        bands = smoothed.mark_area(opacity=0.2).encode(
            y=alt.Y(
                "smooth_q25:Q",
                title="Mean Episode Return" if i == 0 else None,
                axis=alt.Axis(labelFontSize=11, titleFontSize=13),
            ),
            y2=alt.Y2("smooth_q75:Q"),
        )

        lines = smoothed.mark_line(strokeWidth=2).encode(
            y=alt.Y("smooth_iqm:Q"),
            tooltip=[
                alt.Tooltip("optimizer:N", title="Optimizer"),
                alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
                alt.Tooltip("smooth_iqm:Q", title="Return", format=".3f"),
                alt.Tooltip("n_seeds:Q", title="Seeds"),
            ],
        )

        layers = [bands, lines]

        ceil = ceilings.get(task)
        if ceil is not None:
            ceil_line = (
                alt.Chart(pd.DataFrame({"y": [ceil]}))
                .mark_rule(
                    strokeDash=[6, 4], strokeWidth=1.5, color="black", opacity=0.6
                )
                .encode(y="y:Q")
            )
            ceil_label = (
                alt.Chart(
                    pd.DataFrame({"x": [x_lo + (x_hi - x_lo) * 0.05], "y": [ceil]})
                )
                .mark_text(
                    text=f"single-task adam: {ceil:.2f}",
                    align="left",
                    baseline="bottom",
                    dy=-3,
                    fontSize=10,
                    color="black",
                    opacity=0.7,
                )
                .encode(x="x:Q", y="y:Q")
            )
            layers.extend([ceil_line, ceil_label])

        panel = alt.layer(*layers).properties(
            width=340,
            height=320,
            title=alt.TitleParams(
                text=task_labels.get(task, task),
                fontSize=13,
                fontWeight="bold",
            ),
        )
        panels.append(panel)

    if not panels:
        raise ValueError("No data to plot")

    combined = alt.hconcat(*panels).resolve_scale(color="shared")
    return (
        combined.properties(
            title=alt.TitleParams(
                text=title or "Per-Task Mean Episode Return (independent y-axes)",
                fontSize=15,
                fontWeight="bold",
            )
        )
        .configure_axis(grid=True, gridOpacity=0.3)
    )


def create_current_chart(
    df: pd.DataFrame, title: str = "", y_max: float | None = None
) -> alt.Chart:
    """Single-panel chart of current-task relative performance over time.

    y_max: optional upper bound on the y-axis (with clamping). The seaquest
    single-task adam ceiling is unusually low, so methods can exceed 1000%;
    a clamp keeps the meaningful 0–100% range readable.
    """
    max_step = df["step"].max() if len(df) > 0 else 4.5

    base = alt.Chart(df).encode(
        x=alt.X(
            "step:Q",
            title="Training Steps (Millions)",
            scale=alt.Scale(domain=[0, max_step * 1.02], nice=True),
            axis=alt.Axis(labelFontSize=12, titleFontSize=14),
        ),
        color=alt.Color(
            "optimizer:N",
            title="Optimizer",
            legend=alt.Legend(
                title=None,
                symbolOpacity=1.0,
                orient="bottom-left",
                fillColor="rgba(255,255,255,0.9)",
                strokeColor="gray",
                padding=5,
                cornerRadius=3,
                labelFontSize=12,
                symbolSize=150,
            ),
        ),
    )

    smoothed = base.transform_window(
        frame=[-20, 20],
        groupby=["optimizer"],
        smooth_iqm="mean(iqm)",
        smooth_q25="mean(q25)",
        smooth_q75="mean(q75)",
    )

    y_scale = (
        alt.Scale(domain=[0, y_max], clamp=True)
        if y_max is not None
        else alt.Scale(zero=True)
    )

    bands = smoothed.mark_area(opacity=0.2).encode(
        y=alt.Y(
            "smooth_q25:Q",
            title="% of Single-Task Adam (current task)",
            axis=alt.Axis(labelFontSize=12, titleFontSize=13),
            scale=y_scale,
        ),
        y2=alt.Y2("smooth_q75:Q"),
    )

    lines = smoothed.mark_line(strokeWidth=2).encode(
        y=alt.Y("smooth_iqm:Q"),
        tooltip=[
            alt.Tooltip("optimizer:N", title="Optimizer"),
            alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
            alt.Tooltip("smooth_iqm:Q", title="Rel. Perf. %", format=".1f"),
            alt.Tooltip("n_seeds:Q", title="Seeds"),
        ],
    )

    ref_line = (
        alt.Chart(pd.DataFrame({"y": [100]}))
        .mark_rule(strokeDash=[6, 4], strokeWidth=1.5, color="black", opacity=0.5)
        .encode(y="y:Q")
    )

    boundary_data = pd.DataFrame(
        {"x": [b / 1_000_000 for b in TASK_BOUNDARIES]}
    )
    boundary_lines = (
        alt.Chart(boundary_data)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1, color="gray", opacity=0.6)
        .encode(x="x:Q")
    )

    task_label_data = pd.DataFrame(
        [
            {
                "x": (i * STEPS_PER_TASK + STEPS_PER_TASK / 2) / 1_000_000,
                "task": t.replace("_", " ").title(),
            }
            for i, t in enumerate(TASKS)
        ]
    )
    task_labels = (
        alt.Chart(task_label_data)
        .mark_text(dy=-8, fontSize=12, fontWeight="bold", color="gray")
        .encode(x="x:Q", text="task:N")
    )

    chart = (bands + lines + ref_line + boundary_lines + task_labels).properties(
        width=900,
        height=380,
        title=alt.TitleParams(
            text=title or "Current-Task Performance vs Single-Task Adam Ceiling",
            fontSize=16,
            fontWeight="bold",
        ),
    )
    return (
        chart.configure_axis(grid=True, gridOpacity=0.3).interactive()
    )


# ---------------------------------------------------------------------------
# 3. Compute relative performance & aggregate
# ---------------------------------------------------------------------------


def compute_relative_df(
    continual_df: pd.DataFrame,
    ceilings: dict[str, float],
    bin_size: int = 10_000,
) -> pd.DataFrame:
    """Convert raw eval returns to % of single-task adam ceiling.

    Returns aggregated DataFrame with columns:
        optimizer, task, step, iqm, q25, q75, n_seeds
    """
    if continual_df.empty:
        return pd.DataFrame()

    # Add relative %
    df = continual_df.copy()
    df["relative_pct"] = df.apply(
        lambda r: (r["eval_return"] / ceilings[r["task"]]) * 100
        if r["task"] in ceilings and ceilings[r["task"]] > 0
        else np.nan,
        axis=1,
    )
    df = df.dropna(subset=["relative_pct"])

    # Bin steps
    df["binned_step"] = (df["step"] / bin_size).round().astype(int) * bin_size

    # Compute average across tasks per (optimizer, seed, binned_step)
    avg_rows: list[dict] = []
    for (opt, seed, bstep), grp in df.groupby(
        ["optimizer", "seed", "binned_step"]
    ):
        avg_rows.append(
            {
                "optimizer": opt,
                "seed": seed,
                "binned_step": bstep,
                "task": "average",
                "relative_pct": grp["relative_pct"].mean(),
            }
        )
    df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)

    # Aggregate across seeds: IQM, Q25, Q75
    agg_rows: list[dict] = []
    for (opt, task, bstep), grp in df.groupby(["optimizer", "task", "binned_step"]):
        vals = grp.groupby("seed")["relative_pct"].mean().values
        if len(vals) == 0:
            continue
        iqm = compute_iqm(vals)
        if len(vals) > 1:
            q25, q75 = np.percentile(vals, [25, 75])
        else:
            q25, q75 = iqm, iqm
        agg_rows.append(
            {
                "optimizer": opt,
                "task": task,
                "step": bstep / 1_000_000,
                "iqm": iqm,
                "q25": q25,
                "q75": q75,
                "n_seeds": len(vals),
            }
        )

    return pd.DataFrame(agg_rows).sort_values(["optimizer", "task", "step"])


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------


def create_chart(
    df: pd.DataFrame, title: str = "", y_max: float | None = None
) -> alt.Chart:
    """Create Altair chart with 4 vertically-stacked subplots (3 tasks + average)."""
    task_order = TASKS + ["average"]
    task_labels = {
        "space_invaders": "Space Invaders",
        "asterix": "Asterix",
        "seaquest": "Seaquest",
        "average": "Average (all tasks)",
    }

    max_step = df["step"].max() if len(df) > 0 else 4.5

    charts: list[alt.Chart] = []
    for task in task_order:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue

        base = alt.Chart(task_df).encode(
            x=alt.X(
                "step:Q",
                title="Training Steps (Millions)",
                scale=alt.Scale(domain=[0, max_step * 1.02], nice=True),
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            color=alt.Color(
                "optimizer:N",
                title="Optimizer",
                legend=alt.Legend(
                    title=None,
                    symbolOpacity=1.0,
                    orient="bottom-left",
                    fillColor="rgba(255,255,255,0.9)",
                    strokeColor="gray",
                    padding=5,
                    cornerRadius=3,
                    labelFontSize=12,
                    symbolSize=150,
                ),
            ),
        )

        # Smoothing
        smoothed = base.transform_window(
            frame=[-5, 5],
            groupby=["optimizer"],
            smooth_iqm="mean(iqm)",
            smooth_q25="mean(q25)",
            smooth_q75="mean(q75)",
        )

        y_scale = (
            alt.Scale(domain=[0, y_max], clamp=True)
            if y_max is not None
            else alt.Undefined
        )

        bands = smoothed.mark_area(opacity=0.2).encode(
            y=alt.Y(
                "smooth_q25:Q",
                title="% of Single-Task Adam",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13),
                scale=y_scale,
            ),
            y2=alt.Y2("smooth_q75:Q"),
        )

        lines = smoothed.mark_line(strokeWidth=2).encode(
            y=alt.Y("smooth_iqm:Q"),
            tooltip=[
                alt.Tooltip("optimizer:N", title="Optimizer"),
                alt.Tooltip("step:Q", title="Step (M)", format=".2f"),
                alt.Tooltip("smooth_iqm:Q", title="Rel. Perf. %", format=".1f"),
                alt.Tooltip("n_seeds:Q", title="Seeds"),
            ],
        )

        # 100% reference line
        ref_line = (
            alt.Chart(pd.DataFrame({"y": [100]}))
            .mark_rule(strokeDash=[6, 4], strokeWidth=1.5, color="black", opacity=0.5)
            .encode(y="y:Q")
        )

        # Task boundary vertical lines
        boundary_data = pd.DataFrame(
            {"x": [b / 1_000_000 for b in TASK_BOUNDARIES]}
        )
        boundary_lines = (
            alt.Chart(boundary_data)
            .mark_rule(
                strokeDash=[4, 3], strokeWidth=1, color="gray", opacity=0.6
            )
            .encode(x="x:Q")
        )

        chart = (bands + lines + ref_line + boundary_lines).properties(
            width=800,
            height=220,
            title=alt.TitleParams(
                text=task_labels.get(task, task), fontSize=14, fontWeight="bold"
            ),
        )
        charts.append(chart)

    if not charts:
        raise ValueError("No data to plot")

    combined = alt.vconcat(*charts).resolve_scale(color="shared")
    return (
        combined.properties(
            title=alt.TitleParams(
                text=title or "Relative Performance vs Single-Task Adam Ceiling",
                fontSize=16,
                fontWeight="bold",
            )
        )
        .configure_axis(grid=True, gridOpacity=0.3)
        .interactive()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    wandb_entity: str,
    wandb_project: str = "cont-minatar",
    continual_group: str = "minatar_discrete_sac",
    include_legacy: bool = False,
    tail_frac: float = 0.2,
    output_dir: str = "./plots",
    ext: str = "png",
    mode: str = "both",
    current_y_max: float | None = 200.0,
    eval_y_max: float | None = 200.0,
):
    """Plot continual MinAtar performance as % of single-task adam ceiling.

    mode:
        eval      — per-task eval/{task}/mean_return (sparse, only logged in last task)
        current   — charts/mean_episode_return divided by the active task's ceiling
        per_task  — three horizontal panels with raw returns, independent y-axes,
                    single-task adam ceiling shown as a dashed reference line
        both      — eval + current
        all       — eval + current + per_task
    """

    print("Computing single-task adam ceilings...")
    ceilings = compute_ceilings(wandb_entity, wandb_project, tail_frac=tail_frac)
    if not ceilings:
        print("ERROR: no ceilings computed, exiting")
        return

    print(f"\nCeilings: {ceilings}")

    out_dir = Path(output_dir) / ext
    out_dir.mkdir(exist_ok=True, parents=True)

    if mode in ("eval", "both", "all"):
        continual_df = fetch_continual_data(
            wandb_entity, wandb_project, continual_group, include_legacy
        )
        if continual_df.empty:
            print("WARNING: no eval data found")
        else:
            print(f"\nContinual eval data: {len(continual_df)} eval points")
            print(f"Optimizers: {continual_df['optimizer'].unique().tolist()}")
            print(f"Seeds: {continual_df['seed'].unique().tolist()}")

            rel_df = compute_relative_df(continual_df, ceilings)
            if not rel_df.empty:
                print("\nEval relative performance summary:")
                for task in TASKS + ["average"]:
                    task_data = rel_df[rel_df["task"] == task]
                    if task_data.empty:
                        continue
                    for opt in task_data["optimizer"].unique():
                        opt_data = task_data[task_data["optimizer"] == opt]
                        peak = opt_data["iqm"].max()
                        final = opt_data.iloc[-1]["iqm"]
                        print(
                            f"  {task:20s} | {opt:20s} | peak={peak:6.1f}%  final={final:6.1f}%"
                        )

                chart = create_chart(rel_df, y_max=eval_y_max)
                save_path = out_dir / f"relative_performance.{ext}"
                chart.save(str(save_path))
                print(f"\nEval chart saved to: {save_path}")

    if mode in ("current", "per_task", "both", "all"):
        print("\nFetching current-task time-series...")
        cur_df = fetch_current_task_data(
            wandb_entity, wandb_project, continual_group, include_legacy
        )
        if cur_df.empty:
            print("WARNING: no charts/mean_episode_return data found")
            return

        print(f"Current-task data: {len(cur_df)} points")

        if mode in ("current", "both", "all"):
            cur_rel_df = compute_current_task_relative_df(cur_df, ceilings)
            if not cur_rel_df.empty:
                print("\nCurrent-task relative performance summary (avg / final):")
                for opt in sorted(cur_rel_df["optimizer"].unique()):
                    opt_data = cur_rel_df[cur_rel_df["optimizer"] == opt]
                    avg = opt_data["iqm"].mean()
                    final = opt_data.iloc[-1]["iqm"]
                    print(
                        f"  {opt:20s} | avg={avg:7.1f}%  final={final:7.1f}%"
                    )

                cur_chart = create_current_chart(cur_rel_df, y_max=current_y_max)
                save_path = out_dir / f"relative_performance_current.{ext}"
                cur_chart.save(str(save_path))
                print(f"\nCurrent-task chart saved to: {save_path}")

        if mode in ("per_task", "all"):
            per_task_df = aggregate_per_task_raw(cur_df)
            if not per_task_df.empty:
                print("\nPer-task raw return summary (max iqm seen):")
                for task in TASKS:
                    task_df = per_task_df[per_task_df["task"] == task]
                    if task_df.empty:
                        continue
                    print(f"  {task}:")
                    for opt in sorted(task_df["optimizer"].unique()):
                        opt_data = task_df[task_df["optimizer"] == opt]
                        peak = opt_data["iqm"].max()
                        final = opt_data.iloc[-1]["iqm"]
                        print(
                            f"    {opt:20s} peak={peak:8.2f}  final={final:8.2f}"
                        )

                pt_chart = create_per_task_chart(per_task_df, ceilings)
                save_path = out_dir / f"relative_performance_per_task.{ext}"
                pt_chart.save(str(save_path))
                print(f"\nPer-task chart saved to: {save_path}")


if __name__ == "__main__":
    tyro.cli(main)
