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


def create_chart(df: pd.DataFrame, title: str = "") -> alt.Chart:
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

        bands = smoothed.mark_area(opacity=0.2).encode(
            y=alt.Y(
                "smooth_q25:Q",
                title="% of Single-Task Adam",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13),
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
):
    """Plot continual MinAtar performance as % of single-task adam ceiling."""

    print("Computing single-task adam ceilings...")
    ceilings = compute_ceilings(wandb_entity, wandb_project, tail_frac=tail_frac)
    if not ceilings:
        print("ERROR: no ceilings computed, exiting")
        return

    print(f"\nCeilings: {ceilings}")

    continual_df = fetch_continual_data(
        wandb_entity, wandb_project, continual_group, include_legacy
    )
    if continual_df.empty:
        print("ERROR: no continual run data found, exiting")
        return

    print(f"\nContinual data: {len(continual_df)} eval points")
    print(f"Optimizers: {continual_df['optimizer'].unique().tolist()}")
    print(f"Seeds: {continual_df['seed'].unique().tolist()}")

    rel_df = compute_relative_df(continual_df, ceilings)
    if rel_df.empty:
        print("ERROR: no relative performance data, exiting")
        return

    # Summary stats
    print("\nRelative performance summary:")
    for task in TASKS + ["average"]:
        task_data = rel_df[rel_df["task"] == task]
        if task_data.empty:
            continue
        for opt in task_data["optimizer"].unique():
            opt_data = task_data[task_data["optimizer"] == opt]
            peak = opt_data["iqm"].max()
            final = opt_data.iloc[-1]["iqm"]
            print(f"  {task:20s} | {opt:20s} | peak={peak:6.1f}%  final={final:6.1f}%")

    chart = create_chart(rel_df)

    save_path = Path(output_dir) / ext / f"relative_performance.{ext}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"\nChart saved to: {save_path}")


if __name__ == "__main__":
    tyro.cli(main)
