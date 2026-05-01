#!/usr/bin/env python3
"""5x2 grid of success-rate learning curves for TD MT1, one subplot per task.

Each line is one delay condition (mode × max_obs × max_act), averaged across
seeds at each environment step. X-axis = charts/total_steps, Y-axis = success
rate. Fixed delays are solid lines, continual delays are dashed; color encodes
delay magnitude (viridis, 0 → 12).

Usage:
    python -m continual_learning.plots.td_delay_curves
    python -m continual_learning.plots.td_delay_curves --metric charts/success_rate
    python -m continual_learning.plots.td_delay_curves --out td_delay_curves.png
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib import colormaps
from matplotlib.lines import Line2D

ENTITY = "lucmc"
PROJECT = "TD MT1 results"

DEFAULT_METRIC = "charts/success_rate"
STEP_KEY = "charts/total_steps"

TASKS = [
    "button-press-topdown-v3",
    "door-open-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "peg-insert-side-v3",
    "pick-place-v3",
    "push-v3",
    "reach-v3",
    "window-close-v3",
    "window-open-v3",
]

COND_ORDER = [
    ("fixed", 0, 0),
    ("fixed", 2, 2),
    ("fixed", 4, 4),
    ("fixed", 8, 8),
    ("fixed", 12, 12),
    ("continual", 4, 4),
    ("continual", 8, 8),
    ("continual", 12, 12),
]

MAG_SCALE = [0, 2, 4, 8, 12]


def cond_label(c: tuple) -> str:
    mode, obs, act = c
    return f"{mode}  obs={obs:>2}  act={act:>2}"


def cond_style(c: tuple) -> tuple:
    mode, obs, _ = c
    idx = MAG_SCALE.index(obs) if obs in MAG_SCALE else 0
    color = colormaps["viridis"](idx / max(1, len(MAG_SCALE) - 1))
    ls = "-" if mode == "fixed" else "--"
    return color, ls


def fetch(metric: str, total_steps_filter: int | None = None,
          clip_step: int | None = None) -> tuple[dict, dict]:
    """Returns ({(task, cond): [(steps, vals), ...]}, {(task, cond): n_seeds}).

    clip_step: if set, drop history rows with steps > clip_step so longer runs
        line up with shorter ones on a common x-axis.
    """
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=500))
    print(f"Scanning {len(runs)} runs "
          f"({sum(1 for r in runs if r.state == 'finished')} finished)...")
    if total_steps_filter is not None:
        print(f"Filtering to total_steps == {total_steps_filter:,}")
    if clip_step is not None:
        print(f"Clipping each run's history at step <= {clip_step:,}")

    data: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for r in runs:
        cfg = dict(r.config) if r.config else {}
        if cfg.get("algorithm") != "sac" or cfg.get("optimizer") != "adam":
            continue
        if total_steps_filter is not None and cfg.get("total_steps") != total_steps_filter:
            continue
        cond = (
            cfg.get("delay_mode"),
            int(cfg.get("max_obs_delay", -1)),
            int(cfg.get("max_act_delay", -1)),
        )
        task = cfg.get("task_name", "unknown")
        try:
            hist = r.history(keys=[metric, STEP_KEY], samples=2000, pandas=True)
        except Exception:
            continue
        if hist.empty or metric not in hist.columns or STEP_KEY not in hist.columns:
            continue
        hist = hist.dropna(subset=[metric, STEP_KEY]).sort_values(STEP_KEY)
        if clip_step is not None:
            hist = hist[hist[STEP_KEY] <= clip_step]
        if hist.empty:
            continue
        steps = hist[STEP_KEY].values.astype(float)
        vals = hist[metric].values.astype(float)
        data[(task, cond)].append((steps, vals))

    counts = {k: len(v) for k, v in data.items()}
    return data, counts


def mean_curve(runs: list, step_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each run onto step_grid and return (mean, sem)."""
    interp = []
    for steps, vals in runs:
        y = np.interp(step_grid, steps, vals, left=np.nan, right=np.nan)
        interp.append(y)
    stacked = np.stack(interp)
    mean = np.nanmean(stacked, axis=0)
    n_eff = np.sum(~np.isnan(stacked), axis=0)
    with np.errstate(invalid="ignore"):
        sem = np.where(n_eff > 1,
                       np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(n_eff),
                       0.0)
    return mean, sem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--out", default="td_delay_curves.png")
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--no-sem", action="store_true",
                   help="Don't shade standard error of the mean.")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Only include runs whose config.total_steps matches this value "
                        "(use 2000000 to align budgets across conditions).")
    p.add_argument("--clip-step", type=int, default=None,
                   help="Truncate each run's history at this step value before plotting "
                        "(use 2000000 to fold 5M-step control runs into the same window "
                        "as 2M-step delayed runs).")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data, counts = fetch(args.metric, total_steps_filter=args.total_steps,
                         clip_step=args.clip_step)

    max_step = 0.0
    for runs in data.values():
        for steps, _ in runs:
            if steps.size:
                max_step = max(max_step, float(steps.max()))
    if max_step == 0:
        print("No data — aborting.")
        return
    step_grid = np.linspace(0, max_step, args.n_grid)

    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, task in zip(axes, TASKS):
        plotted = False
        for c in COND_ORDER:
            runs = data.get((task, c), [])
            if not runs:
                continue
            mean, sem = mean_curve(runs, step_grid)
            color, ls = cond_style(c)
            n = counts.get((task, c), 0)
            ax.plot(step_grid, mean, color=color, linestyle=ls, linewidth=1.4,
                    label=f"{cond_label(c)}  (n={n})")
            if not args.no_sem and n > 1:
                ax.fill_between(step_grid, mean - sem, mean + sem,
                                color=color, alpha=0.12, linewidth=0)
            plotted = True
        ax.set_title(task, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")

    # shared labels + legend
    for ax in axes[-2:]:  # bottom row
        ax.set_xlabel("environment steps")
    for ax in axes[::2]:  # left column
        ax.set_ylabel("success rate")

    # Deduplicated legend drawn once below the grid
    handles, labels = [], []
    for c in COND_ORDER:
        color, ls = cond_style(c)
        total_seeds = sum(counts.get((t, c), 0) for t in TASKS)
        if total_seeds == 0:
            continue
        handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=1.8))
        labels.append(f"{cond_label(c)}  (Σ runs={total_seeds})")

    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"TD MT1: {args.metric} by delay condition (mean over seeds per task)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
