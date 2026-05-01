#!/usr/bin/env python3
"""Quantify the effect of time-delay on MT1 SAC.

The `lucmc/TD MT1 results` project sweeps a sim2real delay wrapper across
delay magnitudes (max_obs_delay, max_act_delay) and modes (fixed | continual)
on 10 MetaWorld v3 tasks; all runs use SAC + Adam. The `lucmc/MT1 results2`
project is a separate reset-method ablation on undelayed MT1.

This script does two things:

1. Within the TD project, compute mean episodic return per delay condition
   (mode, obs, act) per task, then aggregate across tasks and report the Δ
   vs the no-delay control (`fixed, 0, 0`).

2. As a sanity check, compare TD's no-delay control to Adam runs in
   `MT1 results2` to verify the two no-delay populations agree.

Usage:
    python -m continual_learning.plots.td_vs_baseline
    python -m continual_learning.plots.td_vs_baseline --metric charts/success_rate
    python -m continual_learning.plots.td_vs_baseline --plot
    python -m continual_learning.plots.td_vs_baseline --per-task
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import wandb

ENTITY = "lucmc"
TD_PROJECT = "TD MT1 results"
BASE_PROJECT = "MT1 results2"

DEFAULT_METRIC = "charts/mean_episode_return"
FINAL_WINDOW_FRAC = 0.1  # average over the last 10% of logged points

# Order conditions for printing: control first, then fixed by magnitude, then continual.
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
CONTROL = ("fixed", 0, 0)


def cond_label(c: tuple) -> str:
    mode, obs, act = c
    return f"{mode:<9} obs={obs:>2} act={act:>2}"


def _iqm_half_iqr(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) < 2:
        return float(values[0]), 0.0
    q25, q75 = np.percentile(values, [25, 75])
    half = float((q75 - q25) / 2)
    if len(values) >= 4:
        iqm = float(np.mean([v for v in values if q25 <= v <= q75]))
    else:
        iqm = float(np.mean(values))
    return iqm, half


def _final_value(vals: np.ndarray, frac: float) -> Optional[float]:
    if len(vals) == 0:
        return None
    n_tail = max(1, int(len(vals) * frac))
    tail = vals[-n_tail:]
    tail = tail[~np.isnan(tail)]
    if tail.size == 0:
        return None
    return float(np.mean(tail))


def fetch_td(metric: str, total_steps_filter: Optional[int] = None,
             clip_step: Optional[int] = None,
             ) -> dict[tuple, dict[str, list[float]]]:
    """Returns {condition: {task: [final_per_seed, ...]}}.
    Condition is (delay_mode, max_obs_delay, max_act_delay).

    total_steps_filter: only include runs whose config.total_steps matches.
    clip_step: truncate each run's history to rows with _step <= clip_step before
        computing the final-window average. Lets a 5M-step control run be
        compared fairly against a 2M-step delayed run.
    """
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{TD_PROJECT}", per_page=500))
    kept_budget = 0
    print(f"  TD project: {len(runs)} runs "
          f"({sum(1 for r in runs if r.state == 'finished')} finished)")
    if total_steps_filter is not None:
        print(f"  Filtering to total_steps == {total_steps_filter:,}")
    if clip_step is not None:
        print(f"  Clipping each run's history at _step <= {clip_step:,}")

    out: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        cfg = dict(r.config) if r.config else {}
        if cfg.get("algorithm") != "sac" or cfg.get("optimizer") != "adam":
            continue
        if total_steps_filter is not None and cfg.get("total_steps") != total_steps_filter:
            continue
        kept_budget += 1
        cond = (cfg.get("delay_mode"), int(cfg.get("max_obs_delay", -1)),
                int(cfg.get("max_act_delay", -1)))
        task = cfg.get("task_name", "unknown")
        try:
            hist = r.history(keys=[metric, "_step"], samples=2000, pandas=True)
        except Exception:
            continue
        if hist.empty or metric not in hist.columns:
            continue
        hist = hist.dropna(subset=[metric])
        if clip_step is not None and "_step" in hist.columns:
            hist = hist[hist["_step"] <= clip_step]
        if hist.empty:
            continue
        final = _final_value(hist[metric].values, FINAL_WINDOW_FRAC)
        if final is not None:
            out[cond][task].append(final)
    if total_steps_filter is not None:
        print(f"  Kept {kept_budget} runs after budget filter.")
    return out


def fetch_baseline_adam(metric: str) -> dict[str, list[float]]:
    """Returns {task: [final_per_seed, ...]} for Adam runs in MT1 results2."""
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{BASE_PROJECT}", per_page=500))
    print(f"  Baseline project: {len(runs)} runs "
          f"({sum(1 for r in runs if r.state == 'finished')} finished)")

    out: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        cfg = dict(r.config) if r.config else {}
        # Identify Adam runs by name pattern (project lacks consistent config)
        # plot_mt1.extract_algorithm_from_run convention: token after task in `sac_{task}_{algo}_{seed}`.
        name = r.name or ""
        if "_adam_" not in name and not name.endswith("_adam") and "_standard_" not in name:
            continue
        # Pull task from config or group
        task = cfg.get("task_name") or cfg.get("task") or (r.group or "unknown")
        # Strip `-v3` suffix mismatch (TD uses v3, baseline may use v2/v3)
        task_norm = task.replace("-v2", "-v3")
        try:
            hist = r.history(keys=[metric, "_step"], samples=2000, pandas=True)
        except Exception:
            continue
        if hist.empty or metric not in hist.columns:
            continue
        hist = hist.dropna(subset=[metric])
        if hist.empty:
            continue
        final = _final_value(hist[metric].values, FINAL_WINDOW_FRAC)
        if final is not None:
            out[task_norm].append(final)
    return out


def aggregate_across_tasks(td_data: dict, conds: list) -> dict[tuple, dict]:
    """For each condition, pool seed-level finals across all tasks."""
    summary = {}
    for c in conds:
        per_task = td_data.get(c, {})
        flat = [v for vs in per_task.values() for v in vs]
        iqm, half = _iqm_half_iqr(flat)
        n_tasks = sum(1 for vs in per_task.values() if vs)
        n_runs = len(flat)
        summary[c] = {
            "iqm": iqm,
            "half": half,
            "n_runs": n_runs,
            "n_tasks": n_tasks,
            "per_task": per_task,
        }
    return summary


def print_within_td(metric: str, summary: dict) -> list[tuple]:
    print()
    print(f"=== Within-TD project: effect of delay on {metric} ===")
    print(f"   (mean over last {int(FINAL_WINDOW_FRAC*100)}% of each run, "
          f"IQM ± half-IQR pooled across seeds × 10 tasks)")
    print()
    ctrl = summary[CONTROL]
    if ctrl["iqm"] is None:
        print("No control runs found — cannot compute Δ.")
        return []
    print(f"{'Condition':<28} {'N runs':>7} {'Tasks':>6} "
          f"{'Return IQM±hIQR':>20} {'Δ vs no-delay':>18} {'Rel Δ':>8}")
    print("-" * 92)
    rows = []
    for c in COND_ORDER:
        s = summary.get(c, {})
        iqm, half, n_runs, n_tasks = s.get("iqm"), s.get("half"), s.get("n_runs", 0), s.get("n_tasks", 0)
        if iqm is None:
            print(f"{cond_label(c):<28} {n_runs:>7d} {n_tasks:>6d} {'-- (no data)':>20}")
            continue
        delta = iqm - ctrl["iqm"]
        rel = delta / abs(ctrl["iqm"]) * 100 if ctrl["iqm"] != 0 else float("nan")
        marker = "  <- control" if c == CONTROL else ""
        val_str = f"{iqm:>10.1f}±{half or 0:>5.1f}" if half is not None else f"{iqm:>10.1f}"
        d_str = f"{delta:>+10.1f}" if c != CONTROL else "         --"
        r_str = f"{rel:>+6.1f}%" if c != CONTROL else "    --"
        print(f"{cond_label(c):<28} {n_runs:>7d} {n_tasks:>6d} "
              f"{val_str:>20} {d_str:>18} {r_str:>8}{marker}")
        rows.append((c, iqm, half, n_runs, delta, rel))
    print()
    print("Reading the table:")
    print(" - Δ < 0 with growing magnitude as obs/act increase ⇒ delay genuinely degrades return.")
    print(" - Δ ≈ 0 across rows ⇒ SAC is robust to this delay range.")
    print(" - Watch sample size: most non-control cells are ~10 runs (1 seed × 10 tasks).")
    print("   That's enough to spot a strong effect, weak effects need more seeds to confirm.")
    return rows


def print_per_task(metric: str, summary: dict) -> None:
    print()
    print(f"=== Per-task breakdown ({metric}) — mean of seed finals per cell ===")
    tasks = sorted({t for s in summary.values() for t in s["per_task"]})
    header = f"{'Task':<26}" + "".join(f" {cond_label(c):>26}" for c in COND_ORDER)
    print(header)
    print("-" * len(header))
    for t in tasks:
        line = f"{t:<26}"
        for c in COND_ORDER:
            vals = summary.get(c, {}).get("per_task", {}).get(t, [])
            cell = f"{np.mean(vals):>10.1f} (n={len(vals)})" if vals else f"{'--':>20}    "
            line += f" {cell:>26}"
        print(line)


def print_baseline_check(td_summary: dict, base_data: dict) -> None:
    """Compare TD's no-delay (fixed,0,0) Adam to MT1 results2 Adam runs per task."""
    print()
    print("=== Sanity check: TD no-delay vs MT1 results2 (Adam) per task ===")
    print("   (verifies the TD experiment's control matches the broader baseline)")
    ctrl_per_task = td_summary[CONTROL]["per_task"]
    tasks = sorted(set(ctrl_per_task) | set(base_data))
    print(f"{'Task':<26} {'TD no-delay (n)':>22} {'MT1 results2 Adam (n)':>26} {'Δ':>10}")
    print("-" * 92)
    deltas = []
    for t in tasks:
        td_v = ctrl_per_task.get(t, [])
        base_v = base_data.get(t, [])
        if not td_v and not base_v:
            continue
        td_str = f"{np.mean(td_v):>10.1f} ({len(td_v)})" if td_v else "          -- (0)"
        base_str = f"{np.mean(base_v):>10.1f} ({len(base_v)})" if base_v else "          -- (0)"
        if td_v and base_v:
            d = float(np.mean(td_v) - np.mean(base_v))
            deltas.append(d)
            d_str = f"{d:>+10.1f}"
        else:
            d_str = "        --"
        print(f"{t:<26} {td_str:>22} {base_str:>26} {d_str:>10}")
    if deltas:
        print("-" * 92)
        print(f"Mean Δ (TD ctrl − Baseline Adam) across {len(deltas)} tasks: {np.mean(deltas):+.1f}")
        print("If this is small relative to the delay-induced Δ above, the comparison is consistent.")


def save_plot(rows: list, metric: str, path: str) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    labels = [cond_label(r[0]) for r in rows]
    iqm = [r[1] for r in rows]
    err = [r[2] or 0.0 for r in rows]
    deltas = [r[4] for r in rows]
    n = [r[3] for r in rows]

    colors = ["#3b6ea5" if d == 0 else ("#c2543d" if d < 0 else "#5da25d") for d in deltas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))
    ax1.bar(x, iqm, yerr=err, capsize=3, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel(metric)
    ax1.set_title(f"Return by delay condition\n(IQM ± half-IQR, last {int(FINAL_WINDOW_FRAC*100)}% of run)")
    ax1.grid(axis="y", alpha=0.3)
    for xi, ni in zip(x, n):
        ax1.text(xi, 0, f"n={ni}", ha="center", va="bottom", fontsize=8, color="gray")

    ax2.bar(x, deltas, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Δ vs no-delay control")
    ax2.set_title("Delay-induced change in return")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("TD MT1: effect of sim2real delay on SAC + Adam (10 MetaWorld tasks)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"Saved plot to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default=DEFAULT_METRIC,
                   help=f"W&B metric key (default: {DEFAULT_METRIC}).")
    p.add_argument("--per-task", action="store_true",
                   help="Print the per-task × condition table.")
    p.add_argument("--no-baseline-check", action="store_true",
                   help="Skip the cross-project sanity check vs MT1 results2.")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="td_vs_baseline.png")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Only include TD runs whose config.total_steps matches this value.")
    p.add_argument("--clip-step", type=int, default=None,
                   help="Truncate each run's history at this _step before averaging "
                        "(use 2_000_000 to compare longer control runs against 2M-step "
                        "delayed runs at the same evaluation point).")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print(f"Fetching {ENTITY}/{TD_PROJECT} ...")
    td_data = fetch_td(args.metric, total_steps_filter=args.total_steps,
                       clip_step=args.clip_step)
    summary = aggregate_across_tasks(td_data, COND_ORDER)
    rows = print_within_td(args.metric, summary)
    if args.per_task:
        print_per_task(args.metric, summary)
    if not args.no_baseline_check:
        print(f"Fetching {ENTITY}/{BASE_PROJECT} ...")
        base = fetch_baseline_adam(args.metric)
        print_baseline_check(summary, base)
    if args.plot:
        save_plot(rows, args.metric, args.plot_path)


if __name__ == "__main__":
    main()
