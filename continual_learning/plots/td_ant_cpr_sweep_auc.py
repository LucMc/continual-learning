#!/usr/bin/env python3
"""Rank the CPR reset-selectivity sweep by overall AUC (paired seeds).

Target metric: overall AUC of charts/mean_episodic_return = step-normalised area
under the return curve (a time-average return over the whole run).

Aggregates the `cpr_sel` sweep (see experiments/batch_runs/sweep_td_ant.py and
slurm_cpr_ramp_sweep.sh) over its shared seed set, ranks configs, and -- because
the seeds are paired -- also reports each config's per-seed delta against the
baseline grid point (threshold=1.0, sharpness=16 == the current ramp `cpr`),
which is far lower-variance than comparing unpaired means.

Run after the sweep finishes:
    python td_ant_cpr_sweep_auc.py
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb

DEFAULT_ENTITY = "lucmc"
DEFAULT_PROJECT = "TD Ant ramp sweep"
DEFAULT_GROUP = "td_ant_ramp25_obs13_act12_info-none_cpr_sel_sweep"
DEFAULT_METRIC = "charts/mean_episodic_return"

# Baseline grid point == current ramp `cpr` (threshold=1.0, sharpness=16).
BASELINE_THRESHOLD = 1.0
BASELINE_SHARPNESS = 16


def _cfg_from_run(run) -> dict | None:
    """Pull the swept HPs from W&B config, falling back to the run-name tag."""
    cfg = dict(run.config)
    out = {}
    for key, cast in [
        ("threshold", float),
        ("sharpness", float),
        ("transform_type", str),
        ("replacement_rate", float),
        ("update_frequency", int),
        ("seed", int),
    ]:
        if key in cfg and cfg[key] is not None:
            out[key] = cast(cfg[key])

    # Fallback: parse from the run name tag (present even if config wasn't logged).
    for key, pat, cast in [
        ("threshold", r"threshold=([0-9.]+)", float),
        ("sharpness", r"sharpness=([0-9.]+)", float),
        ("transform_type", r"transform_type=([a-z]+)", str),
        ("seed", r"_s(\d+)$", int),
    ]:
        if key not in out:
            m = re.search(pat, run.name)
            if m:
                out[key] = cast(m.group(1))

    if "threshold" not in out or "sharpness" not in out or "seed" not in out:
        return None
    out.setdefault("transform_type", "sigmoid")
    return out


def _auc(run, metric: str, samples: int) -> float | None:
    h = run.history(keys=[metric], samples=samples)
    if h.empty or metric not in h.columns:
        return None
    h = h[["_step", metric]].dropna()
    if len(h) < 2:
        return None
    step = h["_step"].to_numpy(dtype=float)
    ret = h[metric].to_numpy(dtype=float)
    order = np.argsort(step)
    step, ret = step[order], ret[order]
    width = step[-1] - step[0]
    if width <= 0:
        return float(np.mean(ret))
    # step-normalised AUC (trapezoid integral / width); manual to avoid the
    # np.trapz/np.trapezoid rename across NumPy versions.
    area = float(np.sum(np.diff(step) * (ret[:-1] + ret[1:]) / 2.0))
    return area / float(width)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--group", default=DEFAULT_GROUP)
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--include-running", action="store_true",
                   help="Include non-finished runs (default: finished only).")
    args = p.parse_args()

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}",
                         filters={"group": args.group}, per_page=200))
    if not args.include_running:
        runs = [r for r in runs if r.state == "finished"]
    if not runs:
        raise SystemExit(
            f"No runs found in {args.entity}/{args.project} group={args.group!r}. "
            "Has the sweep finished? (try --include-running)"
        )

    # config key -> {seed: auc}
    per_cfg: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in sorted(runs, key=lambda r: r.name):
        c = _cfg_from_run(r)
        if c is None:
            print(f"  skip (no parseable config): {r.name}")
            continue
        auc = _auc(r, args.metric, args.samples)
        if auc is None:
            print(f"  skip (no metric history): {r.name}")
            continue
        key = (c["threshold"], c["sharpness"], c["transform_type"])
        per_cfg[key][c["seed"]] = auc

    if not per_cfg:
        raise SystemExit("No usable runs (configs/metric missing).")

    # Per-config aggregate over seeds.
    rows = []
    for (thr, shp, tt), seed_auc in per_cfg.items():
        vals = np.array(list(seed_auc.values()), dtype=float)
        n = len(vals)
        rows.append({
            "threshold": thr, "sharpness": shp, "transform_type": tt,
            "n_seeds": n,
            "auc_mean": float(vals.mean()),
            "auc_std": float(vals.std(ddof=1)) if n > 1 else 0.0,
            "auc_sem": float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "seeds": sorted(seed_auc),
        })
    agg = pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)

    print(f"\n=== CPR selectivity sweep — overall AUC (group={args.group}) ===")
    print(f"{'threshold':>9} {'sharpness':>9} {'transform':>9} {'n':>3} "
          f"{'AUC mean':>10} {'+/-SEM':>8}")
    for _, r in agg.iterrows():
        print(f"{r.threshold:>9g} {r.sharpness:>9g} {r.transform_type:>9} {r.n_seeds:>3d} "
              f"{r.auc_mean:>10.1f} {r.auc_sem:>8.1f}")

    # Paired comparison vs baseline (shared seeds only).
    base_key = next(
        (k for k in per_cfg if abs(k[0] - BASELINE_THRESHOLD) < 1e-9
         and abs(k[1] - BASELINE_SHARPNESS) < 1e-9),
        None,
    )
    if base_key is None:
        print(f"\n(no baseline config threshold={BASELINE_THRESHOLD}, "
              f"sharpness={BASELINE_SHARPNESS} found — skipping paired deltas)")
        return

    base = per_cfg[base_key]
    print(f"\n=== Paired delta vs baseline (threshold={base_key[0]:g}, "
          f"sharpness={base_key[1]:g}, {base_key[2]}) ===")
    print(f"{'threshold':>9} {'sharpness':>9} {'transform':>9} {'n_pair':>6} "
          f"{'mean_delta':>10} {'+/-SEM':>8} {'paired_z':>8}")
    deltas = []
    for key, seed_auc in per_cfg.items():
        if key == base_key:
            continue
        shared = sorted(set(seed_auc) & set(base))
        if not shared:
            continue
        d = np.array([seed_auc[s] - base[s] for s in shared], dtype=float)
        n = len(d)
        mean_d = float(d.mean())
        sem_d = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        z = mean_d / sem_d if (n > 1 and sem_d > 0) else float("nan")
        deltas.append((key, mean_d, sem_d, z, n))
    for (thr, shp, tt), mean_d, sem_d, z, n in sorted(deltas, key=lambda x: -x[1]):
        print(f"{thr:>9g} {shp:>9g} {tt:>9} {n:>6d} "
              f"{mean_d:>+10.1f} {sem_d:>8.1f} {z:>8.2f}")
    print("\n(paired_z = mean_delta / SEM; |z| >~ 2 suggests a real improvement "
          "over baseline at the swept seed count.)")


if __name__ == "__main__":
    main()
