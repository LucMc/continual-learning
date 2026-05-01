#!/usr/bin/env python3
"""Side-by-side comparison of a {prefix}-tagged sweep vs OLD source runs on metaworld_sac.

Pairs each {prefix}-prefixed method with the original run whose hyperparameters
were copied. Use to verify reproduction of the original results.

Usage:
    python -m continual_learning.plots.new_vs_old_comparison              # NEW vs OLD
    python -m continual_learning.plots.new_vs_old_comparison --prefix NEWNEW
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np
import wandb

ENTITY = "lucmc"
PROJECT = "metaworld_sac"

PAIRS = [
    ("sac_muon_ccbp",                "muon_ccbp"),
    ("sac_muon_cbp",                 "bb_muon_cbp_lp43"),
    ("sac_muon",                     "muon"),
    ("sac_adam",                     "adam"),
    ("sac_muon_shrink_and_perturb",  "bb_muon_shrink_and_perturbsoft"),
    ("sac_muon_redo",                "muon_redo"),
    ("sac_muon_regrama",             "muon_regrama"),
]

TARGET_STEPS = [
    500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000,
    3_500_000, 4_000_000, 4_500_000, 5_000_000, 5_500_000, 6_000_000,
    6_500_000, 7_000_000, 7_500_000, 8_000_000, 8_500_000, 9_000_000,
    9_500_000, 10_000_000,
]
MATURE_THRESHOLD = 3_500_000
FINAL_STEP = TARGET_STEPS[-1]
FINAL_REACHED_FRAC = 0.9


def _iqm_half_iqr(values: list[float]) -> tuple[float | None, float | None]:
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


def _fmt(values: list[float], width: int = 12) -> str:
    iqm, half = _iqm_half_iqr(values)
    if iqm is None:
        return f"{'--':^{width}}"
    if half is None or len(values) < 2:
        return f"{iqm:.3f}".rjust(width)
    return f"{iqm:.3f}±{half:.3f}".rjust(width)


def get_method_old(name: str) -> str:
    parts = name.replace("sac1msb_256hd", "").replace("sac1_bigbuf_256hd", "bb_")
    idx = parts.rfind("_")
    if idx > 0 and parts[idx + 1:].isdigit():
        return parts[:idx]
    return parts


def make_get_method_new(prefix: str):
    def _get(name: str) -> str:
        base = name[len(prefix):] if name.startswith(prefix) else name
        idx = base.rfind("_")
        if idx > 0 and base[idx + 1:].isdigit():
            return base[:idx]
        return base
    return _get


def collect(runs, method_of, wanted: set[str], mat_gate: bool = True,
            require_use_layer_norm: bool | None = None,
            min_max_step: float | None = None) -> dict:
    """Collect per-run summary metrics.

    mat_gate=True: Mat = mean over ALL checkpoints for runs that reached >=3.5M
        (matches muon_leaderboard --mat-gate, which makes Mat == Avg for those runs).
    require_use_layer_norm: if not None, only include runs whose
        config.algorithm.use_layer_norm matches this value.
    """
    by_method: dict[str, list] = defaultdict(list)
    for r in runs:
        m = method_of(r.name)
        if m not in wanted:
            continue
        if require_use_layer_norm is not None:
            algo = r.config.get("algorithm", {})
            if isinstance(algo, dict) and algo.get("use_layer_norm") != require_use_layer_norm:
                continue
        by_method[m].append(r)

    out = {}
    for method, mruns in by_method.items():
        per_run_avgs: list[float] = []
        per_run_mature_avgs: list[float] = []
        per_run_finals: list[float] = []
        max_step = 0

        for r in mruns:
            try:
                hist = r.history(
                    keys=["eval/average_success_rate", "charts/total_steps"],
                    samples=2000,
                    pandas=True,
                )
            except Exception:
                continue
            if hist.empty or "eval/average_success_rate" not in hist.columns:
                continue
            hist = hist.dropna(subset=["eval/average_success_rate"])
            if hist.empty:
                continue

            steps = hist["charts/total_steps"].values
            vals = hist["eval/average_success_rate"].values
            run_max = float(np.nanmax(steps))
            max_step = max(max_step, run_max)

            if min_max_step is not None and run_max < min_max_step:
                continue

            run_vals: list[float] = []
            run_mature_vals: list[float] = []
            for target in TARGET_STEPS:
                if target > run_max * 1.05:
                    continue
                idx = int(np.argmin(np.abs(steps - target)))
                run_vals.append(float(vals[idx]))
                if target >= MATURE_THRESHOLD:
                    run_mature_vals.append(float(vals[idx]))

            if mat_gate:
                run_mature_vals = list(run_vals) if run_max >= MATURE_THRESHOLD else []

            if run_vals:
                per_run_avgs.append(float(np.mean(run_vals)))
            if run_mature_vals:
                per_run_mature_avgs.append(float(np.mean(run_mature_vals)))
            if run_max >= FINAL_STEP * FINAL_REACHED_FRAC:
                final_idx = int(np.argmin(np.abs(steps - FINAL_STEP)))
                per_run_finals.append(float(vals[final_idx]))

        out[method] = {
            "n_runs": len(mruns),
            "max_step": max_step,
            "per_run_avgs": per_run_avgs,
            "per_run_mature_avgs": per_run_mature_avgs,
            "per_run_finals": per_run_finals,
        }
    return out


def _delta(new_vals: list[float], old_vals: list[float]) -> str:
    n_iqm, _ = _iqm_half_iqr(new_vals)
    o_iqm, _ = _iqm_half_iqr(old_vals)
    if n_iqm is None or o_iqm is None:
        return f"{'--':>8}"
    d = n_iqm - o_iqm
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}".rjust(8)


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="NEW",
                        help="Tag prefix for the reproduction sweep (e.g. NEW, NEWNEW).")
    parser.add_argument("--min-steps", type=float, default=3.5e6,
                        help="Drop runs whose max recorded step is below this. "
                        "Avoids contaminating Avg with short crashed runs that only saw the easy first task.")
    args = parser.parse_args()
    prefix = args.prefix
    min_max_step = float(args.min_steps) if args.min_steps and args.min_steps > 0 else None

    wanted_new = {p[0] for p in PAIRS}
    wanted_old = {p[1] for p in PAIRS}

    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    api = wandb.Api()
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    new_runs = [r for r in all_runs if r.name.startswith(prefix)]
    old_runs = [r for r in all_runs if not r.name.startswith("NEW")]
    print(f"  {len(new_runs)} {prefix} runs, {len(old_runs)} OLD candidates")

    get_method_new = make_get_method_new(prefix)

    # Match muon_leaderboard's filter for OLD: use_layer_norm=False, mat_gate semantics.
    # New runs are reported as-is (mat_gate=True so Mat==Avg, matching the OLD definition).
    new_data = collect(new_runs, get_method_new, wanted_new, mat_gate=True,
                       min_max_step=min_max_step)
    old_data = collect(old_runs, get_method_old, wanted_old, mat_gate=True,
                       require_use_layer_norm=False, min_max_step=min_max_step)

    print()
    if prefix == "NEW":
        print("NEW = batch_size=128, lr=1e-4, replay_ratio=12, use_layer_norm=True")
        print("OLD = batch_size=256, lr=1e-2, replay_ratio=4,  use_layer_norm=False (source sweep)")
        print("NOTE: LayerNorm differs between sweeps -- this is a config delta, not just batch_size.")
    else:
        print(f"{prefix} = corrected reproduction (matched config to OLD source).")
        print("OLD = source sweep (use_layer_norm=False filter applied).")
    print("Mat == Avg here (mat_gate mode, matching the source leaderboard).")
    print()

    new_label = prefix
    new_col_header = f"{new_label} method"

    # ---- per metric: side-by-side ----
    n_label = f"N_{new_label}"
    width = 32 + 32 + 12 + 6 + 12 + 6 + 8
    for metric_label, key in [
        ("Avg (mean success over all checkpoints)", "per_run_avgs"),
    ]:
        print("=" * width)
        print(metric_label)
        print("=" * width)
        print(
            f'{new_col_header:<32}{"OLD method":<32}'
            f'{new_label:>12} {n_label:>6} {"OLD":>12} {"N_OLD":>6} {"Δ":>8}'
        )
        print("-" * width)
        for new_m, old_m in PAIRS:
            n = new_data.get(new_m, {})
            o = old_data.get(old_m, {})
            n_vals = n.get(key, [])
            o_vals = o.get(key, [])
            print(
                f'{new_m:<32}{old_m:<32}'
                f'{_fmt(n_vals):>12} {len(n_vals):>6d} '
                f'{_fmt(o_vals):>12} {len(o_vals):>6d} '
                f'{_delta(n_vals, o_vals):>8}'
            )
        print()

    # ---- run counts and coverage ----
    print("=" * 90)
    print("Coverage (N seeds | max steps reached)")
    print("=" * 90)
    print(
        f'{new_col_header:<32}{"OLD method":<32}'
        f'{f"{new_label} (N|max)":>16} {"OLD (N|max)":>16}'
    )
    print("-" * 90)
    for new_m, old_m in PAIRS:
        n = new_data.get(new_m, {"n_runs": 0, "max_step": 0})
        o = old_data.get(old_m, {"n_runs": 0, "max_step": 0})
        n_str = f'{n["n_runs"]} | {n["max_step"]/1e6:.1f}M'
        o_str = f'{o["n_runs"]} | {o["max_step"]/1e6:.1f}M'
        print(f'{new_m:<32}{old_m:<32}{n_str:>16} {o_str:>16}')
    print()
    print(f"Δ = {new_label} - OLD (negative = {new_label} underperforms the source sweep)")


if __name__ == "__main__":
    main()
