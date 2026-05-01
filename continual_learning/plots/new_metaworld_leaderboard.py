#!/usr/bin/env python3
"""Generate a leaderboard for the recent 'NEW' sweep on metaworld_sac.

These runs use batch_size=128 (vs 256 in the older sweep) and a smaller
learning rate. Use this script to verify reproduction across variants.

Usage:
    python -m continual_learning.plots.new_metaworld_leaderboard
    python -m continual_learning.plots.new_metaworld_leaderboard --min-steps 8
    python -m continual_learning.plots.new_metaworld_leaderboard --sort mat
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np
import wandb

ENTITY = "lucmc"
PROJECT = "metaworld_sac"
NAME_PREFIX = "NEW"

TARGET_STEPS = [
    500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000,
    3_500_000, 4_000_000, 4_500_000, 5_000_000, 5_500_000, 6_000_000,
    6_500_000, 7_000_000, 7_500_000, 8_000_000, 8_500_000, 9_000_000,
    9_500_000, 10_000_000,
]

COLS_LABEL = [
    "0.5M", "1M", "1.5M", "2M", "2.5M", "3M", "3.5M", "4M", "4.5M", "5M",
    "5.5M", "6M", "6.5M", "7M", "7.5M", "8M", "8.5M", "9M", "9.5M", "10M",
]

MATURE_THRESHOLD = 3_500_000
FINAL_STEP = TARGET_STEPS[-1]
FINAL_REACHED_FRAC = 0.9  # softer than muon (some NEW runs crashed at ~9.3M)


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


def _fmt_iqm_plain(values: list[float], fallback: float | None) -> str:
    iqm, half = _iqm_half_iqr(values)
    if iqm is None:
        if fallback is None:
            return "  n/a        "
        return f"{fallback:.3f}      "
    if half is None or len(values) < 2:
        return f"{iqm:.3f}      "
    return f"{iqm:.3f}±{half:.3f}"


def get_method(name: str) -> str:
    """Strip the NEW prefix and trailing _<seed>: NEWsac_muon_redo_0 -> sac_muon_redo."""
    base = name[len(NAME_PREFIX):] if name.startswith(NAME_PREFIX) else name
    idx = base.rfind("_")
    if idx > 0 and base[idx + 1:].isdigit():
        return base[:idx]
    return base


def fetch_results(final_step: int = FINAL_STEP, mat_gate: bool = False) -> dict:
    api = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"display_name": {"$regex": f"^{NAME_PREFIX}"}},
    )

    method_runs: dict[str, list] = defaultdict(list)
    for r in runs:
        method_runs[get_method(r.name)].append(r)

    results = {}
    for method, mruns in sorted(method_runs.items()):
        step_values: dict[int, list[float]] = defaultdict(list)
        max_step = 0
        states = [r.state for r in mruns]
        per_run_avgs: list[float] = []
        per_run_mature_avgs: list[float] = []
        per_run_finals: list[float] = []

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
            run_max = np.nanmax(steps)
            max_step = max(max_step, run_max)

            run_vals: list[float] = []
            run_mature_vals: list[float] = []
            for target in TARGET_STEPS:
                if target > run_max * 1.05:
                    continue
                idx = int(np.argmin(np.abs(steps - target)))
                step_values[target].append(vals[idx])
                run_vals.append(vals[idx])
                if target >= MATURE_THRESHOLD:
                    run_mature_vals.append(vals[idx])

            if mat_gate:
                run_mature_vals = list(run_vals) if run_max >= MATURE_THRESHOLD else []

            if run_vals:
                per_run_avgs.append(float(np.mean(run_vals)))
            if run_mature_vals:
                per_run_mature_avgs.append(float(np.mean(run_mature_vals)))
            if run_max >= final_step * FINAL_REACHED_FRAC:
                final_idx = int(np.argmin(np.abs(steps - final_step)))
                per_run_finals.append(float(vals[final_idx]))

        sv = step_values
        all_means = [np.mean(sv[s]) for s in TARGET_STEPS if s in sv and sv[s]]
        overall_avg = np.mean(all_means) if all_means else 0.0
        mature_means = [
            np.mean(sv[s])
            for s in TARGET_STEPS
            if s >= MATURE_THRESHOLD and s in sv and sv[s]
        ]
        mature_avg = np.mean(mature_means) if mature_means else None

        final_avg = float(np.mean(per_run_finals)) if per_run_finals else None

        results[method] = {
            "step_values": sv,
            "max_step": max_step,
            "n_runs": len(mruns),
            "states": states,
            "overall_avg": overall_avg,
            "mature_avg": mature_avg,
            "final_avg": final_avg,
            "per_run_avgs": per_run_avgs,
            "per_run_mature_avgs": per_run_mature_avgs,
            "per_run_finals": per_run_finals,
        }

    return results


def print_full_table(results: dict, sort_key: str = "overall_avg", min_steps: float = 0,
                     final_step: int = FINAL_STEP, mat_gate: bool = False) -> None:
    filtered = {
        m: r for m, r in results.items()
        if r["max_step"] >= min_steps * 1_000_000
    }
    if not filtered:
        print("No methods match the filter criteria.")
        return

    if sort_key == "mat":
        sorted_methods = sorted(
            filtered.keys(),
            key=lambda m: filtered[m]["mature_avg"] if filtered[m]["mature_avg"] is not None else -1,
            reverse=True,
        )
    else:
        sorted_methods = sorted(
            filtered.keys(),
            key=lambda m: filtered[m]["overall_avg"],
            reverse=True,
        )

    header = (
        f'{"#":>2} {"Method":<36} {"N":>2} {"St":>4} {"Max":>5} '
        f'{"Avg (IQR)":>14} {"Mat (IQR)":>14} {"Final (IQR)":>14}'
    )
    for c in COLS_LABEL:
        header += f" {c:>5}"
    print(header)
    print("=" * len(header))

    for rank, method in enumerate(sorted_methods, 1):
        r = filtered[method]
        sv = r["step_values"]
        n_running = sum(1 for s in r["states"] if s == "running")
        st_str = f"{n_running}R" if n_running else "fin"

        avg_str = _fmt_iqm_plain(r["per_run_avgs"], r["overall_avg"])
        mat_str = _fmt_iqm_plain(r["per_run_mature_avgs"], r["mature_avg"])
        fin_str = _fmt_iqm_plain(r["per_run_finals"], r["final_avg"])

        line = (
            f'{rank:>2} {method:<36} {r["n_runs"]:>2} {st_str:>4} '
            f'{r["max_step"]/1e6:>4.1f}M {avg_str:>14} {mat_str:>14} {fin_str:>14}'
        )
        for s in TARGET_STEPS:
            if s in sv and sv[s]:
                mean = np.mean(sv[s])
                n = len(sv[s])
                marker = "*" if n > 1 else " "
                line += f" {mean:.2f}{marker}"
            else:
                line += "   -- "
        print(line)

    n_total = sum(r["n_runs"] for r in filtered.values())
    n_running = sum(
        sum(1 for s in r["states"] if s == "running")
        for r in filtered.values()
    )
    n_done = sum(
        sum(1 for s in r["states"] if s != "running")
        for r in filtered.values()
    )
    print()
    print(f"Total: {len(filtered)} methods | {n_total} runs | {n_running} running | {n_done} finished/crashed")
    mat_desc = (
        f"full 0-max range for runs reaching >= {MATURE_THRESHOLD/1e6:.1f}M"
        if mat_gate else f"mean over steps >= {MATURE_THRESHOLD/1e6:.1f}M"
    )
    print(
        "* = multi-seed mean | St = running(R)/finished(fin) | "
        f"Avg = overall mean | Mat = {mat_desc} | "
        f"Final = IQM at {final_step/1e6:.2f}M (runs reaching >= {final_step*FINAL_REACHED_FRAC/1e6:.2f}M)"
    )


def print_mature_ranking(results: dict, threshold: int = 8_000_000) -> None:
    mature = [
        (m, r) for m, r in results.items()
        if r["max_step"] >= threshold and r["mature_avg"] is not None
    ]
    if not mature:
        print(f"\nNo mature runs (>= {threshold/1e6:.0f}M) found.")
        return

    mature.sort(key=lambda x: x[1]["mature_avg"], reverse=True)

    print()
    print("=" * 60)
    print(f"MATURE RANKING (data >= {threshold/1e6:.0f}M, sorted by Mat)")
    print("=" * 60)
    print(
        f'{"#":>2} {"Method":<36} {"N":>2} {"Max":>5} '
        f'{"Mat (IQR)":>14} {"Avg (IQR)":>14} {"Final (IQR)":>14}'
    )
    print("-" * 96)
    for i, (m, r) in enumerate(mature, 1):
        marker = " <-- baseline" if m == "sac_adam" else ""
        mat_str = _fmt_iqm_plain(r["per_run_mature_avgs"], r["mature_avg"])
        avg_str = _fmt_iqm_plain(r["per_run_avgs"], r["overall_avg"])
        fin_str = _fmt_iqm_plain(r["per_run_finals"], r["final_avg"])
        print(
            f'{i:>2} {m:<36} {r["n_runs"]:>2} {r["max_step"]/1e6:>4.1f}M '
            f'{mat_str:>14} {avg_str:>14} {fin_str:>14}{marker}'
        )


def main():
    parser = argparse.ArgumentParser(
        description="Leaderboard for NEW-prefixed metaworld_sac runs (batch_size=128 reproduction)"
    )
    parser.add_argument("--sort", choices=["avg", "mat"], default="avg",
                        help="Sort by overall avg or mature avg (default: avg)")
    parser.add_argument("--min-steps", type=float, default=0,
                        help="Only show methods with max_step >= this many M (e.g. 8 for >= 8M)")
    parser.add_argument("--no-mature", action="store_true",
                        help="Skip the mature ranking table")
    parser.add_argument("--mature-threshold", type=float, default=8.0,
                        help="Min max_step in M to include in the mature ranking (default: 8)")
    parser.add_argument("--final-step", type=float, default=FINAL_STEP / 1e6,
                        help="Step (in millions) to use for the Final IQM column (default: 10)")
    parser.add_argument("--mat-gate", action="store_true",
                        help="Treat the 3.5M threshold as a per-run gate: Mat averages over "
                             "ALL checkpoints (0-max) for runs that reached >= 3.5M.")
    args = parser.parse_args()

    final_step = int(args.final_step * 1_000_000)

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print(f"Fetching {NAME_PREFIX}-prefixed runs from {ENTITY}/{PROJECT}...")
    print(f"Final-step metric uses step = {final_step/1e6:.2f}M "
          f"(runs must reach >= {final_step*FINAL_REACHED_FRAC/1e6:.2f}M)")
    if args.mat_gate:
        print(f"Mat mode: gate (runs must reach >= {MATURE_THRESHOLD/1e6:.1f}M, averaged over full 0-max range)")
    else:
        print(f"Mat mode: per-checkpoint filter (only steps >= {MATURE_THRESHOLD/1e6:.1f}M)")
    print()
    results = fetch_results(final_step=final_step, mat_gate=args.mat_gate)

    print_full_table(results, sort_key=args.sort, min_steps=args.min_steps,
                     final_step=final_step, mat_gate=args.mat_gate)

    if not args.no_mature:
        print_mature_ranking(results, threshold=int(args.mature_threshold * 1_000_000))


if __name__ == "__main__":
    main()
