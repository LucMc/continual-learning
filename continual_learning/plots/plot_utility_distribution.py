#!/usr/bin/env python3
"""
Visualize CPR utility distributions throughout training.

This script creates a multi-panel visualization showing how utility distributions
evolve from initialization through training, demonstrating the gradient-based
utility calculation and continuous probabilistic reset mechanism.

Phase 1: Uses approximated distributions from mean/std statistics
Phase 2 (future): Can use exact histograms when available from enhanced logging
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tyro
import wandb
from matplotlib.patches import Rectangle
from scipy import stats

# Plotting style configuration
plt.rcParams["font.family"] = "Times New Roman"
BASE_FONTSIZE = 20
COLORS = {
    "utility_hist": "#3498db",  # Blue for utility histogram
    "reset_prob": "#e74c3c",  # Red for reset probability curve
    "threshold": "#2c3e50",  # Dark gray for threshold line
    "low_util_region": "#f39c12",  # Orange highlight for low utility
}


@dataclass
class UtilitySnapshot:
    """Container for utility distribution statistics at one time point."""

    step: int
    mean_util: float
    std_util: float
    low_utility_count: Optional[int] = None
    network: str = "value"  # 'actor' or 'value'
    # Optional: actual histogram data if available (Phase 2)
    histogram_counts: Optional[np.ndarray] = None
    bin_edges: Optional[np.ndarray] = None


def fetch_utility_statistics(
    entity: str,
    project: str,
    group: str,
    run_name_pattern: str,
    network: str = "value",
    target_steps: List[int] = [1_000_000, 50_000_000, 150_000_000, 200_000_000],
) -> Tuple[str, List[UtilitySnapshot]]:
    """
    Fetch utility statistics from W&B at specific training steps.

    Args:
        entity: W&B entity name
        project: W&B project name
        group: W&B group name
        run_name_pattern: Pattern to match run names (e.g., "ccbp")
        network: Which network to fetch ("actor" or "value")
        target_steps: Training steps to extract (init, early, mid, late)

    Returns:
        Tuple of (run_name, List of UtilitySnapshot objects)
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    # Find CPR runs matching pattern, sorted by creation time (newest first)
    matching_runs = sorted(
        [r for r in runs if run_name_pattern in r.name and r.state == "finished"],
        key=lambda r: r.created_at,
        reverse=True,
    )

    if not matching_runs:
        raise ValueError(
            f"No finished runs found matching '{run_name_pattern}' in group '{group}'"
        )

    # Try runs until we find one with histogram data
    run = None
    history = None
    hist_key = f"{network}/utility_histogram"
    mean_key = f"{network}/mean_utils"
    std_key = f"{network}/std_util"
    low_util_key = f"{network}/low_utility"

    for candidate_run in matching_runs[:10]:  # Try up to 10 most recent runs
        print(f"Trying run: {candidate_run.name} (id: {candidate_run.id})")
        candidate_history = candidate_run.history(
            keys=[mean_key, std_key, low_util_key, hist_key, "_step"], samples=5000
        )
        if not candidate_history.empty and hist_key in candidate_history.columns:
            run = candidate_run
            history = candidate_history
            print(f"  ✓ Found histogram data!")
            break
        else:
            print(f"  ✗ No histogram data, trying next...")

    if run is None or history is None:
        raise ValueError(
            f"No runs with histogram data found matching '{run_name_pattern}' in group '{group}'"
        )

    print(f"\nUsing run: {run.name} (id: {run.id})")

    snapshots = []
    has_histogram_data = hist_key in history.columns
    print("✓ Found histogram data in W&B logs - will use exact distributions")

    for target_step in target_steps:
        # Find closest logged step to target
        closest_idx = (history["_step"] - target_step).abs().idxmin()
        row = history.loc[closest_idx]

        # Extract histogram if available
        hist_counts = None
        hist_edges = None
        if has_histogram_data and hist_key in row and row[hist_key] is not None:
            try:
                # W&B histogram is stored as dict with 'values' (counts) and 'bins' (edges)
                hist_obj = row[hist_key]
                if isinstance(hist_obj, dict):
                    # W&B stores histogram as {'values': [...], 'bins': [...], '_type': 'histogram'}
                    if 'values' in hist_obj and 'bins' in hist_obj:
                        hist_counts = np.array(hist_obj['values'])
                        hist_edges = np.array(hist_obj['bins'])
                    # Fallback to other possible dict formats
                    elif 'histogram' in hist_obj or 'counts' in hist_obj:
                        hist_counts = np.array(hist_obj.get('histogram', hist_obj.get('counts', [])))
                        hist_edges = np.array(hist_obj.get('bins', hist_obj.get('edges', [])))
                elif hasattr(hist_obj, 'bins') and hasattr(hist_obj, 'histogram'):
                    # W&B Histogram object format (older API)
                    hist_counts = np.array(hist_obj.histogram)
                    hist_edges = np.array(hist_obj.bins)
            except Exception as e:
                print(f"  Warning: Could not parse histogram for step {target_step}: {e}")

        snapshot = UtilitySnapshot(
            step=int(row["_step"]),
            mean_util=float(row[mean_key]) if mean_key in row and not np.isnan(row[mean_key]) else 1.0,
            std_util=float(row[std_key]) if std_key in row and not np.isnan(row[std_key]) else 0.0,
            low_utility_count=int(row[low_util_key]) if low_util_key in row and not np.isnan(row[low_util_key]) else None,
            network=network,
            histogram_counts=hist_counts,
            bin_edges=hist_edges,
        )
        snapshots.append(snapshot)
        hist_status = "✓ histogram" if snapshot.histogram_counts is not None else "approximated"
        print(f"  Step {snapshot.step}: mean={snapshot.mean_util:.3f}, std={snapshot.std_util:.3f} [{hist_status}]")

    return run.name, snapshots


def approximate_distribution(
    mean: float,
    std: float,
    n_bins: int = 50,
    utility_range: Tuple[float, float] = (0.0, 2.0),
    n_neurons: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate utility distribution using Beta distribution rescaled to utility range.

    The Beta distribution is chosen because:
    - Naturally bounded (we rescale to [0, 2])
    - Flexible shape that can match various mean/std combinations
    - Two parameters can be fit to observed mean/std

    Args:
        mean: Mean utility value
        std: Standard deviation of utilities
        n_bins: Number of histogram bins
        utility_range: Min and max utility values
        n_neurons: Number of neurons to simulate (affects histogram height)

    Returns:
        Tuple of (bin_counts, bin_edges)
    """
    u_min, u_max = utility_range
    range_width = u_max - u_min

    # Normalize mean and std to [0, 1] for Beta distribution
    mean_norm = (mean - u_min) / range_width
    std_norm = std / range_width

    # Prevent edge cases
    mean_norm = np.clip(mean_norm, 0.01, 0.99)
    std_norm = min(std_norm, 0.49)  # Beta std cannot exceed 0.5

    # Fit Beta distribution parameters
    # mean = alpha / (alpha + beta)
    # var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))

    if std_norm < 0.001:  # Near-zero variance, use very peaked distribution
        alpha = mean_norm * 1000
        beta = (1 - mean_norm) * 1000
    else:
        var_norm = std_norm**2
        # Solve for alpha, beta
        common = (mean_norm * (1 - mean_norm) / var_norm) - 1
        alpha = mean_norm * common
        beta = (1 - mean_norm) * common

        # Ensure positive parameters
        alpha = max(alpha, 0.5)
        beta = max(beta, 0.5)

    # Generate Beta distribution samples
    beta_dist = stats.beta(alpha, beta)

    # Create bins and compute density
    bin_edges = np.linspace(u_min, u_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize bin centers to [0, 1] for Beta PDF evaluation
    bin_centers_norm = (bin_centers - u_min) / range_width

    # Compute PDF values and rescale
    pdf_values = beta_dist.pdf(bin_centers_norm)

    # Convert to counts (integrate to n_neurons)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_counts = pdf_values * (n_neurons * bin_width / range_width)

    return bin_counts, bin_edges


def compute_reset_probability(
    utility_values: np.ndarray,
    replacement_rate: float = 0.012,
    threshold: float = 0.95,
    sharpness: float = 16.0,
    transform_type: str = "exp",
) -> np.ndarray:
    """
    Compute reset probability from utility values.

    Matches CPR implementation exactly (ccbp.py lines 66-74).

    Args:
        utility_values: Array of utility values
        replacement_rate: Maximum reset probability (CPR parameter)
        threshold: Utility threshold separating low/high utility (τ)
        sharpness: Steepness of transform curve
        transform_type: Transform function type ('exp', 'sigmoid', 'softplus', 'linear')

    Returns:
        Array of reset probabilities (same shape as utility_values)
    """
    if transform_type == "exp":
        transform = np.minimum(
            np.exp(-sharpness * (utility_values - threshold)), 1.0
        )
    elif transform_type == "sigmoid":
        transform = np.minimum(
            2.0 / (1.0 + np.exp(sharpness * (utility_values - threshold))), 1.0
        )
    elif transform_type == "softplus":
        shift = sharpness * (threshold - utility_values)
        transform = np.minimum(np.log1p(np.exp(shift)) / np.log(2.0), 1.0)
    elif transform_type == "linear":
        transform = np.clip(1.0 - sharpness * (utility_values - threshold), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    return replacement_rate * transform


def plot_utility_panel(
    ax: plt.Axes,
    snapshot: UtilitySnapshot,
    replacement_rate: float = 0.012,
    threshold: float = 1.0,
    sharpness: float = 16.0,
    transform_type: str = "exp",
    n_neurons: int = 1000,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Create one panel showing utility distribution + reset probability.

    Args:
        ax: Matplotlib axes for the panel
        snapshot: Utility statistics for this time point
        replacement_rate: CPR replacement rate
        threshold: CPR threshold (τ)
        sharpness: CPR sharpness parameter
        transform_type: Transform function type
        n_neurons: Number of neurons (for histogram scaling)

    Returns:
        Tuple of (ax_hist, ax_prob) - histogram axis and probability axis
    """
    # Use real histogram if available, otherwise approximate
    if snapshot.histogram_counts is not None and snapshot.bin_edges is not None:
        # Use exact histogram from training logs
        bin_counts = snapshot.histogram_counts
        bin_edges = snapshot.bin_edges
        is_approximated = False
    else:
        # Approximate distribution from mean/std
        bin_counts, bin_edges = approximate_distribution(
            snapshot.mean_util, snapshot.std_util, n_bins=50, n_neurons=n_neurons
        )
        is_approximated = True

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Plot histogram (left y-axis)
    ax.bar(
        bin_centers,
        bin_counts,
        width=bin_width,
        alpha=0.7,
        color=COLORS["utility_hist"],
        label="Neuron count",
        edgecolor="white",
        linewidth=1.0,
    )

    # Configure left y-axis (histogram)
    ax.set_ylabel("Number of neurons", fontsize=BASE_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=BASE_FONTSIZE - 2)
    ax.set_ylim(0, 200)  # Fixed y-axis for easier comparison across panels
    ax.set_xlabel("Utility", fontsize=BASE_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=BASE_FONTSIZE - 2)

    # Create second y-axis for reset probability
    ax2 = ax.twinx()

    # Compute reset probability curve
    utility_range = np.linspace(0, 2.0, 200)
    reset_prob = compute_reset_probability(
        utility_range, replacement_rate, threshold, sharpness, transform_type
    )

    # Plot reset probability curve (right y-axis)
    ax2.plot(
        utility_range,
        reset_prob,
        color=COLORS["reset_prob"],
        linewidth=4.0,
        label="Reset probability",
        zorder=10,
    )

    # Configure right y-axis (probability)
    ax2.set_ylabel(
        "Reset probability",
        fontsize=BASE_FONTSIZE,
        fontweight="bold",
        color=COLORS["reset_prob"],
    )
    ax2.tick_params(
        axis="y", labelcolor=COLORS["reset_prob"], labelsize=BASE_FONTSIZE - 2
    )
    ax2.set_ylim(0, replacement_rate * 1.1)  # Slightly above max possible

    # Add threshold line
    ax.axvline(
        threshold,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=3.0,
        alpha=0.8,
        zorder=5,
    )

    # Highlight low-utility region
    ax.add_patch(
        Rectangle(
            (0, 0),
            threshold,
            ax.get_ylim()[1],
            facecolor=COLORS["low_util_region"],
            alpha=0.05,
            zorder=0,
        )
    )

    # Set x-axis limits
    ax.set_xlim(0, 2.0)

    # Add panel title (training step) - round to nearest million if close
    step_millions = snapshot.step / 1_000_000
    rounded_millions = round(step_millions)
    # Use rounded value if within 5% of a million boundary
    if rounded_millions > 0 and abs(step_millions - rounded_millions) / rounded_millions < 0.05:
        title = f"{rounded_millions:.0f}M steps"
    elif step_millions < 1:
        title = f"{snapshot.step / 1_000:.0f}K steps"
    else:
        title = f"{step_millions:.0f}M steps"

    ax.set_title(title, fontsize=BASE_FONTSIZE + 1, fontweight="bold", pad=10)

    # Add statistics text box
    stats_text = f"μ = {snapshot.mean_util:.3f}\nσ = {snapshot.std_util:.3f}"
    if is_approximated:
        stats_text += "\n(approx.)"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=BASE_FONTSIZE - 3,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return ax, ax2


def create_utility_evolution_plot(
    snapshots: List[UtilitySnapshot],
    replacement_rate: float = 0.012,
    threshold: float = 1.0,
    sharpness: float = 16.0,
    transform_type: str = "exp",
    n_neurons: int = 1000,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create 2x2 panel figure showing utility evolution.

    Args:
        snapshots: List of 4 UtilitySnapshot objects (init, early, mid, late)
        replacement_rate: CPR replacement rate parameter
        threshold: CPR threshold parameter (τ)
        sharpness: CPR sharpness parameter
        transform_type: Transform function type
        n_neurons: Number of neurons (for scaling)
        output_path: Where to save figure (if None, don't save)

    Returns:
        Matplotlib figure object
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot each snapshot
    for idx, snapshot in enumerate(snapshots[:4]):  # Ensure max 4 panels
        plot_utility_panel(
            axes[idx],
            snapshot,
            replacement_rate,
            threshold,
            sharpness,
            transform_type,
            n_neurons,
        )

    # Add overall figure title
    fig.suptitle(
        "CPR Utility Distribution Throughout Training",
        fontsize=BASE_FONTSIZE + 4,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Saved plot to: {output_path}")

    return fig


def main(
    wandb_entity: str = "lucmc",
    wandb_project: str = "crl_experiments",
    group: str = "slippery_ant_full2",
    run_pattern: str = "ccbp_new",
    seeds: List[int] = [0, 1, 2, 3],
    network: str = "value",
    target_steps: List[int] = [1_000_000, 50_000_000, 150_000_000, 200_000_000],
    replacement_rate: float = 0.012,
    threshold: float = 1.0,
    sharpness: float = 16.0,
    transform_type: str = "exp",
    n_neurons: int = 1000,
    output_dir: str = "./plots/utility_distribution",
    ext: str = "png",
    show_plot: bool = False,
) -> None:
    """
    Generate CPR utility distribution visualization.

    Example usage:
        python plot_utility_distribution.py \\
            --wandb-entity lucmc \\
            --wandb-project crl_experiments \\
            --group slippery_ant \\
            --network value \\
            --seeds 0 1 2 3 \\
            --output-dir plots/utility_dist

    Args:
        wandb_entity: W&B entity name
        wandb_project: W&B project name
        group: W&B group name
        run_pattern: Base pattern to match in run names (seed will be appended)
        seeds: List of seeds to generate plots for
        network: Network to visualize ('actor' or 'value')
        target_steps: Training steps to visualize
        replacement_rate: CPR replacement rate
        threshold: CPR threshold τ
        sharpness: CPR sharpness parameter
        transform_type: Transform function ('exp', 'sigmoid', 'softplus', 'linear')
        n_neurons: Approximate number of neurons (for scaling)
        output_dir: Output directory for plots
        ext: Output file extension ('png', 'svg', 'pdf')
        show_plot: Whether to display plot interactively
    """
    print(f"\n{'='*60}")
    print(f"CPR Utility Distribution Visualization")
    print(f"{'='*60}")
    print(f"Fetching from: {wandb_entity}/{wandb_project}/{group}")
    print(f"Network: {network}")
    print(f"Seeds: {seeds}")
    print(f"Target steps: {[f'{s/1e6:.0f}M' if s > 0 else '0' for s in target_steps]}")
    print(f"{'='*60}\n")

    for seed in seeds:
        seed_run_pattern = f"{run_pattern}_{seed}"
        print(f"\n{'─'*60}")
        print(f"Processing seed {seed} (pattern: {seed_run_pattern})")
        print(f"{'─'*60}")

        try:
            # Fetch data from W&B
            run_name, snapshots = fetch_utility_statistics(
                wandb_entity, wandb_project, group, seed_run_pattern, network, target_steps
            )

            print(f"\n✓ Found {len(snapshots)} snapshots:")
            for snap in snapshots:
                print(
                    f"  Step {snap.step:>10}: mean={snap.mean_util:.3f}, std={snap.std_util:.3f}"
                )

            # Create visualization with run name in filename
            output_path = (
                Path(output_dir) / ext / f"cpr_utility_evolution_{run_name}_{network}.{ext}"
            )

            print(f"\nGenerating visualization...")
            print(f"  Replacement rate: {replacement_rate}")
            print(f"  Threshold: {threshold}")
            print(f"  Sharpness: {sharpness}")
            print(f"  Transform: {transform_type}")

            fig = create_utility_evolution_plot(
                snapshots,
                replacement_rate,
                threshold,
                sharpness,
                transform_type,
                n_neurons,
                output_path,
            )

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        except ValueError as e:
            print(f"  ✗ Skipping seed {seed}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"✓ Visualization complete for {len(seeds)} seeds!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    tyro.cli(main)
