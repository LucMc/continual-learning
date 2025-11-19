"""Plot transformation functions used in CCBP."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def exp_transform(x: np.ndarray, sharpness: float, threshold: float) -> np.ndarray:
    return np.minimum(np.exp(-sharpness * (x - threshold)), 1.0)


def sigmoid_transform(x: np.ndarray, sharpness: float, threshold: float) -> np.ndarray:
    return np.minimum(2.0 * (1.0 / (1.0 + np.exp(sharpness * (x - threshold)))), 1.0)


def softplus_transform(x: np.ndarray, sharpness: float, threshold: float) -> np.ndarray:
    shift = sharpness * (threshold - x)
    return np.minimum(np.log1p(np.exp(shift)) / np.log(2.0), 1.0)


def linear_transform(x: np.ndarray, sharpness: float, threshold: float) -> np.ndarray:
    return np.clip(1.0 - sharpness * (x - threshold), 0.0, 1.0)


def main() -> None:
    sharpness = 4.0
    threshold = 0.5
    x = np.linspace(-1.0, 2.0, 600)

    transforms = {
        "Exponential": exp_transform,
        "Sigmoid": sigmoid_transform,
        "Softplus": softplus_transform,
        "Linear": linear_transform,
    }

    base_fontsize = 20
    fig, ax = plt.subplots(figsize=(6, 6.5))

    for label, transform in transforms.items():
        y = transform(x, sharpness, threshold)
        ax.plot(x, y, label=label)

    ax.set_xlabel("utility", fontsize=base_fontsize)
    ax.set_ylabel("transformed utility", fontsize=base_fontsize)
    ax.set_title("CCBP transformation functions", fontsize=base_fontsize + 1)
    ax.legend(title=None,
              fontsize=base_fontsize - 1,
              title_fontsize=base_fontsize,
              loc="center right")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.1)
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.annotate("threshold", xy=(threshold, 1.05), xytext=(threshold + 0.15, 1.03),
                fontsize=base_fontsize - 1, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.tick_params(axis="both", which="major", labelsize=base_fontsize - 1)

    fig.tight_layout()

    output_path = Path(__file__).with_name("ccbp_transform_functions.png")
    fig.savefig(output_path, dpi=500)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
