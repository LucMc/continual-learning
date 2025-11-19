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
    # sharpness = 4.0
    # threshold = 0.5
    sharpness = 10
    threshold = 0.75
    replacement_rate = 0.2
    x = np.linspace(-1.0, 2.0, 600)

    transforms = {
        "Exponential": exp_transform,
        "Linear": linear_transform,
        "Sigmoid": sigmoid_transform,
        "Softplus": softplus_transform,
    }

    base_fontsize = 16
    fig, ax = plt.subplots(figsize=(6, 6.75))

    colors = {
        "Sigmoid": "red",
        "Softplus": "teal",
    }

    for label, transform in transforms.items():
        y = replacement_rate * transform(x, sharpness, threshold)
        area = np.trapz(y, x)
        print(f"{label} transformation area under curve: {area:.4f}")
        plot_kwargs = {"label": label}
        if label in colors:
            plot_kwargs["color"] = colors[label]
        ax.plot(x, y, **plot_kwargs)

    ax.set_xlabel("Utility", fontsize=base_fontsize, fontweight="bold")
    ax.set_ylabel("Transformed Utility", fontsize=base_fontsize, fontweight="bold")
    ax.set_title("CCBP Transformation Functions", fontsize=base_fontsize + 1)
    ax.legend(title=None,
              fontsize=base_fontsize - 1,
              title_fontsize=base_fontsize,
              loc="center right")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 0.3)
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.annotate(r"Threshold $\tau$", xy=(threshold, 0.22), xytext=(threshold + 0.15, 0.22),
                fontsize=base_fontsize - 1, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.tick_params(axis="both", which="major", labelsize=base_fontsize - 1)

    fig.tight_layout()

    output_path = Path(__file__).with_name("ccbp_transform_functions.png")
    fig.savefig(output_path, dpi=500)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
