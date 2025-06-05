from typing import Any

import flax.struct
import flax.traverse_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import wandb
from jaxtyping import Array, Float, PyTree

from continual_learning_2.types import Histogram, LogDict


def log(logs: dict, step: int) -> None:
    for key, value in logs.items():
        if isinstance(value, Histogram):
            logs[key] = wandb.Histogram(value.data, np_histogram=value.np_histogram)  # pyright: ignore[reportArgumentType]
    wandb.log(logs, step=step)


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def pytree_histogram(pytree: PyTree, bins: int = 64) -> dict[str, Histogram]:
    flat_dict = flax.traverse_util.flatten_dict(pytree, sep="/")
    ret = {}
    for k, v in flat_dict.items():
        if isinstance(v, tuple):  # For activations
            v = v[0]
        ret[k] = Histogram(np_histogram=jnp.histogram(v, bins=bins))  # pyright: ignore[reportArgumentType]
    return ret


def get_logs(
    name: str,
    data: Float[npt.NDArray | Array, "..."],
    axis: int | None = None,
    hist: bool = True,
    std: bool = True,
) -> "LogDict":
    ret: "LogDict" = {
        f"{name}_mean": jnp.mean(data, axis=axis),
        f"{name}_min": jnp.min(data, axis=axis),
        f"{name}_max": jnp.max(data, axis=axis),
    }
    if std:
        ret[f"{name}_std"] = jnp.std(data, axis=axis)
    if hist:
        ret[f"{name}"] = Histogram(data.reshape(-1))

    return ret


def explained_variance(
    y_pred: Float[npt.NDArray, " total_num_steps"],
    y_true: Float[npt.NDArray, " total_num_steps"],
) -> float:
    # From SB3 https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py#L50
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
