from __future__ import annotations

from typing import Any, TypeVar

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree
import numpy.typing as npt
import wandb

from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.types import Histogram, LayerActivationsDict, LogDict


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def _to_numpy(a):  # JAX/NumPy -> NumPy
    return np.asarray(a)

def _to_scalar(x: Any) -> float | int:
    arr = _to_numpy(x)
    return arr.item() if arr.shape == () else float(arr.mean())

def _safe_hist_from_sequence(seq, default_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """NumPy 2.x–safe histogram for constant arrays, NaNs/Infs filtered."""
    arr = _to_numpy(seq).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0], dtype=np.int64), np.array([0.0, 1.0], dtype=np.float64)
    
    uniq = np.unique(arr)
    
    # Handle case where all values are identical
    if uniq.size == 1:
        # Create a histogram with 1 bin centered around the single value
        single_val = uniq[0]
        # Create edges that encompass the single value
        edges = np.array([single_val - 0.5, single_val + 0.5], dtype=np.float64)
        counts = np.array([arr.size], dtype=np.int64)
        return counts, edges
    
    num_bins = int(min(default_bins, max(1, uniq.size)))
    hist, edges = np.histogram(arr, bins=num_bins)
    return hist.astype(np.int64), edges.astype(np.float64)

def _as_wandb_hist(value: Histogram) -> wandb.Histogram | None:
    """Prefer precomputed hist; else compute a safe one from data."""
    np_hist = getattr(value, "np_histogram", None)
    if isinstance(np_hist, tuple) and len(np_hist) == 2:
        counts, edges = np_hist
        return wandb.Histogram(
            np_histogram=(_to_numpy(counts).astype(np.int64), _to_numpy(edges).astype(np.float64))
        )
    data = getattr(value, "data", None)
    if data is not None:
        counts, edges = _safe_hist_from_sequence(data)
        return wandb.Histogram(np_histogram=(counts, edges))
    return None


def get_logs(
    name: str,
    data: Float[npt.NDArray | Array, "..."],
    hist: bool = True,
    std: bool = True,
) -> LogDict:
    ret: LogDict = {
        f"{name}_mean": jnp.mean(data),
        f"{name}_min": jnp.min(data),
        f"{name}_max": jnp.max(data),
    }
    if std:
        ret[f"{name}_std"] = jnp.std(data)
    if hist:
        flat = data.reshape(-1)
        # Leave np_histogram=None -> logger will build a safe histogram
        ret[f"{name}"] = Histogram(data=flat, total_events=flat.shape[0])  # pyright: ignore[reportArgumentType]
    return ret

def pytree_histogram(pytree: PyTree[Array], bins: int = 64) -> dict[str, Histogram]:
    """Per‑leaf histograms; jnp.histogram keeps this usable inside JIT."""
    flat = flax.traverse_util.flatten_dict(pytree, sep="/")
    out: dict[str, Histogram] = {}
    for k, v in flat.items():
        if isinstance(v, tuple):  # activations sometimes come as (value, ...)
            v = v[0]
        counts, edges = jnp.histogram(v, bins=bins)
        out[k] = Histogram(total_events=v.size, np_histogram=(counts, edges))
    return out

def explained_variance(
    y_pred: Float[npt.NDArray | Array, " total_num_steps"],
    y_true: Float[npt.NDArray | Array, " total_num_steps"],
) -> Float[Array, ""]:
    var_y = jnp.var(y_true)
    return jnp.where(var_y == 0, jnp.nan, 1 - jnp.var(y_true - y_pred) / var_y)

def get_dormant_neuron_logs(layer_activations: LayerActivationsDict, threshold: float = 0.1) -> LogDict:
    logs: LogDict = {}
    total_dead, total = 0, 0
    for key, act in layer_activations.items():
        chex_rank = 2  # (batch, dim)
        # if you want, assert rank here with chex.assert_rank(act, chex_rank)
        scores = jnp.mean(jnp.abs(act), axis=0)
        scores = scores / (jnp.mean(scores) + 1e-6)
        dead = jnp.sum(scores <= threshold)
        logs[f"{key}_ratio"] = (dead / scores.shape[0]) * 100
        logs[f"{key}_count"] = dead
        total_dead += dead
        total += scores.shape[0]
    logs["total_ratio"] = jnp.array((total_dead / total) * 100)
    logs["total_count"] = total_dead
    return logs

def get_linearised_neuron_logs(layer_preactivations: LayerActivationsDict, threshold: float = 0.9) -> LogDict:
    logs: LogDict = {}
    total_lin, total = 0, 0
    for key, act in layer_preactivations.items():
        pct = jnp.mean(act > 0, axis=0)
        lin = jnp.sum(pct >= threshold)
        logs[f"{key}_ratio"] = (lin / pct.shape[0]) * 100
        logs[f"{key}_count"] = lin
        total_lin += lin
        total += pct.shape[0]
    logs["total_ratio"] = jnp.array((total_lin / total) * 100)
    logs["total_count"] = total_lin
    return logs

def compute_srank(feature_matrix: Float[Array, "num_features feature_dim"], delta: float = 0.01) -> Float[Array, ""]:
    s = jnp.linalg.svd(feature_matrix.astype(jnp.float32), compute_uv=False)
    ratios = jnp.cumsum(s) / jnp.sum(s)
    return jnp.argmax(ratios >= (1.0 - delta)) + 1


def average_histograms(histograms: list[Histogram]) -> Histogram:
    """Average a list of Histogram objects by resampling onto a common grid."""
    if not histograms:
        return Histogram(total_events=0, np_histogram=(np.array([0], dtype=np.int64),
                                                       np.array([0.0, 1.0], dtype=np.float64)))
    data = [(h.np_histogram[0], h.np_histogram[1], h.total_events)
            for h in histograms if h.np_histogram is not None]
    if not data:
        return Histogram(total_events=0, np_histogram=(np.array([0], dtype=np.int64),
                                                       np.array([0.0, 1.0], dtype=np.float64)))

    counts_list = [_to_numpy(c) for (c, _, _) in data]
    edges_list  = [_to_numpy(e) for (_, e, _) in data]
    weights     = [float(np.asarray(t).sum()) for (_, _, t) in data]

    global_min = min(e[0] for e in edges_list)
    global_max = max(e[-1] for e in edges_list)
    max_edges  = max(len(e) for e in edges_list)

    # Match prior behavior: increase resolution relative to the largest source
    target_edges   = np.linspace(global_min, global_max, 2 * max_edges - 1)
    target_centers = (target_edges[:-1] + target_edges[1:]) / 2

    resampled = []
    for counts, edges in zip(counts_list, edges_list):
        centers = (edges[:-1] + edges[1:]) / 2
        resampled.append(np.interp(target_centers, centers, counts))

    avg_counts = np.average(np.stack(resampled, axis=0), axis=0, weights=np.asarray(weights))
    return Histogram(total_events=np.sum(weights),
                     np_histogram=(avg_counts.astype(np.float64), target_edges.astype(np.float64)))

def average_histograms_concatenated(histograms: Histogram) -> Histogram:
    """Average a time/scan‑stacked Histogram (counts/edges stacked on leading dims)."""
    assert histograms.np_histogram is not None
    counts, edges = histograms.np_histogram  # shape: [..., BINS], [..., BINS]
    c_flat = counts.reshape(-1, counts.shape[-1]).astype(jnp.float32)
    e_flat = edges.reshape(-1, edges.shape[-1]).astype(jnp.float32)

    global_min = jnp.min(e_flat[:, 0])
    global_max = jnp.max(e_flat[:, -1])
    max_edges  = e_flat.shape[-1]

    target_edges   = jnp.linspace(global_min, global_max, 2 * max_edges - 1)
    target_centers = (target_edges[:-1] + target_edges[1:]) / 2

    @jax.vmap
    def _resample(c, e):
        centers = (e[:-1] + e[1:]) / 2
        return jnp.interp(target_centers, centers, c)

    resampled = _resample(c_flat, e_flat)  # [N, NEW_BINS]
    weights   = jnp.reshape(histograms.total_events, -1).astype(jnp.float32)  # [N]
    avg_counts = jnp.average(resampled, axis=0, weights=weights)

    return Histogram(total_events=jnp.sum(histograms.total_events),  # pyright: ignore[reportArgumentType]
                     np_histogram=(avg_counts, target_edges))


MetricsType = TypeVar("MetricsType", bound=LogDict | dict[str, float])

def accumulate_metrics(metrics: list[MetricsType]) -> MetricsType:
    ret: dict[str, Any] = {}
    first = metrics[0]
    for k in first:
        v0 = first[k]
        if isinstance(v0, Histogram):
            ret[k] = average_histograms([m[k] for m in metrics])  # type: ignore[index]
        else:
            ret[k] = float(np.mean([_to_scalar(m[k]) for m in metrics]))  # type: ignore[index]
    return ret  # pyright: ignore[reportReturnType]

def accumulate_concatenated_metrics(metrics: LogDict) -> LogDict:
    ret: LogDict = {}
    for k, v in metrics.items():
        if isinstance(v, Histogram):
            ret[k] = average_histograms_concatenated(v)
        else:
            ret[k] = jnp.mean(v)
    return ret  # pyright: ignore[reportReturnType]


def log(logs: dict, step: int) -> None:
    """Legacy helper used in some codepaths."""
    cleaned: dict[str, Any] = {}
    for key, value in logs.items():
        if isinstance(value, Histogram):
            h = _as_wandb_hist(value)
            if h is not None:
                cleaned[key] = h
            else:
                # Optional: summarize unusable histogram data as scalars
                data = getattr(value, "data", None)
                if data is not None:
                    arr = _to_numpy(data)
                    if arr.size:
                        arr = arr[np.isfinite(arr)]
                        if arr.size:
                            cleaned[key + "/mean"] = float(arr.mean())
                            cleaned[key + "/std"]  = float(arr.std())
                            cleaned[key + "/min"]  = float(arr.min())
                            cleaned[key + "/max"]  = float(arr.max())
        else:
            cleaned[key] = _to_scalar(value) if hasattr(value, "shape") else value
    wandb.log(cleaned, step=step)


class Logger:
    cfg: LoggingConfig

    def __init__(self, logging_config: LoggingConfig, run_config: dict):
        self.cfg = logging_config
        self.buffer: list[LogDict] = []
        wandb.init(
            name=self.cfg.run_name,
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            config=run_config,
            mode=self.cfg.wandb_mode,
            group=self.cfg.group,
            resume="allow",
        )

    def accumulate(self, logs: LogDict):
        self.buffer.append(logs)

    def _log(self, logs: LogDict | dict[str, float], step: int):
        cleaned: dict[str, Any] = {}
        for key, value in logs.items():
            if isinstance(value, Histogram):
                h = _as_wandb_hist(value)
                if h is not None:
                    cleaned[key] = h
            else:
                cleaned[key] = _to_scalar(value) if hasattr(value, "shape") else value
        wandb.log(cleaned, step=step)

    def push(self, step: int) -> None:
        if self.buffer:
            logs = accumulate_metrics(self.buffer)
            self.buffer.clear()
            self._log(logs, step)

    def log(self, logs: LogDict | dict[str, float], step: int):
        self._log(logs, step)

    def close(self):
        try:
            wandb.finish()
        except Exception:
            pass

