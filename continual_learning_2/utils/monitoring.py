from typing import Any

import chex
import flax.struct
import flax.traverse_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import wandb
from jaxtyping import Array, Float, PyTree

from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.types import Histogram, Intermediates, LayerActivationsDict, LogDict


def log(logs: dict, step: int) -> None:
    # TODO: probably remove once PPO has been migrated
    for key, value in logs.items():
        if isinstance(value, Histogram):
            logs[key] = wandb.Histogram(value.data, np_histogram=value.np_histogram)  # pyright: ignore[reportArgumentType]
    wandb.log(logs, step=step)


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def pytree_histogram(pytree: PyTree[Array], bins: int = 64) -> dict[str, Histogram]:
    flat_dict = flax.traverse_util.flatten_dict(pytree, sep="/")
    ret = {}
    for k, v in flat_dict.items():
        if isinstance(v, tuple):  # For activations
            v = v[0]
        assert isinstance(v, Array) or isinstance(v, np.ndarray)
        ret[k] = Histogram(
            total_events=v.reshape(-1).shape[0], np_histogram=jnp.histogram(v, bins=bins)
        )  # pyright: ignore[reportArgumentType]
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
        data = data.reshape(-1)
        ret[f"{name}"] = Histogram(data=data, total_events=data.shape[0])

    return ret


def explained_variance(
    y_pred: Float[npt.NDArray, " total_num_steps"],
    y_true: Float[npt.NDArray, " total_num_steps"],
) -> float:
    # From SB3 https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py#L50
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def extract_activations(network_dict: Intermediates) -> LayerActivationsDict:
    def recursive_extract(
        d: Intermediates, current_path: list[str] = []
    ) -> LayerActivationsDict:
        activations = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    sub_activations = recursive_extract(v, current_path + [k])
                    activations.update(sub_activations)
                else:
                    assert isinstance(v, tuple)
                    # HACK: assuming every module only has 1 output
                    activations[k] = v[0]
        return activations

    return recursive_extract(network_dict)


def get_dormant_neuron_logs(
    layer_activations: LayerActivationsDict, dormant_neuron_threshold: float = 0.1
) -> LogDict:
    """Compute the dormant neuron ratio per layer using Equation 1 from
    "The Dormant Neuron Phenomenon in Deep Reinforcement Learning" (Sokar et al., 2023; https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf).

    Adapted from https://github.com/google/dopamine/blob/master/dopamine/labs/redo/tfagents/sac_train_eval.py#L563"""

    all_layers_score: LayerActivationsDict = {}
    dormant_neurons = {}  # To store both mask and count for each layer

    for act_key, act_value in layer_activations.items():
        chex.assert_rank(act_value, 2)
        neurons_score = jnp.mean(jnp.abs(act_value), axis=0)
        neurons_score = neurons_score / (jnp.mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score

        mask = jnp.where(
            neurons_score <= dormant_neuron_threshold,
            jnp.ones_like(neurons_score, dtype=jnp.int32),
            jnp.zeros_like(neurons_score, dtype=jnp.int32),
        )
        num_dormant_neurons = jnp.sum(mask)

        dormant_neurons[act_key] = {"mask": mask, "count": num_dormant_neurons}

    logs = {}

    total_dead_neurons = 0
    total_hidden_count = 0
    for layer_name, layer_score in all_layers_score.items():
        num_dormant_neurons = dormant_neurons[layer_name]["count"]
        logs[f"{layer_name}_ratio"] = (num_dormant_neurons / layer_score.shape[0]) * 100
        logs[f"{layer_name}_count"] = num_dormant_neurons
        total_dead_neurons += num_dormant_neurons
        total_hidden_count += layer_score.shape[0]

    logs.update(
        {
            "total_ratio": jnp.array((total_dead_neurons / total_hidden_count) * 100),
            "total_count": total_dead_neurons,
        }
    )

    return logs


def average_histograms(histograms: list[Histogram]) -> Histogram:
    data = [  # counts, edges, total_events
        (h.np_histogram[0], h.np_histogram[1], h.total_events)
        for h in histograms
        if h.np_histogram is not None
    ]

    global_min = min([h[1][0] for h in data])
    global_max = max([h[1][-1] for h in data])
    max_edges = max([len(h[1]) for h in data])

    target_bin_edges = np.linspace(global_min, global_max, 2 * max_edges - 1)
    target_bin_centers = (target_bin_edges[:-1] + target_bin_edges[1:]) / 2

    resampled_counts_list, weights = [], []
    for counts, bin_edges, total_events in data:
        original_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        resampled_counts = np.interp(target_bin_centers, original_bin_centers, counts)
        resampled_counts_list.append(resampled_counts)
        weights.append(total_events)

    averaged_counts = np.average(
        np.array(resampled_counts_list), axis=0, weights=np.array(weights)
    )

    return Histogram(
        total_events=np.sum(weights),
        np_histogram=(averaged_counts, target_bin_edges),
    )


def accumulate_metrics(metrics: list[LogDict]) -> LogDict:
    ret = {}
    for k in metrics[0]:
        if not isinstance(metrics[0][k], Histogram):
            ret[k] = float(np.mean([m[k] for m in metrics]))  # pyright: ignore[reportArgumentType,reportCallIssue]
        else:
            ret[k] = average_histograms([m[k] for m in metrics])  # pyright: ignore[reportArgumentType,reportCallIssue]

    return ret


class Logger:
    cfg: LoggingConfig

    def __init__(self, logging_config: LoggingConfig, run_config: dict):
        self.cfg = logging_config
        self.buffer = []
        wandb.init(
            name=self.cfg.run_name,
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            config=run_config,
            resume="allow",
        )

    def get_distribution_logs(
        self,
        name: str,
        data: Float[npt.NDArray | Array, "..."],
        axis: int | None = None,
        log_histogram: bool = True,
        log_std: bool = True,
    ) -> "LogDict":
        ret: "LogDict" = {
            f"{name}_mean": jnp.mean(data, axis=axis),
            f"{name}_min": jnp.min(data, axis=axis),
            f"{name}_max": jnp.max(data, axis=axis),
        }
        if log_std:
            ret[f"{name}_std"] = jnp.std(data, axis=axis)
        if log_histogram:
            data = data.reshape(-1)
            ret[f"{name}"] = Histogram(data=data, total_events=data.shape[0])

        return ret

    def accumulate(self, logs: LogDict):
        self.buffer.append(logs)

    def _log(self, logs: LogDict, step: int):
        for key, value in logs.items():
            if isinstance(value, Histogram):
                logs[key] = wandb.Histogram(value.data, np_histogram=value.np_histogram)  # pyright: ignore[reportArgumentType]
        wandb.log(logs, step=step)

    def push(self, step: int) -> None:
        if len(self.buffer) == 0:
            return

        logs = accumulate_metrics(self.buffer)
        self.buffer.clear()
        self._log(logs, step)

    def log(self, logs: LogDict, step: int):
        self._log(logs, step)

    def close(self):
        wandb.finish()

    def __del__(self):
        self.close()
