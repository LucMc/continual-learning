import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from functools import cache

@cache
def get_api(): return wandb.Api()

def fetch_runs(entity: str, project: str, filters: Optional[Dict] = None) -> List[Any]:
    return list(get_api().runs(f"{entity}/{project}", filters=filters))

def get_run_data(run: Any, metrics: List[str]) -> Dict[str, np.ndarray]:
    history = run.history(keys=metrics)
    return {k: history[k].dropna().values for k in metrics if k in history.columns}

def iqm(values: np.ndarray, axis: int = 0) -> np.ndarray:
    q25, q75 = np.percentile(values, [25, 75], axis=axis)
    if axis == 0:
        mask = (values >= q25) & (values <= q75)
        return np.mean(values[mask], axis=axis)
    result = np.zeros(values.shape[1-axis])
    for i in range(values.shape[1-axis]):
        slice_vals = values[:, i] if axis == 1 else values[i, :]
        mask = (slice_vals >= q25[i]) & (slice_vals <= q75[i])
        result[i] = np.mean(slice_vals[mask]) if np.any(mask) else np.nan
    return result

def fetch_group_data(entity: str, project: str, group_name: str, metrics: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    runs = fetch_runs(entity, project, {"group": group_name})
    return {run.name: get_run_data(run, metrics) for run in runs}

def aggregate_metrics(data: Dict[str, Dict[str, np.ndarray]], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = [run_data[metric] for run_data in data.values() if metric in run_data]
    if not values: return np.array([]), np.array([]), np.array([]), np.array([])

    max_len = max(len(v) for v in values)
    padded = [np.pad(v, (0, max_len - len(v)), 'constant', constant_values=np.nan) for v in values]
    stacked = np.stack(padded)

    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0), np.array([iqm(stacked[:, i]) for i in range(stacked.shape[1])]), stacked

def get_best_runs(entity: str, project: str, metric: str, direction: str = 'max', n: int = 5) -> List[Any]:
    order = f"-summary_metrics.{metric}" if direction == 'max' else f"summary_metrics.{metric}"
    return list(get_api().runs(f"{entity}/{project}", {"$and": [{"summary_metrics": {"$exists": True}}]}, order=order))[:n]

def compare_groups(entity: str, project: str, groups: List[str], metric: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {group: aggregate_metrics(fetch_group_data(entity, project, group, [metric]), metric)[:2] for group in groups}