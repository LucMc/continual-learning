"""Isolated MetaWorld environment factory for subprocess workers.

This module is intentionally kept free of JAX imports so it can be safely
imported in spawned subprocesses without triggering GPU memory allocation.
"""

import os

# Force CPU-only JAX in worker processes (set before any JAX imports)
os.environ["JAX_PLATFORMS"] = "cpu"


def make_metaworld_env(task_name: str, seed: int, task_idx: int):
    """Factory function for creating MetaWorld envs in subprocesses.

    This function is called in spawned subprocess workers. JAX is forced to
    CPU-only mode to prevent GPU memory conflicts with the main process.
    """
    import metaworld

    ml1 = metaworld.ML1(task_name, seed=seed)
    task_cls = ml1.train_classes[task_name]
    tasks = ml1.train_tasks

    env = task_cls()
    env.set_task(tasks[task_idx % len(tasks)])
    return env
