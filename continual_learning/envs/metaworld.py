"""MetaWorld MT10 environment wrapper for multi-task RL benchmarking.

This module provides Gym-based wrappers for MetaWorld environments,
supporting the MT10 multi-task benchmark for SAC/BRO experiments.
"""

from typing import Generator

import jax
import jax.numpy as jnp
import numpy as np

from continual_learning.configs.envs import EnvConfig
from continual_learning.envs.base import (
    Agent,
    ContinualLearningEnv,
    Timestep,
    VectorEnv,
)
from continual_learning.types import Action, Observation


class MetaWorldVectorEnv(VectorEnv):
    """Wrapper for a single MetaWorld task with vectorized environments.

    This wraps MetaWorld tasks using gymnasium's vector environment interface,
    converting between numpy arrays (Gym) and JAX arrays (training).
    """

    def __init__(
        self,
        task_name: str,
        task_cls: type,
        tasks: list,
        num_envs: int,
        seed: int,
    ):
        """Initialize MetaWorld vector environment.

        Args:
            task_name: Name of the task
            task_cls: MetaWorld task class
            tasks: List of task instances with different goals
            num_envs: Number of parallel environments
            seed: Random seed
        """
        self.task_name = task_name
        self.num_envs = num_envs
        self.seed = seed

        # Create environments
        self._envs = []
        for i in range(num_envs):
            env = task_cls()
            # Set task (goal) - cycle through available tasks
            task_idx = i % len(tasks)
            env.set_task(tasks[task_idx])
            env.reset(seed=seed + i)
            self._envs.append(env)

        # Get observation and action specs from first env
        self._obs_dim = self._envs[0].observation_space.shape[0]
        self._action_dim = self._envs[0].action_space.shape[0]

        self._current_obs: np.ndarray | None = None

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def init(self) -> Observation:
        """Reset all environments and return initial observations."""
        obs_list = []
        for i, env in enumerate(self._envs):
            obs, _ = env.reset(seed=self.seed + i)
            obs_list.append(obs)

        self._current_obs = np.stack(obs_list, axis=0)
        return jnp.array(self._current_obs)

    def step(self, action: Action) -> Timestep:
        """Step all environments with given actions.

        Args:
            action: Actions of shape (num_envs, action_dim)

        Returns:
            Timestep with next observations, rewards, done flags, and info
        """
        # Convert JAX array to numpy
        actions_np = np.asarray(action)

        next_obs_list = []
        rewards_list = []
        terminated_list = []
        truncated_list = []
        infos = {"success": []}

        for i, env in enumerate(self._envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])

            # Auto-reset on done
            if terminated or truncated:
                obs, _ = env.reset()

            next_obs_list.append(obs)
            rewards_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos["success"].append(info.get("success", False))

        # Stack and convert to JAX arrays
        next_obs = jnp.array(np.stack(next_obs_list, axis=0))
        rewards = jnp.array(np.array(rewards_list)[:, None])  # (num_envs, 1)
        terminated = jnp.array(np.array(terminated_list)[:, None])
        truncated = jnp.array(np.array(truncated_list)[:, None])

        self._current_obs = np.stack(next_obs_list, axis=0)

        return Timestep(
            next_observation=next_obs,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            info=infos,
        )

    def save(self) -> dict:
        """Save environment state (not fully supported for Gym envs)."""
        return {"current_obs": self._current_obs, "task_name": self.task_name}

    def load(self, checkpoint: dict):
        """Load environment state (limited support)."""
        self._current_obs = checkpoint.get("current_obs")


class MetaWorldMT10Benchmark(ContinualLearningEnv):
    """MetaWorld MT10 multi-task benchmark.

    MT10 contains 10 manipulation tasks that test multi-task learning capabilities.
    Each task has different goals and requires different manipulation strategies.
    """

    def __init__(self, seed: int, config: EnvConfig):
        """Initialize MT10 benchmark.

        Args:
            seed: Random seed
            config: Environment configuration
        """
        try:
            import metaworld
        except ImportError:
            raise ImportError(
                "MetaWorld is not installed. Install with: pip install metaworld"
            )

        self._seed = seed
        self._config = config
        self._num_envs = config.num_envs

        # Initialize MT10 benchmark
        self._mt10 = metaworld.MT10(seed=seed)
        self._task_names = list(self._mt10.train_classes.keys())

        # Get specs from first task
        first_task_cls = self._mt10.train_classes[self._task_names[0]]
        first_env = first_task_cls()
        first_env.set_task(self._mt10.train_tasks[0])

        self._obs_dim = first_env.observation_space.shape[0]
        self._action_dim = first_env.action_space.shape[0]

        self._current_task_idx = 0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_tasks(self) -> int:
        return len(self._task_names)

    @property
    def task_names(self) -> list[str]:
        return self._task_names

    @property
    def tasks(self) -> Generator[VectorEnv, None, None]:
        """Yield vector environments for each task."""
        for task_name in self._task_names:
            task_cls = self._mt10.train_classes[task_name]  # pyright: ignore[reportOptionalSubscript]
            # Get all task instances (different goals) for this task class
            task_instances = [
                t for t in self._mt10.train_tasks  # pyright: ignore[reportOptionalIterable]
                if t.env_name == task_name
            ]

            yield MetaWorldVectorEnv(
                task_name=task_name,
                task_cls=task_cls,
                tasks=task_instances,
                num_envs=self._num_envs,
                seed=self._seed,
            )
            self._current_task_idx += 1

    @property
    def observation_spec(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=(self._num_envs, self._obs_dim),
            dtype=jnp.float32,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def evaluate(self, agent: Agent, forgetting: bool = False) -> dict[str, float] | None:
        """Evaluate agent on all tasks.

        Args:
            agent: Agent to evaluate
            forgetting: Whether to measure forgetting on previous tasks

        Returns:
            Dictionary of evaluation metrics
        """
        # TODO: Implement evaluation across tasks
        return None

    def save(self) -> dict:
        """Save benchmark state."""
        return {
            "current_task_idx": self._current_task_idx,
            "seed": self._seed,
        }

    def load(self, checkpoint: dict, envs_checkpoint: VectorEnv):
        """Load benchmark state."""
        self._current_task_idx = checkpoint.get("current_task_idx", 0)


class MetaWorldSingleTaskEnv(VectorEnv):
    """Wrapper for a single MetaWorld task with optional async parallelism."""

    def __init__(
        self,
        task_name: str,
        num_envs: int = 1,
        seed: int = 0,
        async_envs: bool = True,
    ):
        """Initialize single task environment.

        Args:
            task_name: Name of the MetaWorld task
            num_envs: Number of parallel environments
            seed: Random seed
            async_envs: If True, use multiprocessing for parallel env stepping
        """
        try:
            import metaworld
        except ImportError:
            raise ImportError(
                "MetaWorld is not installed. Install with: pip install metaworld"
            )

        self.task_name = task_name
        self.num_envs = num_envs
        self.seed = seed
        self._async = async_envs and num_envs > 1

        if self._async:
            # Use gymnasium's AsyncVectorEnv for true parallelism
            # Use "spawn" context to avoid JAX fork() deadlock warning
            import gymnasium
            from functools import partial

            # Import factory from isolated module that sets JAX_PLATFORMS=cpu
            # This prevents GPU memory conflicts in spawned worker processes
            from continual_learning.envs._metaworld_worker import make_metaworld_env

            env_fns = [
                partial(make_metaworld_env, task_name, seed, i)
                for i in range(num_envs)
            ]
            # "spawn" is safer with JAX (avoids fork() with threads)
            self._vec_env = gymnasium.vector.AsyncVectorEnv(env_fns, context="spawn")
            self._obs_dim = self._vec_env.single_observation_space.shape[0]
            self._action_dim = self._vec_env.single_action_space.shape[0]
        else:
            # Original sequential implementation
            ml1 = metaworld.ML1(task_name, seed=seed)
            self._task_cls = ml1.train_classes[task_name]
            self._tasks = ml1.train_tasks

            self._envs = []
            for i in range(num_envs):
                env = self._task_cls()
                task_idx = i % len(self._tasks)
                env.set_task(self._tasks[task_idx])
                env.reset(seed=seed + i)
                self._envs.append(env)

            self._obs_dim = self._envs[0].observation_space.shape[0]
            self._action_dim = self._envs[0].action_space.shape[0]

        self._current_obs: np.ndarray | None = None

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def init(self) -> Observation:
        """Reset all environments."""
        if self._async:
            obs, _ = self._vec_env.reset(seed=self.seed)
            self._current_obs = obs
        else:
            obs_list = []
            for i, env in enumerate(self._envs):
                obs, _ = env.reset(seed=self.seed + i)
                obs_list.append(obs)
            self._current_obs = np.stack(obs_list, axis=0)

        return jnp.array(self._current_obs)

    def step(self, action: Action) -> Timestep:
        """Step environments."""
        actions_np = np.asarray(action)

        if self._async:
            # AsyncVectorEnv handles auto-reset internally
            next_obs, rewards, terminated, truncated, infos = self._vec_env.step(
                actions_np
            )
            self._current_obs = next_obs

            # Extract success from info dict (gymnasium vectorizes this differently)
            if "success" in infos:
                success_list = list(infos["success"])
            elif "final_info" in infos:
                # On auto-reset, success is in final_info
                success_list = [
                    fi.get("success", False) if fi else False
                    for fi in infos.get("final_info", [{}] * self.num_envs)
                ]
            else:
                success_list = [False] * self.num_envs

            return Timestep(
                next_observation=jnp.array(next_obs),
                reward=jnp.array(rewards[:, None]),
                terminated=jnp.array(terminated[:, None]),
                truncated=jnp.array(truncated[:, None]),
                info={"success": success_list},
            )
        else:
            # Original sequential implementation
            next_obs_list = []
            rewards_list = []
            terminated_list = []
            truncated_list = []
            infos = {"success": []}

            for i, env in enumerate(self._envs):
                obs, reward, terminated, truncated, info = env.step(actions_np[i])

                if terminated or truncated:
                    obs, _ = env.reset()

                next_obs_list.append(obs)
                rewards_list.append(reward)
                terminated_list.append(terminated)
                truncated_list.append(truncated)
                infos["success"].append(info.get("success", False))

            next_obs = jnp.array(np.stack(next_obs_list, axis=0))
            rewards = jnp.array(np.array(rewards_list)[:, None])
            terminated = jnp.array(np.array(terminated_list)[:, None])
            truncated = jnp.array(np.array(truncated_list)[:, None])

            self._current_obs = np.stack(next_obs_list, axis=0)

            return Timestep(
                next_observation=next_obs,
                reward=rewards,
                terminated=terminated,
                truncated=truncated,
                info=infos,
            )

    def save(self) -> dict:
        return {"current_obs": self._current_obs}

    def load(self, checkpoint: dict):
        self._current_obs = checkpoint.get("current_obs")

    def close(self):
        """Clean up environments."""
        if self._async:
            self._vec_env.close()
        else:
            for env in self._envs:
                if hasattr(env, "close"):
                    env.close()
