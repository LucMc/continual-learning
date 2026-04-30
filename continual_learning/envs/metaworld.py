"""MetaWorld MT10 environment wrapper for multi-task RL benchmarking. """

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
        task_idx: int | None = None,
        num_tasks: int | None = None,
    ):
        """Initialize MetaWorld vector environment.

        Args:
            task_name: Name of the task
            task_cls: MetaWorld task class
            tasks: List of task instances with different goals
            num_envs: Number of parallel environments
            seed: Random seed
            task_idx: Index of this task (0-indexed) for one-hot encoding
            num_tasks: Total number of tasks for one-hot encoding
        """
        self.task_name = task_name
        self.num_envs = num_envs
        self.seed = seed
        self._task_idx = task_idx
        self._num_tasks = num_tasks

        # Create environments
        self._envs = []
        for i in range(num_envs):
            env = task_cls()
            # Set task (goal) - cycle through available tasks
            goal_idx = i % len(tasks)
            env.set_task(tasks[goal_idx])
            env.reset(seed=seed + i)
            self._envs.append(env)

        # Get observation and action specs from first env
        raw_obs_dim = self._envs[0].observation_space.shape[0]
        self._raw_obs_dim = raw_obs_dim
        self._obs_dim = raw_obs_dim + num_tasks if num_tasks is not None else raw_obs_dim
        self._action_dim = self._envs[0].action_space.shape[0]

        # Precompute one-hot encoding for this task
        if task_idx is not None and num_tasks is not None:
            self._one_hot = np.zeros(num_tasks, dtype=np.float32)
            self._one_hot[task_idx] = 1.0
        else:
            self._one_hot = None

        self._current_obs: np.ndarray | None = None

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _append_task_id(self, obs: np.ndarray) -> np.ndarray:
        """Append one-hot task ID to observations if configured."""
        if self._one_hot is None:
            return obs
        # obs shape: (num_envs, raw_obs_dim), one_hot shape: (num_tasks,)
        one_hot_batch = np.broadcast_to(self._one_hot, (obs.shape[0], len(self._one_hot)))
        return np.concatenate([obs, one_hot_batch], axis=-1)

    def init(self) -> Observation:
        """Reset all environments and return initial observations."""
        obs_list = []
        for i, env in enumerate(self._envs):
            obs, _ = env.reset(seed=self.seed + i)
            obs_list.append(obs)

        self._current_obs = np.stack(obs_list, axis=0)
        return jnp.array(self._append_task_id(self._current_obs))

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
        raw_next_obs = np.stack(next_obs_list, axis=0)
        next_obs = jnp.array(self._append_task_id(raw_next_obs))
        rewards = jnp.array(np.array(rewards_list)[:, None])  # (num_envs, 1)
        terminated = jnp.array(np.array(terminated_list)[:, None])
        truncated = jnp.array(np.array(truncated_list)[:, None])

        self._current_obs = raw_next_obs

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

    def close(self):
        """Clean up MuJoCo environments."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


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

        raw_obs_dim = first_env.observation_space.shape[0]
        # Include one-hot task ID in observation
        self._obs_dim = raw_obs_dim + len(self._task_names)
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
        for task_idx, task_name in enumerate(self._task_names):
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
                task_idx=task_idx,
                num_tasks=len(self._task_names),
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
    """Wrapper for a single MetaWorld task with vectorized environments."""

    def __init__(
        self,
        task_name: str,
        num_envs: int = 1,
        seed: int = 0,
        apply_delay_wrapper: bool = False,
        max_obs_delay: int = 0,
        max_act_delay: int = 0,
        delay_mode: str = "fixed",
        resample_every: int | None = None,
        interval_emb_type: str | None = None,
        delay_emb_type: str | None = "one_hot",
    ):
        """Initialize single task environment.

        Args:
            task_name: Name of the MetaWorld task
            num_envs: Number of parallel environments
            seed: Random seed
            apply_delay_wrapper: If True, wrap each env with
                ContinualRandomIntervalDelayWrapper.
            max_obs_delay: Max observation delay (inclusive). 0 = always 0 delay.
            max_act_delay: Max action delay (inclusive). 0 = always 0 delay.
            delay_mode: "fixed", "multi-task", or "continual".
            resample_every: For "continual" mode, step period between resamples.
            interval_emb_type: Delay-interval embedding type (None, "two_hot",
                "float", "scalar"). Must be None when max_*_delay == 0.
            delay_emb_type: Delay embedding type ("one_hot", "float", "scalar").
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

        ml1 = metaworld.ML1(task_name, seed=seed)
        self._task_cls = ml1.train_classes[task_name]
        self._tasks = ml1.train_tasks

        self._envs = []
        for i in range(num_envs):
            env = self._task_cls()
            task_idx = i % len(self._tasks)
            env.set_task(self._tasks[task_idx])
            env.reset(seed=seed + i)
            if apply_delay_wrapper:
                from continual_learning.utils.continual_delay_wrapper import (
                    ContinualRandomIntervalDelayWrapper,
                )
                env = ContinualRandomIntervalDelayWrapper(
                    env,
                    obs_delay_range=range(0, max_obs_delay + 1),
                    act_delay_range=range(0, max_act_delay + 1),
                    mode=delay_mode,
                    resample_every=resample_every,
                    interval_emb_type=interval_emb_type,
                    delay_emb_type=delay_emb_type,
                    output="standard",
                    give_kappa=False,
                )
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
        obs_list = []
        for i, env in enumerate(self._envs):
            obs, _ = env.reset(seed=self.seed + i)
            obs_list.append(obs)
        self._current_obs = np.stack(obs_list, axis=0)
        return jnp.array(self._current_obs)

    def step(self, action: Action) -> Timestep:
        """Step environments."""
        actions_np = np.asarray(action)

        next_obs_list = []
        rewards_list = []
        terminated_list = []
        truncated_list = []
        infos: dict = {
            "success": [],
            "realised_obs_delay": [],
            "realised_act_delay": [],
        }

        for i, env in enumerate(self._envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])

            if terminated or truncated:
                obs, _ = env.reset()

            next_obs_list.append(obs)
            rewards_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos["success"].append(info.get("success", False))

            if "realised_obs_delay" in info:
                infos["realised_obs_delay"].append(info["realised_obs_delay"])
                infos["realised_act_delay"].append(info["realised_act_delay"])
                # Intervals are diagnostic — take env 0 as representative.
                if i == 0:
                    infos["current_obs_interval"] = info["current_obs_interval"]
                    infos["current_act_interval"] = info["current_act_interval"]
                    infos["overall_obs_interval"] = info["overall_obs_interval"]
                    infos["overall_act_interval"] = info["overall_act_interval"]

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
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()

