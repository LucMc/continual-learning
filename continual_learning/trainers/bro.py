"""BRO (Bigger, Regularized, Optimistic) Trainer.

This module implements the BRO algorithm as described in:
Nauman et al. 2024 - "Bigger, Regularized, Optimistic: scaling for compute
and sample-efficient continuous control"

Key components:
- Distributional critic with quantile regression
- Conservative + Optimistic dual actor system
- Learnable temperature, optimism, and regularizer coefficients
- BroNet architecture with LayerNorm and residual connections
- Fixed reset schedule for high replay ratio training

This trainer wraps the BROLearner for continual learning on MetaWorld MT10.
"""

import dataclasses
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.logging import LoggingConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.envs.base import ContinualLearningEnv, VectorEnv
from continual_learning.envs import get_benchmark
from continual_learning.trainers.bro_learner import BROLearner, BROConfig
from continual_learning.types import LogDict, Observation
from continual_learning.utils.monitoring import Logger, prefix_dict
from continual_learning.utils.replay_buffer import ReplayBuffer, ReplayBufferState


def _serialize_config(obj):
    """Convert a config object to a wandb-serializable dict.

    Includes the class name as 'type' and skips non-serializable fields (callables).
    Recurses into nested dataclass/NamedTuple configs.
    """
    if isinstance(obj, dict):
        return {k: _serialize_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not hasattr(obj, '_fields'):
        return [_serialize_config(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {"type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            if callable(val) and not isinstance(val, (int, float, str, bool)):
                d[f.name] = str(val)
            else:
                d[f.name] = _serialize_config(val)
        return d
    if hasattr(obj, '_asdict'):
        d = {"type": type(obj).__name__}
        for k, v in obj._asdict().items():
            if callable(v) and not isinstance(v, (int, float, str, bool)):
                d[k] = str(v)
            else:
                d[k] = _serialize_config(v)
        return d
    return obj


class BROTrainer:
    """BRO Trainer: Full BRO algorithm with dual actors and distributional RL.

    This trainer implements the complete BRO algorithm for sample-efficient
    continuous control with high replay ratios. It supports comparing different
    optimization methods on the MetaWorld benchmark in a continual learning setting.
    """

    def __init__(
        self,
        seed: int,
        bro_config: BROConfig,
        env_cfg: EnvConfig,
        train_cfg: RLTrainingConfig,
        logs_cfg: LoggingConfig,
    ):
        """Initialize BRO trainer.

        Args:
            seed: Random seed
            bro_config: BRO algorithm configuration
            env_cfg: Environment configuration
            train_cfg: Training configuration
            logs_cfg: Logging configuration
        """
        self.key = jax.random.PRNGKey(seed)
        self.cfg = bro_config
        self.train_cfg = train_cfg
        self.seed = seed

        # Initialize logger
        self.logger = Logger(
            logs_cfg,
            run_config={
                "algorithm": _serialize_config(bro_config),
                "benchmark": _serialize_config(env_cfg),
                "training": _serialize_config(train_cfg),
            },
        )

        # Get benchmark/environment
        benchmark = get_benchmark(seed, env_cfg)
        if not isinstance(benchmark, ContinualLearningEnv):
            raise ValueError(
                "BRO trainer requires ContinualLearningEnv (Gym-based), not JittableContinualLearningEnv"
            )
        self.benchmark = benchmark

        # Get dimensions from environment
        self.obs_dim = benchmark.observation_spec.shape[-1]
        self.action_dim = benchmark.action_dim

        # Initialize BRO learner
        self.learner = BROLearner(
            seed=seed,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            cfg=bro_config,
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=bro_config.buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        self.buffer_state = self.replay_buffer.init()

        # Training counters
        self.total_steps = 0
        self.total_episodes = 0
        self.total_gradient_steps = 0

        # Episode tracking
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_episode_reward = np.zeros(env_cfg.num_envs)
        self._current_episode_length = np.zeros(env_cfg.num_envs, dtype=int)

        # Continual learning tracking
        self._completed_task_names: list[str] = []
        self._best_success_rates: dict[str, float] = {}
        self._final_success_rates: dict[str, float] = {}
        self._current_task_name: str | None = None
        self._eval_frequency = 10_000

    def select_action(
        self,
        obs: Observation,
        deterministic: bool = False,
    ) -> jax.Array:
        """Select action from policy.

        Args:
            obs: Current observation
            deterministic: If True, use conservative actor (for evaluation)

        Returns:
            Selected action clipped to [-1, 1]
        """
        if deterministic:
            # Use conservative actor for evaluation
            actions = self.learner.sample_actions(
                obs, temperature=1.0, use_optimistic=False
            )
        else:
            # Use optimistic actor for exploration
            actions = self.learner.sample_actions(
                obs, temperature=1.0, use_optimistic=True
            )

        return jnp.clip(actions, -1.0, 1.0)

    def collect_step(self, envs: VectorEnv, obs: Observation) -> tuple[Observation, dict]:
        """Collect one step of experience.

        Args:
            envs: Vector environment
            obs: Current observation

        Returns:
            Tuple of (next_obs, info_dict)
        """
        # Select action using optimistic actor
        action = self.select_action(obs)

        # Handle NaN
        action = jnp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # Step environment
        timestep = envs.step(action)

        # Store transition in replay buffer
        self.buffer_state = ReplayBuffer.add(
            self.buffer_state,
            obs=obs,
            action=action,
            reward=timestep.reward,
            next_obs=timestep.next_observation,
            done=timestep.terminated | timestep.truncated,
        )

        # Update episode tracking
        rewards_np = np.asarray(timestep.reward).squeeze(-1)
        dones_np = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)

        self._current_episode_reward += rewards_np
        self._current_episode_length += 1

        # Record completed episodes
        for i, done in enumerate(dones_np):
            if done:
                self._episode_rewards.append(float(self._current_episode_reward[i]))
                self._episode_lengths.append(int(self._current_episode_length[i]))
                self._current_episode_reward[i] = 0
                self._current_episode_length[i] = 0
                self.total_episodes += 1

        self.total_steps += self.benchmark.num_envs

        info = {
            "success": timestep.info.get("success", [False] * self.benchmark.num_envs),
        }

        return timestep.next_observation, info

    def update(self) -> LogDict:
        """Perform one BRO update step.

        Returns:
            Dictionary of training metrics
        """
        self.key, sample_key = jax.random.split(self.key)

        # Sample batch from replay buffer
        batch = ReplayBuffer.sample(
            self.buffer_state,
            sample_key,
            self.cfg.batch_size,
        )

        # Perform BRO update (handles reset schedule internally)
        logs = self.learner.update(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards.squeeze(-1),
            next_observations=batch.next_observations,
            dones=batch.dones.squeeze(-1),
            env_step=self.total_steps,
        )

        self.total_gradient_steps += 1
        return logs

    def train_on_task(self, envs: VectorEnv, task_name: str, steps_per_task: int):
        """Train on a single task.

        Args:
            envs: Vector environment for this task
            task_name: Name of the task
            steps_per_task: Number of environment steps for this task
        """
        self._current_task_name = task_name

        # Initialize environment
        obs = envs.init()

        # Reset episode tracking for this task
        self._episode_rewards = []
        self._episode_lengths = []
        self._current_episode_reward = np.zeros(self.benchmark.num_envs)
        self._current_episode_length = np.zeros(self.benchmark.num_envs, dtype=int)

        task_start_step = self.total_steps
        start_time = time.time()
        last_log_step = self.total_steps
        last_eval_step = self.total_steps

        all_logs: list[LogDict] = []
        task_successes: list[float] = []

        while self.total_steps - task_start_step < steps_per_task:
            # Collect one step of experience
            obs, info = self.collect_step(envs, obs)

            # Track successes for this task
            if info.get("success"):
                task_successes.extend(info["success"])

            # Training updates (high replay ratio)
            if self.total_steps >= self.cfg.learning_starts:
                all_logs = []
                for _ in range(self.cfg.updates_per_step):
                    logs = self.update()
                    all_logs.append(logs)

            # Logging
            if self.total_steps - last_log_step >= 1000:
                elapsed = time.time() - start_time
                sps = (self.total_steps - task_start_step) / max(elapsed, 1e-6)

                log_dict: LogDict = {
                    "charts/total_steps": self.total_steps,
                    "charts/total_episodes": self.total_episodes,
                    "charts/total_gradient_steps": self.total_gradient_steps,
                    "charts/SPS": sps,
                    "charts/updates_per_step": self.cfg.updates_per_step,
                    "charts/buffer_size": int(self.buffer_state.size),
                    "charts/num_completed_tasks": len(self._completed_task_names),
                }

                # Episode metrics
                if self._episode_rewards:
                    log_dict["charts/mean_episode_return"] = float(np.mean(
                        self._episode_rewards[-100:]
                    ))
                    log_dict["charts/mean_episode_length"] = float(np.mean(
                        self._episode_lengths[-100:]
                    ))

                # Success rate (MetaWorld specific)
                recent_successes = info.get("success", [])
                if recent_successes:
                    log_dict["charts/success_rate"] = float(np.mean(recent_successes))

                # Training metrics (average over updates)
                if all_logs:
                    avg_logs: dict[str, float] = {}
                    for key in all_logs[0].keys():
                        values = [float(l[key]) for l in all_logs if key in l]
                        if values:
                            avg_logs[key] = float(np.mean(values))
                    log_dict.update(prefix_dict("train", avg_logs))

                self.logger.log(log_dict, step=self.total_steps)
                last_log_step = self.total_steps

                # Print progress
                temp_val = all_logs[-1].get("temperature/value", 0) if all_logs else 0
                optimism_val = all_logs[-1].get("optimism/value", 0) if all_logs else 0
                print(
                    f"Step {self.total_steps}, Eps: {self.total_episodes}, "
                    f"Return: {log_dict.get('charts/mean_episode_return', 0):.2f}, "
                    f"Temp: {temp_val:.3f}, Opt: {optimism_val:.3f}, "
                    f"SPS: {sps:.1f}",
                    flush=True,
                )

            # Periodic CL evaluation
            if self.total_steps - last_eval_step >= self._eval_frequency:
                self._log_cl_metrics()
                last_eval_step = self.total_steps

        # End of task: record final success rate
        if task_successes:
            final_success = float(np.mean(task_successes[-1000:]))
        else:
            eval_metrics = self.evaluate(envs, num_episodes=20)
            final_success = eval_metrics["success_rate"]

        self._final_success_rates[task_name] = final_success
        self._best_success_rates[task_name] = max(
            self._best_success_rates.get(task_name, 0.0), final_success
        )

        # Mark task as completed
        self._completed_task_names.append(task_name)
        print(f"  Task {task_name} completed. Final success rate: {final_success:.2%}", flush=True)

        # Final CL evaluation at end of task
        self._log_cl_metrics()

    def train(self):
        """Main training loop across all tasks."""
        print(f"Starting BRO training with updates_per_step={self.cfg.updates_per_step}", flush=True)
        print(f"Buffer size: {self.cfg.buffer_size}, Batch size: {self.cfg.batch_size}", flush=True)
        print(f"Learning starts: {self.cfg.learning_starts}", flush=True)
        print(f"Distributional: {self.cfg.distributional}, N quantiles: {self.cfg.n_quantiles}", flush=True)
        print(f"Reset steps: {self.cfg.reset_steps}", flush=True)
        print(f"CL evaluation frequency: every {self._eval_frequency} steps", flush=True)

        for task_idx, envs in enumerate(self.benchmark.tasks):
            task_name = getattr(envs, "task_name", f"task_{task_idx}")
            print(f"\n=== Training on task {task_idx}: {task_name} ===", flush=True)

            self.train_on_task(envs, task_name, self.train_cfg.steps_per_task)

        # Final summary
        print("\n=== Training Complete ===", flush=True)
        print(f"Tasks completed: {len(self._completed_task_names)}", flush=True)
        print(f"Final success rates:", flush=True)
        for task_name, success_rate in self._final_success_rates.items():
            forgetting = self._best_success_rates.get(task_name, success_rate) - success_rate
            print(f"  {task_name}: {success_rate:.2%} (forgetting: {forgetting:.2%})", flush=True)

        avg_final = np.mean(list(self._final_success_rates.values())) if self._final_success_rates else 0.0
        print(f"Average final success rate: {avg_final:.2%}", flush=True)

        self.logger.close()

    def evaluate(self, envs: VectorEnv, num_episodes: int = 10) -> dict:
        """Evaluate current policy on environment.

        Args:
            envs: Environment to evaluate on
            num_episodes: Number of episodes to run

        Returns:
            Dictionary of evaluation metrics
        """
        obs = envs.init()
        episode_returns = []
        episode_lengths = []
        successes = []

        current_return = np.zeros(self.benchmark.num_envs)
        current_length = np.zeros(self.benchmark.num_envs, dtype=int)

        while len(episode_returns) < num_episodes:
            action = self.select_action(obs, deterministic=True)
            timestep = envs.step(action)

            current_return += np.asarray(timestep.reward).squeeze(-1)
            current_length += 1

            dones = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)
            for i, done in enumerate(dones):
                if done:
                    episode_returns.append(float(current_return[i]))
                    episode_lengths.append(int(current_length[i]))
                    successes.append(timestep.info.get("success", [False])[i])
                    current_return[i] = 0
                    current_length[i] = 0

            obs = timestep.next_observation

        return {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "success_rate": np.mean(successes) if successes else 0.0,
        }

    def _create_task_env(self, task_name: str) -> VectorEnv:
        """Create a fresh environment for a specific task (for evaluation).

        Args:
            task_name: Name of the MetaWorld task

        Returns:
            VectorEnv for the task
        """
        from continual_learning.envs.metaworld import MetaWorldMT10Benchmark, MetaWorldVectorEnv

        if not isinstance(self.benchmark, MetaWorldMT10Benchmark):
            raise TypeError("CL evaluation requires MetaWorldMT10Benchmark")

        mt10 = self.benchmark._mt10
        task_cls = mt10.train_classes[task_name]
        task_instances = [
            t for t in mt10.train_tasks
            if t.env_name == task_name
        ]

        return MetaWorldVectorEnv(
            task_name=task_name,
            task_cls=task_cls,
            tasks=task_instances,
            num_envs=self.benchmark.num_envs,
            seed=self.seed + 1000,
        )

    def evaluate_all_tasks(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate current policy on all completed tasks.

        Args:
            num_episodes: Number of episodes per task

        Returns:
            Dictionary with per-task and aggregate metrics
        """
        results: dict[str, float] = {}

        task_success_rates = []
        for task_name in self._completed_task_names:
            env = self._create_task_env(task_name)
            try:
                metrics = self.evaluate(env, num_episodes)
            finally:
                env.close()
            success_rate = metrics["success_rate"]

            results[f"eval/{task_name}/success_rate"] = success_rate
            results[f"eval/{task_name}/mean_return"] = metrics["mean_return"]
            task_success_rates.append(success_rate)

            if task_name in self._best_success_rates:
                self._best_success_rates[task_name] = max(
                    self._best_success_rates[task_name], success_rate
                )

        # Also evaluate on current task
        if self._current_task_name and self._current_task_name not in self._completed_task_names:
            env = self._create_task_env(self._current_task_name)
            try:
                metrics = self.evaluate(env, num_episodes)
            finally:
                env.close()
            success_rate = metrics["success_rate"]

            results[f"eval/{self._current_task_name}/success_rate"] = success_rate
            results[f"eval/{self._current_task_name}/mean_return"] = metrics["mean_return"]
            task_success_rates.append(success_rate)

            if self._current_task_name not in self._best_success_rates:
                self._best_success_rates[self._current_task_name] = success_rate
            else:
                self._best_success_rates[self._current_task_name] = max(
                    self._best_success_rates[self._current_task_name], success_rate
                )

        if task_success_rates:
            results["eval/average_success_rate"] = float(np.mean(task_success_rates))

        # Compute forgetting
        if self._completed_task_names:
            forgetting_values = []
            for task_name in self._completed_task_names:
                if task_name in self._final_success_rates and task_name in results:
                    current = results.get(f"eval/{task_name}/success_rate", 0.0)
                    final = self._final_success_rates[task_name]
                    forgetting = max(0.0, final - current)
                    forgetting_values.append(forgetting)
                    results[f"eval/{task_name}/forgetting"] = forgetting

            if forgetting_values:
                results["eval/average_forgetting"] = float(np.mean(forgetting_values))

        return results

    def _log_cl_metrics(self):
        """Log continual learning metrics."""
        if not self._completed_task_names and not self._current_task_name:
            return

        print(f"  Evaluating on {len(self._completed_task_names)} completed tasks...", flush=True)
        metrics = self.evaluate_all_tasks(num_episodes=10)

        self.logger.log(metrics, step=self.total_steps)

        avg_success = metrics.get("eval/average_success_rate", 0.0)
        avg_forgetting = metrics.get("eval/average_forgetting", 0.0)
        print(f"  Avg Success: {avg_success:.2%}, Avg Forgetting: {avg_forgetting:.2%}", flush=True)
