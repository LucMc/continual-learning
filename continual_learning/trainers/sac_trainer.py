
"""Soft Actor-Critic (SAC) implementation for off-policy RL.

Adapted from https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py

This module implements SAC with:
- Twin Q-networks for reduced overestimation
- Automatic entropy tuning
- Soft target updates
- Support for the custom optimizer pattern (returns params, not updates)

We also create a SACTrainer to wrap the SAC algorithm for use with the framework's
reset method optimizers (REDO, ReGrAMA, CBP, etc.).
"""

import time
from functools import partial
from typing import NamedTuple

import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.scope import DenyList
from jaxtyping import Array, Float, PRNGKeyArray

from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.logging import LoggingConfig
from continual_learning.configs.rl import SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.envs import get_benchmark
from continual_learning.envs.base import ContinualLearningEnv, VectorEnv
from continual_learning.models import get_model_cls
from continual_learning.models.rl import QNetwork, TanhPolicy
from continual_learning.optim import get_optimizer
from continual_learning.types import LogDict, Observation
from continual_learning.utils.monitoring import Logger, prefix_dict
from continual_learning.utils.replay_buffer import ReplayBatch, ReplayBuffer
from continual_learning.utils.training import TrainState


def flatten_activations(features: dict) -> dict:
    """Flatten nested activation dict for reset methods.

    Strips the first (module scope) level so keys match {k[-2]: v} in reset methods.
    Applies spatial mean-pooling to conv activations (>2D) so CBP's mean_feature_act
    stays (C,), matching bias shape.
    For twin networks (q1/q2), q2 values overwrite q1 — consistent with how
    reset methods merge params via {k[-2]: v}.
    """
    flat = {}

    def _process(d: dict) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                _process(value)
            else:
                arr = value[0]  # unwrap sow tuple
                if arr.ndim > 2:  # conv: (B, H, W, C) → spatial mean → (B, C)
                    arr = arr.mean(axis=tuple(range(1, arr.ndim - 1)))
                flat[key] = (arr,)

    for module_feats in features.values():
        if isinstance(module_feats, dict):
            _process(module_feats)

    return flat


class SACTrainState(NamedTuple):
    """Training state for SAC algorithm."""

    actor: TrainState
    critic: TrainState
    target_critic_params: optax.Params
    log_alpha: Float[Array, "1"]
    alpha_optimizer_state: optax.OptState
    key: PRNGKeyArray


class SAC:
    """Soft Actor-Critic algorithm implementation."""

    @staticmethod
    def init_state(
        key: PRNGKeyArray,
        obs_dim: int,
        action_dim: int,
        cfg: SACConfig,
    ) -> SACTrainState:
        """Initialize SAC training state.

        Args:
            key: Random key for initialization
            obs_dim: Observation dimension
            action_dim: Action dimension
            cfg: SAC configuration

        Returns:
            Initialized SACTrainState
        """
        key, actor_key, critic_key = jax.random.split(key, 3)

        # Create dummy inputs for initialization
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        # Initialize actor (policy)
        actor_network_cls = get_model_cls(cfg.actor_config.network)
        actor_module = TanhPolicy(actor_network_cls, cfg.actor_config)
        actor_params = actor_module.lazy_init(
            actor_key,
            dummy_obs,
            training=False,
            mutable=DenyList(["activations", "preactivations"]),
        )

        actor = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=get_optimizer(cfg.actor_config.optimizer),
            kernel_init=cfg.actor_config.network.kernel_init,
            bias_init=cfg.actor_config.network.bias_init,
        )

        # Initialize critic (twin Q-networks)
        critic_network_cls = get_model_cls(cfg.critic_config.network)
        critic_module = QNetwork(critic_network_cls, cfg.critic_config)
        critic_params = critic_module.init(
            critic_key,
            dummy_obs,
            dummy_action,
            training=False,
            mutable=DenyList(["activations", "preactivations"]),
        )

        critic = TrainState.create(
            apply_fn=critic_module.apply,
            params=critic_params,
            tx=get_optimizer(cfg.critic_config.optimizer),
            kernel_init=cfg.critic_config.network.kernel_init,
            bias_init=cfg.critic_config.network.bias_init,
        )

        # Initialize target critic (copy of critic)
        target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        # Initialize entropy coefficient (log_alpha) - shape (1,) to match reference
        log_alpha = jnp.full((1,), jnp.log(cfg.alpha), dtype=jnp.float32)
        alpha_optimizer = optax.adam(learning_rate=cfg.alpha_lr)
        alpha_optimizer_state = alpha_optimizer.init(log_alpha)

        return SACTrainState(
            actor=actor,
            critic=critic,
            target_critic_params=target_critic_params,
            log_alpha=log_alpha,
            alpha_optimizer_state=alpha_optimizer_state,
            key=key,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("cfg", "target_entropy"))
    def update(
        state: SACTrainState,
        batch: ReplayBatch,
        cfg: SACConfig,
        target_entropy: float,
    ) -> tuple[SACTrainState, LogDict]:
        """Perform a full SAC update step.

        1. Inside actor loss: update alpha, then critic (using new alpha)
        2. Compute actor loss using updated critic
        3. Update actor
        4. Soft update target networks

        Args:
            state: Current training state
            batch: Batch of transitions from replay buffer
            cfg: SAC configuration
            target_entropy: Target entropy for auto-tuning

        Returns:
            Updated state and combined log dictionary
        """
        key, actor_loss_key, critic_loss_key = jax.random.split(state.key, 3)

        alpha_optimizer = optax.adam(learning_rate=cfg.alpha_lr)

        def update_critic(
            critic: TrainState,
            alpha_val: Float[Array, "1"],
        ) -> tuple[TrainState, LogDict]:
            """Update critic using TD learning."""
            # Sample a' from current policy for next states
            # Use sample_and_log_prob for numerical stability (avoids atanh(tanh(z)))
            next_dist = state.actor.apply_fn(state.actor.params, batch.next_observations)
            next_actions, next_action_log_probs = next_dist.sample_and_log_prob(
                seed=critic_loss_key
            )

            # Get target Q-values
            next_q1, next_q2 = state.critic.apply_fn(
                state.target_critic_params,
                batch.next_observations,
                next_actions,
            )

            # Stack and take min over critics
            q_values = jnp.stack([next_q1, next_q2], axis=0)
            min_qf_next_target = (
                jnp.min(q_values, axis=0) - alpha_val * next_action_log_probs.reshape(-1, 1)
            )

            next_q_value = jax.lax.stop_gradient(
                batch.rewards + (1 - batch.dones) * cfg.gamma * min_qf_next_target
            )

            def critic_loss_fn(params):
                # Get current Q-values with activation collection for reset methods
                (q1, q2), intermediates = critic.apply_fn(
                    params,
                    batch.observations,
                    batch.actions,
                    training=True,
                    mutable=("activations",),
                )
                q_pred = jnp.stack([q1, q2], axis=0)

                # 0.5 * MSE loss, mean over batch, sum over critics
                loss = 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum()
                return loss, (q_pred.mean(), intermediates)

            (critic_loss_value, (qf_values, intermediates)), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic.params)

            # Extract features for reset methods
            critic_feats = flatten_activations(intermediates.get("activations", {}))

            # Update critic using custom optimizer pattern
            new_critic = critic.apply_gradients(grads=critic_grads, features=critic_feats)

            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

            return new_critic, {
                "losses/qf_values": qf_values,
                "losses/qf_loss": critic_loss_value,
                "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            }

        def update_alpha(
            log_alpha: Float[Array, "1"],
            alpha_opt_state: optax.OptState,
            log_probs: Float[Array, " batch"],
        ) -> tuple[Float[Array, "1"], optax.OptState, Float[Array, "1"], LogDict]:
            """Update entropy coefficient."""

            def alpha_loss_fn(log_alpha_param):
                return (-log_alpha_param * (log_probs.reshape(-1, 1) + target_entropy)).mean()

            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
            updates, new_alpha_opt_state = alpha_optimizer.update(alpha_grads, alpha_opt_state)
            new_log_alpha = optax.apply_updates(log_alpha, updates)
            alpha_val = jnp.exp(new_log_alpha)

            return (
                new_log_alpha,
                new_alpha_opt_state,
                alpha_val,
                {
                    "losses/alpha_loss": alpha_loss_value,
                    "alpha": alpha_val.sum(),
                },
            )

        def actor_loss_fn(actor_params):
            """Compute actor loss, also triggers alpha and critic updates."""
            # Sample actions from policy with activation collection
            # Use sample_and_log_prob for numerical stability (avoids atanh(tanh(z)))
            dist, intermediates = state.actor.apply_fn(
                actor_params,
                batch.observations,
                training=True,
                mutable=("activations",),
            )
            action_samples, log_probs = dist.sample_and_log_prob(seed=actor_loss_key)

            # Update alpha first (using log_probs from current policy) if auto_entropy enabled
            if cfg.auto_entropy:
                new_log_alpha, new_alpha_opt_state, alpha_val, alpha_logs = update_alpha(
                    state.log_alpha, state.alpha_optimizer_state, log_probs
                )
            else:
                # Fixed alpha - no update
                new_log_alpha = state.log_alpha
                new_alpha_opt_state = state.alpha_optimizer_state
                alpha_val = jnp.exp(state.log_alpha)
                alpha_logs = {"losses/alpha_loss": jnp.array(0.0), "alpha": alpha_val.sum()}
            alpha_val = jax.lax.stop_gradient(alpha_val)

            # Update critic (using new alpha)
            new_critic, critic_logs = update_critic(state.critic, alpha_val)
            logs = {**alpha_logs, **critic_logs}

            # Compute actor loss using updated critic's Q-values
            q1, q2 = new_critic.apply_fn(new_critic.params, batch.observations, action_samples)
            q_values = jnp.stack([q1, q2], axis=0)
            min_qf_values = jnp.min(q_values, axis=0)

            loss = (alpha_val * log_probs.reshape(-1, 1) - min_qf_values).mean()

            return loss, (new_log_alpha, new_alpha_opt_state, new_critic, logs, intermediates)

        (
            actor_loss_value,
            (new_log_alpha, new_alpha_opt_state, new_critic, logs, intermediates),
        ), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)

        # Extract features for reset methods
        actor_feats = flatten_activations(intermediates.get("activations", {}))

        # Update actor using custom optimizer pattern
        new_actor = state.actor.apply_gradients(grads=actor_grads, features=actor_feats)

        flat_grads, _ = flatten_util.ravel_pytree(actor_grads)
        logs["metrics/actor_grad_magnitude"] = jnp.linalg.norm(flat_grads)

        flat_params_act, _ = flatten_util.ravel_pytree(state.actor.params)
        logs["metrics/actor_params_norm"] = jnp.linalg.norm(flat_params_act)

        flat_params_crit, _ = flatten_util.ravel_pytree(state.critic.params)
        logs["metrics/critic_params_norm"] = jnp.linalg.norm(flat_params_crit)

        # Soft update target networks
        new_target_critic_params = optax.incremental_update(
            new_critic.params,
            state.target_critic_params,
            cfg.tau,
        )

        new_state = SACTrainState(
            actor=new_actor,
            critic=new_critic,
            target_critic_params=new_target_critic_params,
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt_state,
            key=key,
        )

        return new_state, {**logs, "losses/actor_loss": actor_loss_value}

    @staticmethod
    @jax.jit
    def sample_action(
        state: SACTrainState,
        observation: Float[Array, "... obs_dim"],
    ) -> tuple[SACTrainState, Float[Array, "... action_dim"]]:
        """Sample action from policy.

        Args:
            state: Current training state
            observation: Environment observation

        Returns:
            Tuple of (updated state with new key, sampled action)
        """
        key, action_key = jax.random.split(state.key)
        dist = state.actor.apply_fn(state.actor.params, observation)
        action = dist.sample(seed=action_key)
        new_state = state._replace(key=key)
        return new_state, action

    @staticmethod
    @jax.jit
    def eval_action(
        state: SACTrainState,
        observation: Float[Array, "... obs_dim"],
    ) -> Float[Array, "... action_dim"]:
        """Get deterministic action (mode) from policy.

        For TanhNormal distributions, mode = tanh(mean) since:
        1. The mode of Normal(mean, std) is mean
        2. Tanh is monotonically increasing

        Args:
            state: Current training state
            observation: Environment observation

        Returns:
            Deterministic action (mode of the policy distribution)
        """
        dist = state.actor.apply_fn(state.actor.params, observation)
        # Access the mean from the base MultivariateNormalDiag distribution
        mean = dist.distribution.loc
        return jnp.tanh(mean)


class SACTrainer:
    """SAC Trainer with reset method support for MetaWorld."""

    def __init__(
        self,
        seed: int,
        sac_config: SACConfig,
        env_cfg: EnvConfig,
        train_cfg: RLTrainingConfig,
        logs_cfg: LoggingConfig,
    ):
        self.key = jax.random.PRNGKey(seed)
        self.cfg = sac_config
        self.train_cfg = train_cfg
        self.seed = seed

        self.logger = Logger(
            logs_cfg,
            run_config={
                "algorithm": sac_config,
                "benchmark": env_cfg,
                "training": train_cfg,
            },
        )

        benchmark = get_benchmark(seed, env_cfg)
        if not isinstance(benchmark, ContinualLearningEnv):
            raise ValueError("SACTrainer requires ContinualLearningEnv")
        self.benchmark = benchmark

        self.obs_dim = benchmark.observation_spec.shape[-1]
        self.action_dim = benchmark.action_dim

        self.target_entropy = (
            sac_config.target_entropy
            if sac_config.target_entropy is not None
            else -float(self.action_dim)
        )

        self.replay_buffer = ReplayBuffer(
            capacity=sac_config.buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )

        self.key, init_key = jax.random.split(self.key)
        self.sac_state = SAC.init_state(
            key=init_key,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            cfg=sac_config,
        )

        self.buffer_state = self.replay_buffer.init()

        self.total_steps = 0
        self.total_episodes = 0
        self.total_gradient_steps = 0

        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_successes: list[float] = []
        self._current_episode_reward = np.zeros(env_cfg.num_envs)
        self._current_episode_length = np.zeros(env_cfg.num_envs, dtype=int)

        self._completed_task_names: list[str] = []
        self._best_success_rates: dict[str, float] = {}
        self._final_success_rates: dict[str, float] = {}
        self._current_task_name: str | None = None
        self._eval_frequency = 10_000

    def select_action(self, obs: Observation, deterministic: bool = False) -> jax.Array:
        self.key, action_key = jax.random.split(self.key)
        dist = self.sac_state.actor.apply_fn(self.sac_state.actor.params, obs)

        if deterministic:
            try:
                action = dist.mode()
            except NotImplementedError:
                base_mean = dist.distribution.loc
                action = jnp.tanh(base_mean)
        else:
            action = dist.sample(seed=action_key)

        return jnp.clip(action, -1.0, 1.0)

    def collect_step(self, envs: VectorEnv, obs: Observation) -> tuple[Observation, dict]:
        action = self.select_action(obs)
        action = jnp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        timestep = envs.step(action)

        self.buffer_state = ReplayBuffer.add(
            self.buffer_state,
            obs=obs,
            action=action,
            reward=timestep.reward,
            next_obs=timestep.next_observation,
            done=timestep.terminated | timestep.truncated,
        )

        rewards_np = np.asarray(timestep.reward).squeeze(-1)
        dones_np = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)
        success_flags = timestep.info.get(
            "success", [False] * self.benchmark.num_envs
        )

        self._current_episode_reward += rewards_np
        self._current_episode_length += 1

        for i, done in enumerate(dones_np):
            if done:
                self._episode_rewards.append(float(self._current_episode_reward[i]))
                self._episode_lengths.append(int(self._current_episode_length[i]))
                self._episode_successes.append(float(success_flags[i]))
                self._current_episode_reward[i] = 0
                self._current_episode_length[i] = 0
                self.total_episodes += 1

        self.total_steps += self.benchmark.num_envs

        return timestep.next_observation, {}

    def update(self) -> LogDict:
        self.key, sample_key = jax.random.split(self.key)

        batch = ReplayBuffer.sample(
            self.buffer_state,
            sample_key,
            self.cfg.batch_size,
        )

        self.sac_state, logs = SAC.update(
            self.sac_state,
            batch,
            self.cfg,
            self.target_entropy,
        )

        self.total_gradient_steps += 1
        return logs

    def _reset_on_task_change(self):
        """Reset replay buffer and optimizer states at task boundaries.

        Follows the Continual World paper convention (reset_buffer_on_task_change=True,
        reset_optimizer_on_task_change=True) to prevent cross-task contamination.
        """
        # Clear replay buffer — prevents task 1 data from crowding out task 2 learning
        self.buffer_state = self.replay_buffer.init()

        # Reset actor and critic optimizer states (zero out Adam momentum/variance)
        new_actor_opt_state = self.sac_state.actor.tx.init(self.sac_state.actor.params)
        new_critic_opt_state = self.sac_state.critic.tx.init(self.sac_state.critic.params)

        # Reset alpha (entropy coefficient) and its optimizer to initial values
        new_log_alpha = jnp.full((1,), jnp.log(self.cfg.alpha), dtype=jnp.float32)
        alpha_optimizer = optax.adam(learning_rate=self.cfg.alpha_lr)
        new_alpha_opt_state = alpha_optimizer.init(new_log_alpha)

        self.sac_state = self.sac_state._replace(
            actor=self.sac_state.actor.replace(opt_state=new_actor_opt_state),
            critic=self.sac_state.critic.replace(opt_state=new_critic_opt_state),
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt_state,
        )

    def train_on_task(self, envs: VectorEnv, task_name: str, steps_per_task: int):
        self._current_task_name = task_name

        if self._completed_task_names:
            self._reset_on_task_change()

        obs = envs.init()

        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_successes = []
        self._current_episode_reward = np.zeros(self.benchmark.num_envs)
        self._current_episode_length = np.zeros(self.benchmark.num_envs, dtype=int)

        task_start_step = self.total_steps
        start_time = time.time()
        last_log_step = self.total_steps
        last_eval_step = self.total_steps

        all_logs: list[LogDict] = []

        while self.total_steps - task_start_step < steps_per_task:
            obs, _ = self.collect_step(envs, obs)

            if self.total_steps >= self.cfg.learning_starts:
                all_logs = []
                for _ in range(self.cfg.replay_ratio):
                    logs = self.update()
                    all_logs.append(logs)

            if self.total_steps - last_log_step >= 1000:
                elapsed = time.time() - start_time
                sps = (self.total_steps - task_start_step) / max(elapsed, 1e-6)

                log_dict: LogDict = {
                    "charts/total_steps": self.total_steps,
                    "charts/total_episodes": self.total_episodes,
                    "charts/total_gradient_steps": self.total_gradient_steps,
                    "charts/SPS": sps,
                    "charts/replay_ratio": self.cfg.replay_ratio,
                    "charts/buffer_size": int(self.buffer_state.size),
                    "charts/num_completed_tasks": len(self._completed_task_names),
                }

                if self._episode_rewards:
                    log_dict["charts/mean_episode_return"] = float(np.mean(
                        self._episode_rewards[-100:]
                    ))
                    log_dict["charts/mean_episode_length"] = float(np.mean(
                        self._episode_lengths[-100:]
                    ))

                if self._episode_successes:
                    log_dict["charts/success_rate"] = float(
                        np.mean(self._episode_successes[-100:])
                    )

                if all_logs:
                    avg_logs: dict[str, float] = {}
                    for key in all_logs[0].keys():
                        values = [float(l[key]) for l in all_logs if key in l]
                        if values:
                            avg_logs[key] = float(np.mean(values))
                    log_dict.update(prefix_dict("train", avg_logs))

                self.logger.log(log_dict, step=self.total_steps)
                last_log_step = self.total_steps

                print(
                    f"Step {self.total_steps}, Eps: {self.total_episodes}, "
                    f"Return: {log_dict.get('charts/mean_episode_return', 0):.2f}, "
                    f"SPS: {sps:.1f}",
                    flush=True,
                )

            if self.total_steps - last_eval_step >= self._eval_frequency:
                self._log_cl_metrics()
                last_eval_step = self.total_steps

        eval_metrics = self.evaluate(envs, num_episodes=20)
        final_success = eval_metrics["success_rate"]

        self._final_success_rates[task_name] = final_success
        self._best_success_rates[task_name] = max(
            self._best_success_rates.get(task_name, 0.0), final_success
        )

        self._completed_task_names.append(task_name)
        print(f"  Task {task_name} completed. Final success rate: {final_success:.2%}", flush=True)

        self._log_cl_metrics()

    def train(self):
        print(f"Starting SAC training with replay_ratio={self.cfg.replay_ratio}", flush=True)
        print(f"Buffer size: {self.cfg.buffer_size}, Batch size: {self.cfg.batch_size}", flush=True)
        print(f"Learning starts: {self.cfg.learning_starts}", flush=True)

        for task_idx, envs in enumerate(self.benchmark.tasks):
            task_name = getattr(envs, "task_name", f"task_{task_idx}")
            print(f"\n=== Training on task {task_idx}: {task_name} ===", flush=True)

            self.train_on_task(envs, task_name, self.train_cfg.steps_per_task)

        print("\n=== Training Complete ===", flush=True)
        print(f"Tasks completed: {len(self._completed_task_names)}", flush=True)
        print("Final success rates:", flush=True)
        for task_name, success_rate in self._final_success_rates.items():
            forgetting = self._best_success_rates.get(task_name, success_rate) - success_rate
            print(f"  {task_name}: {success_rate:.2%} (forgetting: {forgetting:.2%})", flush=True)

        avg_final = np.mean(list(self._final_success_rates.values())) if self._final_success_rates else 0.0
        print(f"Average final success rate: {avg_final:.2%}", flush=True)

        self.logger.close()

    def evaluate(self, envs: VectorEnv, num_episodes: int = 10) -> dict:
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
        from continual_learning.envs.metaworld import (
            MetaWorldMT10Benchmark,
            MetaWorldVectorEnv,
        )

        if not isinstance(self.benchmark, MetaWorldMT10Benchmark):
            raise TypeError("CL evaluation requires MetaWorldMT10Benchmark")

        mt10 = self.benchmark._mt10
        task_cls = mt10.train_classes[task_name]
        task_instances = [t for t in mt10.train_tasks if t.env_name == task_name]
        task_names = self.benchmark.task_names
        task_idx = task_names.index(task_name)

        return MetaWorldVectorEnv(
            task_name=task_name,
            task_cls=task_cls,
            tasks=task_instances,
            num_envs=self.benchmark.num_envs,
            seed=self.seed + 1000,
            task_idx=task_idx,
            num_tasks=len(task_names),
        )

    def evaluate_all_tasks(self, num_episodes: int = 10) -> dict[str, float]:
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

        if self._completed_task_names:
            forgetting_values = []
            for task_name in self._completed_task_names:
                if task_name in self._final_success_rates:
                    current = results.get(f"eval/{task_name}/success_rate", 0.0)
                    final = self._final_success_rates[task_name]
                    forgetting = max(0.0, final - current)
                    forgetting_values.append(forgetting)
                    results[f"eval/{task_name}/forgetting"] = forgetting

            if forgetting_values:
                results["eval/average_forgetting"] = float(np.mean(forgetting_values))

        return results

    def _log_cl_metrics(self):
        if not self._completed_task_names and not self._current_task_name:
            return

        print(f"  Evaluating on {len(self._completed_task_names)} completed tasks...", flush=True)
        metrics = self.evaluate_all_tasks(num_episodes=10)
        self.logger.log(metrics, step=self.total_steps)

        avg_success = metrics.get("eval/average_success_rate", 0.0)
        avg_forgetting = metrics.get("eval/average_forgetting", 0.0)
        print(f"  Avg Success: {avg_success:.2%}, Avg Forgetting: {avg_forgetting:.2%}", flush=True)
