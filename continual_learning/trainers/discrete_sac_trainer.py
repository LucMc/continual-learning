"""Discrete Soft Actor-Critic (SAC) for environments with discrete action spaces.

Implements the discrete SAC algorithm from Christodoulou (2019):
  "Soft Actor-Critic for Discrete Action Settings"
  https://arxiv.org/abs/1910.07207

Key differences from continuous SAC:
- Policy outputs a Categorical distribution (not TanhNormal)
- Q-network takes only observations, outputs Q(s, :) for all actions
- Target value uses expectation over all actions (not a sampled action)
- Entropy computed analytically as H(π) = -Σ π log π
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
from continual_learning.configs.models import CNNConfig
from continual_learning.configs.logging import LoggingConfig
from continual_learning.configs.rl import SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.envs import get_benchmark
from continual_learning.envs.base import ContinualLearningEnv, VectorEnv
from continual_learning.models import get_model_cls
from continual_learning.models.discrete_rl import CategoricalPolicy, DiscreteQNetwork
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


class DiscreteSACTrainState(NamedTuple):
    """Training state for Discrete SAC algorithm."""

    actor: TrainState
    critic: TrainState
    target_critic_params: optax.Params
    log_alpha: Float[Array, "1"]
    alpha_optimizer_state: optax.OptState
    key: PRNGKeyArray


class DiscreteSAC:
    """Discrete Soft Actor-Critic algorithm."""

    @staticmethod
    def init_state(
        key: PRNGKeyArray,
        obs_dim: int,
        n_actions: int,
        cfg: SACConfig,
    ) -> DiscreteSACTrainState:
        """Initialize Discrete SAC training state."""
        key, actor_key, critic_key = jax.random.split(key, 3)

        dummy_obs = jnp.zeros((1, obs_dim))

        # Actor: categorical policy over n_actions
        actor_network_cls = get_model_cls(cfg.actor_config.network)
        input_spatial_shape = (10, 10, 10) if isinstance(cfg.actor_config.network, CNNConfig) else None
        actor_module = CategoricalPolicy(actor_network_cls, cfg.actor_config, n_actions, input_spatial_shape)
        actor_params = actor_module.init(
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

        # Critic: twin Q-network, obs → Q(s, :)
        critic_network_cls = get_model_cls(cfg.critic_config.network)
        input_spatial_shape_c = (10, 10, 10) if isinstance(cfg.critic_config.network, CNNConfig) else None
        critic_module = DiscreteQNetwork(critic_network_cls, cfg.critic_config, n_actions, input_spatial_shape_c)
        critic_params = critic_module.init(
            critic_key,
            dummy_obs,
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

        target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        log_alpha = jnp.full((1,), jnp.log(cfg.alpha), dtype=jnp.float32)
        alpha_optimizer = optax.adam(learning_rate=cfg.alpha_lr)
        alpha_optimizer_state = alpha_optimizer.init(log_alpha)

        return DiscreteSACTrainState(
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
        state: DiscreteSACTrainState,
        batch: ReplayBatch,
        cfg: SACConfig,
        target_entropy: float,
        valid_action_mask: Array,
    ) -> tuple[DiscreteSACTrainState, LogDict]:
        """Perform a full Discrete SAC update step.

        Args:
            state: Current training state
            batch: Batch of transitions (actions stored as float integers in batch.actions)
            cfg: SAC configuration
            target_entropy: Target entropy for auto-tuning (static; triggers recompile on change)
            valid_action_mask: Bool array (n_actions,) — True for valid actions in current task

        Returns:
            Updated state and log dictionary
        """
        key, actor_loss_key, critic_loss_key = jax.random.split(state.key, 3)

        alpha_optimizer = optax.adam(learning_rate=cfg.alpha_lr)

        def update_critic(
            critic: TrainState,
            alpha_val: Float[Array, "1"],
        ) -> tuple[TrainState, LogDict]:
            """Update critic using discrete soft Bellman target."""
            # Policy probs for next states
            next_dist = state.actor.apply_fn(state.actor.params, batch.next_observations)
            masked_next_logits = jnp.where(valid_action_mask, next_dist.logits, -1e9)
            next_probs = jax.nn.softmax(masked_next_logits, axis=-1)   # (B, n_actions)
            next_log_probs = jnp.log(next_probs + 1e-8)

            # Target Q from frozen target critic
            next_q1, next_q2 = state.critic.apply_fn(
                state.target_critic_params, batch.next_observations
            )
            min_q_next = jnp.minimum(next_q1, next_q2)                 # (B, n_actions)

            # Soft value: V(s') = Σ_a π(a|s') * (min_Q(s', a) - α * log π(a|s'))
            v_next = (
                next_probs * (min_q_next - alpha_val * next_log_probs)
            ).sum(axis=-1, keepdims=True)                               # (B, 1)

            target = jax.lax.stop_gradient(
                batch.rewards + (1 - batch.dones) * cfg.gamma * v_next
            )                                                           # (B, 1)

            def critic_loss_fn(params):
                (q1, q2), intermediates = critic.apply_fn(
                    params,
                    batch.observations,
                    training=True,
                    mutable=("activations",),
                )
                # Index Q-values for the taken action
                B = q1.shape[0]
                action_idx = batch.actions.astype(jnp.int32).squeeze(-1)  # (B,)
                q1_taken = q1[jnp.arange(B), action_idx, None]            # (B, 1)
                q2_taken = q2[jnp.arange(B), action_idx, None]

                loss = (
                    0.5 * ((q1_taken - target) ** 2).mean()
                    + 0.5 * ((q2_taken - target) ** 2).mean()
                )
                mean_q = jnp.stack([q1_taken, q2_taken]).mean()
                return loss, (mean_q, intermediates)

            (critic_loss_value, (qf_values, intermediates)), critic_grads = (
                jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
            )

            critic_feats = flatten_activations(intermediates.get("activations", {}))
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
            entropy: Float[Array, " batch"],
        ) -> tuple[Float[Array, "1"], optax.OptState, Float[Array, "1"], LogDict]:
            """Update entropy coefficient using batch-averaged policy entropy."""

            def alpha_loss_fn(log_alpha_param):
                # Minimise: log_α * (H(π) - H_target)
                # When H < H_target → gradient < 0 → log_α increases → more entropy pressure
                return (log_alpha_param * (entropy - target_entropy)).mean()

            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
            updates, new_alpha_opt_state = alpha_optimizer.update(alpha_grads, alpha_opt_state)
            new_log_alpha = optax.apply_updates(log_alpha, updates)
            new_log_alpha = jnp.clip(new_log_alpha, -5.0, 2.0)
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
            """Compute actor loss; also triggers alpha and critic updates."""
            dist, intermediates = state.actor.apply_fn(
                actor_params,
                batch.observations,
                training=True,
                mutable=("activations",),
            )
            masked_logits = jnp.where(valid_action_mask, dist.logits, -1e9)
            probs = jax.nn.softmax(masked_logits, axis=-1)              # (B, n_actions)
            log_probs = jnp.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(axis=-1)                  # (B,)

            # Update alpha first (using current policy entropy)
            if cfg.auto_entropy:
                new_log_alpha, new_alpha_opt_state, alpha_val, alpha_logs = update_alpha(
                    state.log_alpha, state.alpha_optimizer_state, entropy
                )
            else:
                new_log_alpha = state.log_alpha
                new_alpha_opt_state = state.alpha_optimizer_state
                alpha_val = jnp.exp(state.log_alpha)
                alpha_logs = {"losses/alpha_loss": jnp.array(0.0), "alpha": alpha_val.sum()}
            alpha_val = jax.lax.stop_gradient(alpha_val)

            # Update critic using new alpha
            new_critic, critic_logs = update_critic(state.critic, alpha_val)
            logs = {**alpha_logs, **critic_logs}

            # Actor loss: E_s[Σ_a π(a|s) * (α * log π(a|s) - min_Q(s, a))]
            q1, q2 = new_critic.apply_fn(new_critic.params, batch.observations)
            min_q = jnp.minimum(q1, q2)                                 # (B, n_actions)
            loss = (probs * (alpha_val * log_probs - min_q)).sum(axis=-1).mean()

            return loss, (new_log_alpha, new_alpha_opt_state, new_critic, logs, intermediates)

        (
            actor_loss_value,
            (new_log_alpha, new_alpha_opt_state, new_critic, logs, intermediates),
        ), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)

        actor_feats = flatten_activations(intermediates.get("activations", {}))
        new_actor = state.actor.apply_gradients(grads=actor_grads, features=actor_feats)

        flat_grads, _ = flatten_util.ravel_pytree(actor_grads)
        logs["metrics/actor_grad_magnitude"] = jnp.linalg.norm(flat_grads)

        flat_params_act, _ = flatten_util.ravel_pytree(state.actor.params)
        logs["metrics/actor_params_norm"] = jnp.linalg.norm(flat_params_act)

        flat_params_crit, _ = flatten_util.ravel_pytree(state.critic.params)
        logs["metrics/critic_params_norm"] = jnp.linalg.norm(flat_params_crit)

        # Soft update target critic
        new_target_critic_params = optax.incremental_update(
            new_critic.params, state.target_critic_params, cfg.tau
        )

        new_state = DiscreteSACTrainState(
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
        state: DiscreteSACTrainState,
        observation: Array,
        valid_action_mask: Array,
    ) -> tuple[DiscreteSACTrainState, Array]:
        """Sample discrete action from policy (stochastic)."""
        key, action_key = jax.random.split(state.key)
        dist = state.actor.apply_fn(state.actor.params, observation)
        masked_logits = jnp.where(valid_action_mask, dist.logits, -1e9)
        action = jax.random.categorical(action_key, masked_logits)     # (num_envs,) int
        new_state = state._replace(key=key)
        return new_state, action

    @staticmethod
    @jax.jit
    def eval_action(
        state: DiscreteSACTrainState,
        observation: Array,
        valid_action_mask: Array,
    ) -> Array:
        """Select greedy (deterministic) action."""
        dist = state.actor.apply_fn(state.actor.params, observation)
        masked_logits = jnp.where(valid_action_mask, dist.logits, -1e9)
        return jnp.argmax(masked_logits, axis=-1)                      # (num_envs,) int


class DiscreteSACTrainer:
    """Discrete SAC Trainer for environments with discrete action spaces."""

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
            raise ValueError("DiscreteSACTrainer requires ContinualLearningEnv")
        self.benchmark = benchmark

        self.obs_dim = benchmark.observation_spec.shape[-1]   # 1000
        self.max_n_actions = benchmark.action_dim             # MAX_N_ACTIONS

        # target_entropy updated per task; initialise from MAX_N_ACTIONS
        self._target_entropy_override = sac_config.target_entropy
        self.target_entropy = float(
            sac_config.target_entropy
            if sac_config.target_entropy is not None
            else 0.5 * np.log(self.max_n_actions)
        )

        # valid_action_mask: True for valid actions; updated per task
        self._valid_action_mask = jnp.ones(self.max_n_actions, dtype=bool)

        # Replay buffer: action_dim=1 (scalar integer stored as float)
        self.replay_buffer = ReplayBuffer(
            capacity=sac_config.buffer_size,
            obs_dim=self.obs_dim,
            action_dim=1,
        )

        self.key, init_key = jax.random.split(self.key)
        self.sac_state = DiscreteSAC.init_state(
            key=init_key,
            obs_dim=self.obs_dim,
            n_actions=self.max_n_actions,
            cfg=sac_config,
        )

        self.buffer_state = self.replay_buffer.init()

        self.total_steps = 0
        self.total_episodes = 0
        self.total_gradient_steps = 0

        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_episode_reward = np.zeros(env_cfg.num_envs)
        self._current_episode_length = np.zeros(env_cfg.num_envs, dtype=int)

        self._completed_task_names: list[str] = []
        self._best_returns: dict[str, float] = {}
        self._final_returns: dict[str, float] = {}
        self._current_task_name: str | None = None
        self._task_envs: dict[str, VectorEnv] = {}
        self._eval_frequency = 10_000

    def select_action(self, obs: Observation, deterministic: bool = False) -> jax.Array:
        if deterministic:
            action = DiscreteSAC.eval_action(
                self.sac_state, obs, self._valid_action_mask
            )
        else:
            self.sac_state, action = DiscreteSAC.sample_action(
                self.sac_state, obs, self._valid_action_mask
            )
        return action  # (num_envs,) int

    def collect_step(self, envs: VectorEnv, obs: Observation) -> tuple[Observation, dict]:
        action = self.select_action(obs)

        timestep = envs.step(action)

        # Store integer action as scalar float in buffer (action_dim=1)
        action_for_buffer = action.reshape(self.benchmark.num_envs, 1).astype(jnp.float32)

        self.buffer_state = ReplayBuffer.add(
            self.buffer_state,
            obs=obs,
            action=action_for_buffer,
            reward=timestep.reward,
            next_obs=timestep.next_observation,
            done=timestep.terminated | timestep.truncated,
        )

        rewards_np = np.asarray(timestep.reward).squeeze(-1)
        dones_np = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)

        self._current_episode_reward += rewards_np
        self._current_episode_length += 1

        for i, done in enumerate(dones_np):
            if done:
                self._episode_rewards.append(float(self._current_episode_reward[i]))
                self._episode_lengths.append(int(self._current_episode_length[i]))
                self._current_episode_reward[i] = 0
                self._current_episode_length[i] = 0
                self.total_episodes += 1

        self.total_steps += self.benchmark.num_envs
        return timestep.next_observation, {}

    def update(self) -> LogDict:
        self.key, sample_key = jax.random.split(self.key)

        batch = ReplayBuffer.sample(self.buffer_state, sample_key, self.cfg.batch_size)

        self.sac_state, logs = DiscreteSAC.update(
            self.sac_state,
            batch,
            self.cfg,
            self.target_entropy,
            self._valid_action_mask,
        )

        self.total_gradient_steps += 1
        return logs

    def _reset_on_task_change(self):
        """Reset replay buffer and optimizer states at task boundaries."""
        self.buffer_state = self.replay_buffer.init()

        new_actor_opt_state = self.sac_state.actor.tx.init(self.sac_state.actor.params)
        new_critic_opt_state = self.sac_state.critic.tx.init(self.sac_state.critic.params)

        new_log_alpha = jnp.full((1,), jnp.log(self.cfg.alpha), dtype=jnp.float32)
        alpha_optimizer = optax.adam(learning_rate=self.cfg.alpha_lr)
        new_alpha_opt_state = alpha_optimizer.init(new_log_alpha)

        self.sac_state = self.sac_state._replace(
            actor=self.sac_state.actor.replace(opt_state=new_actor_opt_state),
            critic=self.sac_state.critic.replace(opt_state=new_critic_opt_state),
            target_critic_params=jax.tree.map(
                lambda x: x.copy(), self.sac_state.critic.params
            ),
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt_state,
        )

    def train_on_task(self, envs: VectorEnv, task_name: str, steps_per_task: int):
        self._current_task_name = task_name
        self._task_envs[task_name] = envs

        # Update action mask and target entropy for this task's action space
        n_actions = getattr(envs, "n_actions", self.max_n_actions)
        self._valid_action_mask = jnp.arange(self.max_n_actions) < n_actions
        self.target_entropy = float(
            self._target_entropy_override
            if self._target_entropy_override is not None
            else 0.5 * np.log(n_actions)
        )

        if self._completed_task_names:
            self._reset_on_task_change()

        obs = envs.init()

        self._episode_rewards = []
        self._episode_lengths = []
        self._current_episode_reward = np.zeros(self.benchmark.num_envs)
        self._current_episode_length = np.zeros(self.benchmark.num_envs, dtype=int)

        task_start_step = self.total_steps
        start_time = time.time()
        last_log_step = self.total_steps
        last_eval_step = self.total_steps

        all_logs: list[LogDict] = []

        while self.total_steps - task_start_step < steps_per_task:
            obs, _ = self.collect_step(envs, obs)

            if self.total_steps - task_start_step >= self.cfg.learning_starts:
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
                    log_dict["charts/mean_episode_return"] = float(
                        np.mean(self._episode_rewards[-100:])
                    )
                    log_dict["charts/mean_episode_length"] = float(
                        np.mean(self._episode_lengths[-100:])
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
        final_return = eval_metrics["mean_return"]

        self._final_returns[task_name] = final_return
        self._best_returns[task_name] = max(
            self._best_returns.get(task_name, final_return), final_return
        )

        self._completed_task_names.append(task_name)
        print(f"  Task {task_name} completed. Final mean return: {final_return:.2f}", flush=True)

        self._log_cl_metrics()

    def train(self):
        print(
            f"Starting Discrete SAC training with replay_ratio={self.cfg.replay_ratio}",
            flush=True,
        )
        print(
            f"Buffer size: {self.cfg.buffer_size}, Batch size: {self.cfg.batch_size}",
            flush=True,
        )
        print(f"Learning starts: {self.cfg.learning_starts}", flush=True)

        for task_idx, envs in enumerate(self.benchmark.tasks):
            task_name = getattr(envs, "task_name", f"task_{task_idx}")
            n_actions = getattr(envs, "n_actions", self.max_n_actions)
            print(
                f"\n=== Training on task {task_idx}: {task_name} "
                f"({n_actions} actions) ===",
                flush=True,
            )
            self.train_on_task(envs, task_name, self.train_cfg.steps_per_task)

        print("\n=== Training Complete ===", flush=True)
        print(f"Tasks completed: {len(self._completed_task_names)}", flush=True)
        print("Final mean returns:", flush=True)
        for task_name, mean_return in self._final_returns.items():
            forgetting = self._best_returns.get(task_name, mean_return) - mean_return
            print(
                f"  {task_name}: {mean_return:.2f} (forgetting: {forgetting:.2f})",
                flush=True,
            )

        avg_final = (
            np.mean(list(self._final_returns.values())) if self._final_returns else 0.0
        )
        print(f"Average final mean return: {avg_final:.2f}", flush=True)

        self.logger.close()

    def evaluate(self, envs: VectorEnv, num_episodes: int = 10) -> dict:
        obs = envs.init()
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

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
                    current_return[i] = 0
                    current_length[i] = 0

            obs = timestep.next_observation

        return {
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "mean_length": float(np.mean(episode_lengths)),
        }

    def evaluate_all_tasks(self, num_episodes: int = 10) -> dict[str, float]:
        results: dict[str, float] = {}
        return_values: list[float] = []

        for task_name, env in self._task_envs.items():
            # Temporarily switch action mask for this task's action space
            n_actions = getattr(env, "n_actions", self.max_n_actions)
            prev_mask = self._valid_action_mask
            self._valid_action_mask = jnp.arange(self.max_n_actions) < n_actions

            metrics = self.evaluate(env, num_episodes)

            self._valid_action_mask = prev_mask  # restore

            mean_return = metrics["mean_return"]
            results[f"eval/{task_name}/mean_return"] = mean_return
            return_values.append(mean_return)

            if task_name in self._best_returns:
                self._best_returns[task_name] = max(
                    self._best_returns[task_name], mean_return
                )

        if return_values:
            results["eval/average_mean_return"] = float(np.mean(return_values))

        # Compute forgetting for completed tasks
        forgetting_values = []
        for task_name in self._completed_task_names:
            if task_name in self._final_returns:
                current = results.get(f"eval/{task_name}/mean_return", 0.0)
                final = self._final_returns[task_name]
                forgetting = max(0.0, final - current)
                forgetting_values.append(forgetting)
                results[f"eval/{task_name}/forgetting"] = forgetting

        if forgetting_values:
            results["eval/average_forgetting"] = float(np.mean(forgetting_values))

        return results

    def _log_cl_metrics(self):
        if not self._task_envs:
            return

        print(
            f"  Evaluating on {len(self._task_envs)} task(s)...",
            flush=True,
        )
        metrics = self.evaluate_all_tasks(num_episodes=10)
        self.logger.log(metrics, step=self.total_steps)

        avg_return = metrics.get("eval/average_mean_return", 0.0)
        avg_forgetting = metrics.get("eval/average_forgetting", 0.0)
        print(
            f"  Avg Return: {avg_return:.2f}, Avg Forgetting: {avg_forgetting:.2f}",
            flush=True,
        )
