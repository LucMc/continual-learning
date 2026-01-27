"""Soft Actor-Critic (SAC) implementation for off-policy RL.

Adapted from https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py

This module implements SAC with:
- Twin Q-networks for reduced overestimation
- Automatic entropy tuning
- Soft target updates
- Support for the custom optimizer pattern (returns params, not updates)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import optax
from flax.core.scope import DenyList
from jaxtyping import Array, Float, PRNGKeyArray

from continual_learning.configs.rl import SACConfig
from continual_learning.models import get_model_cls
from continual_learning.models.rl import QNetwork, TanhPolicy
from continual_learning.optim import get_optimizer
from continual_learning.types import LogDict
from continual_learning.utils.replay_buffer import ReplayBatch
from continual_learning.utils.training import TrainState


def flatten_activations(features: dict) -> dict:
    """Flatten nested activation dict for reset methods.

    SAC networks use nested modules (q1/q2, main), which creates
    nested activation dicts like {"q1": {"main": {"layer_0_act": ...}}}.
    Reset methods expect flat dicts like {"layer_0_act": ...}.

    For twin networks (critic with Q1/Q2), we prefix with module name
    to avoid collisions.
    """
    flat = {}

    def _flatten(d: dict, prefix: str = "") -> None:
        for key, value in d.items():
            new_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                flat[new_key] = value

    _flatten(features)
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

        Matches the reference implementation's update order:
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
