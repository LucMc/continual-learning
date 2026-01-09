"""Soft Actor-Critic (SAC) implementation for off-policy RL.

This module implements SAC with:
- Twin Q-networks for reduced overestimation
- Automatic entropy tuning
- Soft target updates
- Support for the custom optimizer pattern (returns params, not updates)
- Numerical stability improvements (gradient clipping, NaN handling)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.core.scope import DenyList
from jaxtyping import Array, Float, PRNGKeyArray

from continual_learning.configs.rl import SACConfig
from continual_learning.models import get_model_cls
from continual_learning.models.rl import TanhPolicy, QNetwork
from continual_learning.optim import get_optimizer
from continual_learning.types import LogDict
from continual_learning.utils.replay_buffer import ReplayBatch
from continual_learning.utils.training import TrainState


# Numerical stability constants
MAX_LOG_ALPHA = 2.0  # exp(2) ≈ 7.4
MIN_LOG_ALPHA = -10.0  # exp(-10) ≈ 0.00005
GRAD_CLIP_NORM = 1.0
MAX_Q_VALUE = 1e6


class SACTrainState(NamedTuple):
    """Training state for SAC algorithm."""

    actor: TrainState
    critic: TrainState
    target_critic_params: optax.Params
    log_alpha: Float[Array, ""]
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

        # Initialize entropy coefficient (log_alpha)
        log_alpha = jnp.log(cfg.alpha).astype(jnp.float32)
        alpha_optimizer = optax.adam(learning_rate=3e-4)
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
    @partial(jax.jit, static_argnames=("cfg",))
    def update_critic(
        state: SACTrainState,
        batch: ReplayBatch,
        cfg: SACConfig,
    ) -> tuple[SACTrainState, LogDict]:
        """Update critic (Q-networks) using TD learning.

        Args:
            state: Current training state
            batch: Batch of transitions from replay buffer
            cfg: SAC configuration

        Returns:
            Updated state and log dictionary
        """
        key, action_key = jax.random.split(state.key)
        alpha = jnp.exp(state.log_alpha)

        # Compute target Q-values
        # Sample actions from current policy for next states
        next_dist = state.actor.apply_fn(state.actor.params, batch.next_observations)
        next_actions = next_dist.sample(seed=action_key)

        # Clip sampled actions for numerical stability
        next_actions = jnp.clip(next_actions, -0.999, 0.999)

        next_log_probs = next_dist.log_prob(next_actions)
        # Clip log probs to avoid -inf
        next_log_probs = jnp.clip(next_log_probs, -100.0, 100.0)

        # Get target Q-values (minimum of two Q-networks)
        next_q1, next_q2 = state.critic.apply_fn(
            state.target_critic_params,
            batch.next_observations,
            next_actions,
        )
        next_q = jnp.minimum(next_q1, next_q2)
        # Clip Q-values for stability
        next_q = jnp.clip(next_q, -MAX_Q_VALUE, MAX_Q_VALUE)

        # Compute target with entropy bonus
        target_q = batch.rewards + cfg.gamma * (1 - batch.dones) * (
            next_q - alpha * next_log_probs[..., None]
        )
        # Clip target Q-values
        target_q = jnp.clip(target_q, -MAX_Q_VALUE, MAX_Q_VALUE)
        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            # Get current Q-values with activation collection for reset methods
            (q1, q2), intermediates = state.critic.apply_fn(
                critic_params,
                batch.observations,
                batch.actions,
                training=True,
                mutable=("activations",),
            )

            # MSE loss for both Q-networks
            q1_loss = jnp.mean(jnp.square(q1 - target_q))
            q2_loss = jnp.mean(jnp.square(q2 - target_q))
            total_loss = q1_loss + q2_loss

            return total_loss, (q1, q2, intermediates)

        (critic_loss, (q1, q2, intermediates)), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic.params)

        # Clip gradients for stability
        critic_grads = jax.tree.map(
            lambda g: jnp.clip(g, -GRAD_CLIP_NORM, GRAD_CLIP_NORM),
            critic_grads,
        )

        # Replace NaN gradients with zeros
        critic_grads = jax.tree.map(
            lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g),
            critic_grads,
        )

        # Extract features for reset methods
        critic_feats = intermediates.get("activations", {})

        # Update critic using custom optimizer pattern
        new_critic = state.critic.apply_gradients(grads=critic_grads, features=critic_feats)

        logs = {
            "critic/loss": critic_loss,
            "critic/q1_mean": q1.mean(),
            "critic/q2_mean": q2.mean(),
            "critic/target_q_mean": target_q.mean(),
        }

        new_state = SACTrainState(
            actor=state.actor,
            critic=new_critic,
            target_critic_params=state.target_critic_params,
            log_alpha=state.log_alpha,
            alpha_optimizer_state=state.alpha_optimizer_state,
            key=key,
        )

        return new_state, logs

    @staticmethod
    @partial(jax.jit, static_argnames=("cfg",))
    def update_actor(
        state: SACTrainState,
        batch: ReplayBatch,
        cfg: SACConfig,
    ) -> tuple[SACTrainState, LogDict]:
        """Update actor (policy) using reparameterization trick.

        Args:
            state: Current training state
            batch: Batch of transitions from replay buffer
            cfg: SAC configuration

        Returns:
            Updated state and log dictionary
        """
        key, action_key = jax.random.split(state.key)
        alpha = jnp.exp(state.log_alpha)

        def actor_loss_fn(actor_params):
            # Sample actions from policy with activation collection
            dist, intermediates = state.actor.apply_fn(
                actor_params,
                batch.observations,
                training=True,
                mutable=("activations",),
            )
            actions = dist.sample(seed=action_key)

            # Clip actions for numerical stability in log_prob
            actions = jnp.clip(actions, -0.999, 0.999)
            log_probs = dist.log_prob(actions)
            # Clip log probs
            log_probs = jnp.clip(log_probs, -100.0, 100.0)

            # Get Q-values for sampled actions
            q1, q2 = state.critic.apply_fn(
                state.critic.params,
                batch.observations,
                actions,
            )
            q_min = jnp.minimum(q1, q2)

            # Policy loss: maximize Q - alpha * log_prob
            actor_loss = jnp.mean(alpha * log_probs - q_min.squeeze(-1))

            return actor_loss, (log_probs, intermediates)

        (actor_loss, (log_probs, intermediates)), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor.params)

        # Clip gradients for stability
        actor_grads = jax.tree.map(
            lambda g: jnp.clip(g, -GRAD_CLIP_NORM, GRAD_CLIP_NORM),
            actor_grads,
        )

        # Replace NaN gradients with zeros
        actor_grads = jax.tree.map(
            lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g),
            actor_grads,
        )

        # Extract features for reset methods
        actor_feats = intermediates.get("activations", {})

        # Update actor using custom optimizer pattern
        new_actor = state.actor.apply_gradients(grads=actor_grads, features=actor_feats)

        logs = {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
        }

        new_state = SACTrainState(
            actor=new_actor,
            critic=state.critic,
            target_critic_params=state.target_critic_params,
            log_alpha=state.log_alpha,
            alpha_optimizer_state=state.alpha_optimizer_state,
            key=key,
        )

        return new_state, logs

    @staticmethod
    @partial(jax.jit, static_argnames=("target_entropy",))
    def update_alpha(
        state: SACTrainState,
        batch: ReplayBatch,
        target_entropy: float,
    ) -> tuple[SACTrainState, LogDict]:
        """Update entropy coefficient (alpha) for automatic entropy tuning.

        Args:
            state: Current training state
            batch: Batch of transitions from replay buffer
            target_entropy: Target entropy (typically -action_dim)

        Returns:
            Updated state and log dictionary
        """
        key, action_key = jax.random.split(state.key)

        # Get log probs from current policy
        dist = state.actor.apply_fn(state.actor.params, batch.observations)
        actions = dist.sample(seed=action_key)

        # Clip actions for numerical stability
        actions = jnp.clip(actions, -0.999, 0.999)
        log_probs = dist.log_prob(actions)
        # Clip log probs
        log_probs = jnp.clip(log_probs, -100.0, 100.0)

        def alpha_loss_fn(log_alpha: Float[Array, ""]) -> Float[Array, ""]:
            alpha = jnp.exp(log_alpha)
            return -jnp.mean(alpha * (log_probs + target_entropy))

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state.log_alpha)

        # Clip alpha gradients
        alpha_grads = jnp.clip(alpha_grads, -GRAD_CLIP_NORM, GRAD_CLIP_NORM)
        alpha_grads = jnp.where(jnp.isnan(alpha_grads), 0.0, alpha_grads)

        # Standard optax update for alpha (not using custom pattern)
        alpha_optimizer = optax.adam(learning_rate=3e-4)
        updates, new_alpha_opt_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state
        )
        new_log_alpha = jnp.asarray(optax.apply_updates(state.log_alpha, updates))

        # Clip log_alpha to reasonable bounds
        new_log_alpha = jnp.clip(new_log_alpha, MIN_LOG_ALPHA, MAX_LOG_ALPHA)

        logs = {
            "alpha/loss": alpha_loss,
            "alpha/value": jnp.exp(new_log_alpha),
        }

        new_state = SACTrainState(
            actor=state.actor,
            critic=state.critic,
            target_critic_params=state.target_critic_params,
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt_state,
            key=key,
        )

        return new_state, logs

    @staticmethod
    @jax.jit
    def soft_update_target(
        state: SACTrainState,
        tau: float,
    ) -> SACTrainState:
        """Perform soft update of target critic parameters.

        Args:
            state: Current training state
            tau: Soft update coefficient (0 < tau <= 1)

        Returns:
            Updated state with new target parameters
        """
        new_target_params = jax.tree.map(
            lambda target, source: tau * source + (1 - tau) * target,
            state.target_critic_params,
            state.critic.params,
        )

        return SACTrainState(
            actor=state.actor,
            critic=state.critic,
            target_critic_params=new_target_params,
            log_alpha=state.log_alpha,
            alpha_optimizer_state=state.alpha_optimizer_state,
            key=state.key,
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

        This includes:
        1. Critic update (twin Q-networks)
        2. Actor update (policy)
        3. Alpha update (entropy coefficient, if auto-tuning)
        4. Soft target update

        Args:
            state: Current training state
            batch: Batch of transitions from replay buffer
            cfg: SAC configuration
            target_entropy: Target entropy for auto-tuning

        Returns:
            Updated state and combined log dictionary
        """
        # Update critic
        state, critic_logs = SAC.update_critic(state, batch, cfg)

        # Update actor
        state, actor_logs = SAC.update_actor(state, batch, cfg)

        # Update alpha (entropy coefficient)
        alpha_logs = {}
        if cfg.auto_entropy:
            state, alpha_logs = SAC.update_alpha(state, batch, target_entropy)

        # Soft update target networks
        state = SAC.soft_update_target(state, cfg.tau)

        logs = {**critic_logs, **actor_logs, **alpha_logs}
        return state, logs
