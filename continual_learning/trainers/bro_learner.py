"""BRO (Bigger, Regularized, Optimistic) Learner.

This module implements the full BRO algorithm as described in:
Nauman et al. 2024 - "Bigger, Regularized, Optimistic: scaling for compute
and sample-efficient continuous control"

Key components:
- Distributional critic with quantile regression
- Conservative + Optimistic dual actor system
- Learnable temperature, optimism, and regularizer coefficients
- BroNet architecture with LayerNorm and residual connections
- Fixed reset schedule for high replay ratio training
- Support for reset methods (REDO, ReGrAMA, CBP, CCBP, ShrinkAndPerturb)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray

from continual_learning.configs.optim import (
    OptimizerConfig,
    ResetMethodConfig,
    AdamwConfig,
    AdamConfig,
)
from continual_learning.models.rl import (
    BRONormalTanhPolicy,
    BRODualTanhPolicy,
    BRODistributionalCritic,
    Temperature,
    Adjustment,
    orthogonal_init,
)
from continual_learning.optim import get_optimizer
from continual_learning.types import LogDict
from continual_learning.utils.training import TrainState


class BROConfig(NamedTuple):
    """Configuration for BRO algorithm.

    Supports pluggable optimizers including reset methods (REDO, ReGrAMA, CBP, CCBP, ShrinkAndPerturb).
    """
    # Optimizer configs (supports reset methods)
    actor_optimizer: OptimizerConfig | ResetMethodConfig = AdamwConfig(learning_rate=3e-4)
    critic_optimizer: OptimizerConfig | ResetMethodConfig = AdamwConfig(learning_rate=3e-4)

    # Learning rates for scalar params (no reset methods needed)
    temp_lr: float = 3e-4
    adj_lr: float = 3e-5  # Lower LR for adjustment coefficients

    # SAC parameters
    discount: float = 0.99
    tau: float = 0.005

    # BRO-specific parameters
    init_temperature: float = 1.0
    init_optimism: float = 1.0
    init_regularizer: float = 0.25
    pessimism: float = 0.0
    kl_target: float = 0.05
    std_multiplier: float = 0.75

    # Distributional RL
    distributional: bool = True
    n_quantiles: int = 100

    # Network architecture
    hidden_dims: int = 256
    depth: int = 1

    # Training
    updates_per_step: int = 10
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5000

    # Reset schedule (steps at which to reset networks)
    # This is BRO's built-in reset mechanism, complementary to optimizer reset methods
    reset_steps: tuple[int, ...] = (15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001)


class BROState(NamedTuple):
    """Training state for BRO algorithm.

    Uses TrainState for networks (supports reset methods).
    Uses raw params/opt_state for scalar learnable coefficients (no reset needed).
    """
    # Networks using TrainState (supports reset methods)
    actor: TrainState
    actor_o: TrainState
    critic: TrainState
    target_critic_params: FrozenDict

    # Learnable coefficients (keep as raw params - no reset needed for scalars)
    temp_params: FrozenDict
    temp_opt_state: optax.OptState
    optimism_params: FrozenDict
    optimism_opt_state: optax.OptState
    regularizer_params: FrozenDict
    regularizer_opt_state: optax.OptState

    # Random key
    key: PRNGKeyArray

    # Step counter
    step: int


def tree_norm(tree):
    """Compute L2 norm of a pytree."""
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))


def clip_grads(grads, max_norm: float = 1.0):
    """Clip gradients by global norm."""
    grad_norm = tree_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads), grad_norm


def flatten_activations(features: dict) -> dict:
    """Flatten nested activation dict for reset methods.

    BRO networks use nested modules (BroNet_0, BroNet_1), which creates
    nested activation dicts like {"BroNet_0": {"Dense_0_act": ...}}.
    Reset methods expect flat dicts with keys matching param paths.

    Always include full path to match param structure (e.g., "BroNet_0/Dense_0_act"
    maps to param path ("BroNet_0", "Dense_0", "kernel")).
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


def huber_loss(td_errors: jax.Array, kappa: float = 1.0) -> jax.Array:
    """Huber loss for quantile regression."""
    return jnp.where(
        jnp.abs(td_errors) <= kappa,
        0.5 * td_errors ** 2,
        kappa * (jnp.abs(td_errors) - 0.5 * kappa)
    )


def quantile_huber_loss(td_errors: jax.Array, taus: jax.Array, kappa: float = 1.0) -> jax.Array:
    """Quantile Huber loss for distributional RL."""
    element_wise_huber = huber_loss(td_errors, kappa)
    mask = jax.lax.stop_gradient(jnp.where(td_errors < 0, 1.0, 0.0))
    element_wise_quantile_huber = jnp.abs(taus[..., None] - mask) * element_wise_huber / kappa
    return element_wise_quantile_huber.sum(axis=1).mean()


class BROLearner:
    """BRO algorithm learner."""

    def __init__(
        self,
        seed: int,
        obs_dim: int,
        action_dim: int,
        cfg: BROConfig,
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.target_entropy = -action_dim / 2  # BRO uses -action_dim / 2

        # Quantile taus for distributional critic
        quantile_taus = jnp.arange(0, cfg.n_quantiles + 1) / cfg.n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)

        # Initialize state
        self.state = self._init_state(seed, obs_dim, action_dim)

        # Store initial params for reset
        self._init_seed = seed

        # Adjust reset schedule for lower updates_per_step
        if cfg.updates_per_step == 2:
            self.reset_steps = cfg.reset_steps[:1]
        else:
            self.reset_steps = cfg.reset_steps

    def _init_state(self, seed: int, obs_dim: int, action_dim: int) -> BROState:
        """Initialize BRO training state with TrainState for reset method support."""
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 8)
        (key, actor_key, actor_o_key, critic_key,
         temp_key, optimism_key, regularizer_key, rng_key) = keys

        cfg = self.cfg
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        # Initialize conservative actor with TrainState
        actor_module = BRONormalTanhPolicy(
            action_dim=action_dim,
            hidden_dims=cfg.hidden_dims,
            depth=cfg.depth,
        )
        actor_params = actor_module.init(actor_key, dummy_obs)
        actor = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=get_optimizer(cfg.actor_optimizer),
            kernel_init=orthogonal_init(),
            bias_init=jax.nn.initializers.zeros,
        )

        # Initialize optimistic actor with TrainState
        actor_o_module = BRODualTanhPolicy(
            action_dim=action_dim,
            hidden_dims=cfg.hidden_dims,
            depth=cfg.depth,
        )
        actor_o_params = actor_o_module.init(
            actor_o_key, dummy_obs, dummy_action, dummy_action, cfg.std_multiplier
        )
        actor_o = TrainState.create(
            apply_fn=actor_o_module.apply,
            params=actor_o_params,
            tx=get_optimizer(cfg.actor_optimizer),
            kernel_init=orthogonal_init(),
            bias_init=jax.nn.initializers.zeros,
        )

        # Initialize distributional critic with TrainState
        n_quantiles = cfg.n_quantiles if cfg.distributional else 1
        critic_module = BRODistributionalCritic(
            n_quantiles=n_quantiles,
            hidden_dims=cfg.hidden_dims,
            depth=cfg.depth,
        )
        critic_params = critic_module.init(critic_key, dummy_obs, dummy_action)
        critic = TrainState.create(
            apply_fn=critic_module.apply,
            params=critic_params,
            tx=get_optimizer(cfg.critic_optimizer),
            kernel_init=orthogonal_init(),
            bias_init=jax.nn.initializers.zeros,
        )
        target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        # Initialize temperature (scalar param - no reset method needed)
        temp = Temperature(initial_temperature=cfg.init_temperature)
        temp_params = temp.init(temp_key)
        temp_tx = optax.adam(learning_rate=cfg.temp_lr, b1=0.5)
        temp_opt_state = temp_tx.init(temp_params)

        # Calculate init values for bounded adjustments
        log_val_min, log_val_max = -10.0, 7.5

        def calc_init_value(init_val):
            return np.exp(np.arctanh(
                (np.log(init_val) - log_val_min) / ((log_val_max - log_val_min) * 0.5) - 1
            ))

        # Initialize optimism (scalar param - no reset method needed)
        optimism = Adjustment(
            init_value=calc_init_value(cfg.init_optimism),
            log_val_min=log_val_min,
            log_val_max=log_val_max,
        )
        optimism_params = optimism.init(optimism_key)
        optimism_tx = optax.adam(learning_rate=cfg.adj_lr, b1=0.5)
        optimism_opt_state = optimism_tx.init(optimism_params)

        # Initialize regularizer (scalar param - no reset method needed)
        regularizer = Adjustment(
            init_value=calc_init_value(cfg.init_regularizer),
            log_val_min=log_val_min,
            log_val_max=log_val_max,
        )
        regularizer_params = regularizer.init(regularizer_key)
        regularizer_tx = optax.adam(learning_rate=cfg.adj_lr, b1=0.5)
        regularizer_opt_state = regularizer_tx.init(regularizer_params)

        return BROState(
            actor=actor,
            actor_o=actor_o,
            critic=critic,
            target_critic_params=target_critic_params,
            temp_params=temp_params,
            temp_opt_state=temp_opt_state,
            optimism_params=optimism_params,
            optimism_opt_state=optimism_opt_state,
            regularizer_params=regularizer_params,
            regularizer_opt_state=regularizer_opt_state,
            key=rng_key,
            step=1,
        )

    def reset(self):
        """Reset all networks to initial state."""
        self.state = self._init_state(self._init_seed, self.obs_dim, self.action_dim)

    def sample_actions(
        self,
        observations: jax.Array,
        temperature: float = 1.0,
        use_optimistic: bool = True,
    ) -> jax.Array:
        """Sample actions from policy.

        Args:
            observations: Current observations
            temperature: Sampling temperature
            use_optimistic: If True, use optimistic actor for exploration

        Returns:
            Sampled actions clipped to [-1, 1]
        """
        key, action_key = jax.random.split(self.state.key)

        if use_optimistic:
            # Get conservative actor's mean/std
            _, mu_c, std_c = self.state.actor.apply_fn(
                self.state.actor.params,
                observations,
                temperature=temperature,
                return_params=True,
            )

            # Sample from optimistic actor
            dist, _, _ = self.state.actor_o.apply_fn(
                self.state.actor_o.params,
                observations,
                mu_c,
                std_c,
                self.cfg.std_multiplier,
            )
            actions = dist.sample(seed=action_key)
        else:
            # Sample from conservative actor
            dist = self.state.actor.apply_fn(
                self.state.actor.params,
                observations,
                temperature=temperature,
            )
            actions = dist.sample(seed=action_key)

        # Update key
        self.state = self.state._replace(key=key)

        return jnp.clip(actions, -1.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def _update_critic(
        self,
        state: BROState,
        observations: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_observations: jax.Array,
        dones: jax.Array,
    ) -> tuple[BROState, LogDict]:
        """Update distributional critic with activation collection for reset methods."""
        cfg = self.cfg
        key, action_key = jax.random.split(state.key)

        # Get temperature
        temperature = Temperature(initial_temperature=cfg.init_temperature).apply(state.temp_params)

        # Sample next actions from conservative actor
        next_dist = state.actor.apply_fn(state.actor.params, next_observations)
        next_actions = next_dist.sample(seed=action_key)
        next_log_probs = next_dist.log_prob(next_actions)

        # Get target Q-values
        next_q1, next_q2 = state.critic.apply_fn(state.target_critic_params, next_observations, next_actions)

        # Pessimism-weighted combination
        next_q = (next_q1 + next_q2) / 2 - cfg.pessimism * jnp.abs(next_q1 - next_q2) / 2

        if cfg.distributional:
            # Distributional target
            target_q = rewards[..., None, None] + cfg.discount * (1 - dones)[..., None, None] * next_q[:, None, :]
            target_q = target_q - cfg.discount * temperature * (1 - dones)[..., None, None] * next_log_probs[..., None, None]
        else:
            target_q = rewards + cfg.discount * (1 - dones) * next_q
            target_q = target_q - cfg.discount * temperature * (1 - dones) * next_log_probs[..., None]

        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            # Collect activations for reset methods
            (q1, q2), intermediates = state.critic.apply_fn(
                critic_params, observations, actions,
                training=True,
                mutable=("activations",),
            )

            if cfg.distributional:
                td_errors1 = target_q - q1[..., None]
                td_errors2 = target_q - q2[..., None]
                loss = (
                    quantile_huber_loss(td_errors1, self.quantile_taus) +
                    quantile_huber_loss(td_errors2, self.quantile_taus)
                )
            else:
                loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            return loss, {"q1": q1.mean(), "q2": q2.mean(), "intermediates": intermediates}

        (critic_loss, aux), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic.params)

        # Clip gradients for stability
        grads, grad_norm = clip_grads(grads, max_norm=1.0)

        # Get activation features for reset methods
        features = flatten_activations(aux["intermediates"].get("activations", {}))

        # Update critic using TrainState (supports reset methods)
        new_critic = state.critic.apply_gradients(grads=grads, features=features)

        logs = {
            "critic/loss": critic_loss,
            "critic/q1_mean": aux["q1"],
            "critic/q2_mean": aux["q2"],
            "critic/grad_norm": grad_norm,
        }

        new_state = state._replace(
            critic=new_critic,
            key=key,
        )

        return new_state, logs

    @partial(jax.jit, static_argnums=(0,))
    def _update_actor(
        self,
        state: BROState,
        observations: jax.Array,
    ) -> tuple[BROState, LogDict]:
        """Update conservative actor with activation collection for reset methods."""
        cfg = self.cfg
        key, action_key = jax.random.split(state.key)

        temperature = Temperature(initial_temperature=cfg.init_temperature).apply(state.temp_params)

        def actor_loss_fn(actor_params):
            # Collect activations for reset methods
            (dist, mu, std), intermediates = state.actor.apply_fn(
                actor_params, observations,
                return_params=True,
                training=True,
                mutable=("activations",),
            )

            actions = dist.sample(seed=action_key)
            log_probs = dist.log_prob(actions)

            # Get Q-values (no gradient through critic)
            q1, q2 = state.critic.apply_fn(state.critic.params, observations, actions)

            # Pessimism-weighted Q
            q = (q1 + q2) / 2 - cfg.pessimism * jnp.abs(q1 - q2) / 2
            if cfg.distributional:
                q = q.mean(-1)

            actor_loss = (log_probs * temperature - q).mean()

            return actor_loss, {"entropy": -log_probs.mean(), "std": std.mean(), "intermediates": intermediates}

        (actor_loss, aux), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)

        # Clip gradients for stability
        grads, grad_norm = clip_grads(grads, max_norm=1.0)

        # Get activation features for reset methods
        features = flatten_activations(aux["intermediates"].get("activations", {}))

        # Update actor using TrainState (supports reset methods)
        new_actor = state.actor.apply_gradients(grads=grads, features=features)

        logs = {
            "actor/loss": actor_loss,
            "actor/entropy": aux["entropy"],
            "actor/std": aux["std"],
            "actor/grad_norm": grad_norm,
        }

        new_state = state._replace(
            actor=new_actor,
            key=key,
        )

        return new_state, logs

    @partial(jax.jit, static_argnums=(0,))
    def _update_actor_optimistic(
        self,
        state: BROState,
        observations: jax.Array,
    ) -> tuple[BROState, LogDict]:
        """Update optimistic actor with activation collection for reset methods."""
        cfg = self.cfg
        key, action_key = jax.random.split(state.key)

        optimism = Adjustment(
            init_value=1.0,
            log_val_min=-10.0,
            log_val_max=7.5,
        ).apply(state.optimism_params)

        regularizer = Adjustment(
            init_value=1.0,
            log_val_min=-10.0,
            log_val_max=7.5,
        ).apply(state.regularizer_params)

        def actor_o_loss_fn(actor_o_params):
            # Get conservative actor's mean/std (no gradient through this)
            _, mu_c, std_c = state.actor.apply_fn(
                state.actor.params, observations, return_params=True
            )

            # Get optimistic actor's distribution with activation collection
            (dist, mu_o, std_o), intermediates = state.actor_o.apply_fn(
                actor_o_params,
                observations,
                mu_c,
                std_c,
                cfg.std_multiplier,
                training=True,
                mutable=("activations",),
            )

            actions = dist.sample(seed=action_key)

            # Get Q-values (no gradient through critic)
            q1, q2 = state.critic.apply_fn(state.critic.params, observations, actions)

            # KL divergence between optimistic and conservative
            std_o_scaled = std_o / cfg.std_multiplier
            kl = (
                jnp.log(std_c / std_o_scaled) +
                (std_o_scaled ** 2 + (mu_o - mu_c) ** 2) / (2 * std_c ** 2) -
                0.5
            ).sum(-1)

            # Optimism-weighted upper bound on Q
            q_ub = (q1 + q2) / 2 + optimism * jnp.abs(q1 - q2) / 2
            if cfg.distributional:
                q_ub = q_ub.mean(-1)

            # Loss: maximize Q upper bound with KL regularization
            actor_o_loss = (-q_ub).mean() + regularizer * kl.mean()

            return actor_o_loss, {
                "kl": kl.mean(),
                "std_c": std_c.mean(),
                "std_o": std_o.mean(),
                "q_mean": ((q1 + q2) / 2).mean(),
                "q_std": (jnp.abs(q1 - q2) / 2).mean(),
                "intermediates": intermediates,
            }

        (actor_o_loss, aux), grads = jax.value_and_grad(actor_o_loss_fn, has_aux=True)(state.actor_o.params)

        # Clip gradients for stability
        grads, grad_norm = clip_grads(grads, max_norm=1.0)

        # Get activation features for reset methods
        features = flatten_activations(aux["intermediates"].get("activations", {}))

        # Update optimistic actor using TrainState (supports reset methods)
        new_actor_o = state.actor_o.apply_gradients(grads=grads, features=features)

        logs = {
            "actor_o/loss": actor_o_loss,
            "actor_o/kl": aux["kl"],
            "actor_o/std_c": aux["std_c"],
            "actor_o/std_o": aux["std_o"],
            "actor_o/q_mean": aux["q_mean"],
            "actor_o/q_std": aux["q_std"],
            "actor_o/grad_norm": grad_norm,
        }

        new_state = state._replace(
            actor_o=new_actor_o,
            key=key,
        )

        return new_state, logs

    @partial(jax.jit, static_argnums=(0,))
    def _update_temperature(
        self,
        state: BROState,
        entropy: float,
    ) -> tuple[BROState, LogDict]:
        """Update temperature coefficient."""
        cfg = self.cfg

        def temp_loss_fn(temp_params):
            temperature = Temperature(initial_temperature=cfg.init_temperature).apply(temp_params)
            return temperature * (entropy - self.target_entropy), {"temperature": temperature}

        (temp_loss, aux), grads = jax.value_and_grad(temp_loss_fn, has_aux=True)(state.temp_params)

        temp_tx = optax.adam(learning_rate=cfg.temp_lr, b1=0.5)
        updates, new_temp_opt_state = temp_tx.update(grads, state.temp_opt_state, state.temp_params)
        new_temp_params = optax.apply_updates(state.temp_params, updates)

        logs = {
            "temperature/loss": temp_loss,
            "temperature/value": aux["temperature"],
        }

        new_state = state._replace(
            temp_params=new_temp_params,
            temp_opt_state=new_temp_opt_state,
        )

        return new_state, logs

    @partial(jax.jit, static_argnums=(0,))
    def _update_optimism(
        self,
        state: BROState,
        empirical_kl: float,
    ) -> tuple[BROState, LogDict]:
        """Update optimism coefficient."""
        cfg = self.cfg

        def optimism_loss_fn(optimism_params):
            optimism = Adjustment(
                init_value=1.0,
                log_val_min=-10.0,
                log_val_max=7.5,
            ).apply(optimism_params)
            # Increase optimism if KL is below target, decrease if above
            return (optimism - cfg.pessimism) * (empirical_kl - cfg.kl_target), {"optimism": optimism}

        (optimism_loss, aux), grads = jax.value_and_grad(optimism_loss_fn, has_aux=True)(state.optimism_params)

        optimism_tx = optax.adam(learning_rate=cfg.adj_lr, b1=0.5)
        updates, new_optimism_opt_state = optimism_tx.update(grads, state.optimism_opt_state, state.optimism_params)
        new_optimism_params = optax.apply_updates(state.optimism_params, updates)

        logs = {
            "optimism/loss": optimism_loss,
            "optimism/value": aux["optimism"],
        }

        new_state = state._replace(
            optimism_params=new_optimism_params,
            optimism_opt_state=new_optimism_opt_state,
        )

        return new_state, logs

    @partial(jax.jit, static_argnums=(0,))
    def _update_regularizer(
        self,
        state: BROState,
        empirical_kl: float,
    ) -> tuple[BROState, LogDict]:
        """Update regularizer (KL weight) coefficient."""
        cfg = self.cfg

        def regularizer_loss_fn(regularizer_params):
            kl_weight = Adjustment(
                init_value=1.0,
                log_val_min=-10.0,
                log_val_max=7.5,
            ).apply(regularizer_params)
            # Increase regularizer if KL is above target
            return -kl_weight * (empirical_kl - cfg.kl_target), {"kl_weight": kl_weight}

        (regularizer_loss, aux), grads = jax.value_and_grad(regularizer_loss_fn, has_aux=True)(state.regularizer_params)

        regularizer_tx = optax.adam(learning_rate=cfg.adj_lr, b1=0.5)
        updates, new_regularizer_opt_state = regularizer_tx.update(grads, state.regularizer_opt_state, state.regularizer_params)
        new_regularizer_params = optax.apply_updates(state.regularizer_params, updates)

        logs = {
            "regularizer/loss": regularizer_loss,
            "regularizer/kl_weight": aux["kl_weight"],
        }

        new_state = state._replace(
            regularizer_params=new_regularizer_params,
            regularizer_opt_state=new_regularizer_opt_state,
        )

        return new_state, logs

    @partial(jax.jit, static_argnums=(0,))
    def _soft_update_target(self, state: BROState) -> BROState:
        """Soft update of target critic parameters."""
        tau = self.cfg.tau
        new_target_params = jax.tree.map(
            lambda t, s: tau * s + (1 - tau) * t,
            state.target_critic_params,
            state.critic.params,
        )
        return state._replace(target_critic_params=new_target_params)

    def update(
        self,
        observations: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_observations: jax.Array,
        dones: jax.Array,
        env_step: int,
    ) -> LogDict:
        """Perform a full BRO update step.

        Args:
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags
            env_step: Current environment step (for reset schedule)

        Returns:
            Dictionary of logged metrics
        """
        # Check for reset
        if env_step in self.reset_steps:
            print(f"[BRO] Resetting networks at step {env_step}", flush=True)
            self.reset()

        all_logs: LogDict = {}

        # Update critic
        self.state, critic_logs = self._update_critic(
            self.state, observations, actions, rewards, next_observations, dones
        )
        all_logs.update(critic_logs)

        # Soft update target
        self.state = self._soft_update_target(self.state)

        # Update conservative actor
        self.state, actor_logs = self._update_actor(self.state, observations)
        all_logs.update(actor_logs)

        # Update optimistic actor
        self.state, actor_o_logs = self._update_actor_optimistic(self.state, observations)
        all_logs.update(actor_o_logs)

        # Update temperature
        self.state, temp_logs = self._update_temperature(self.state, actor_logs["actor/entropy"])
        all_logs.update(temp_logs)

        # Update optimism and regularizer based on KL
        empirical_kl = actor_o_logs["actor_o/kl"] / self.action_dim
        self.state, optimism_logs = self._update_optimism(self.state, empirical_kl)
        all_logs.update(optimism_logs)

        self.state, regularizer_logs = self._update_regularizer(self.state, empirical_kl)
        all_logs.update(regularizer_logs)

        # Increment step
        self.state = self.state._replace(step=self.state.step + 1)

        return all_logs
