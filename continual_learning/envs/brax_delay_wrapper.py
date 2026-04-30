# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""JIT-compatible Brax wrapper modelling random observation/action delays.

Sits OUTSIDE ``brax.envs.training.wrap(...)`` so it operates on already-batched,
already-auto-reset state. Per-task delay sub-intervals are passed at
construction time; the wrapper samples the realised per-step delay uniformly
from this interval inside JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from brax.envs.base import State, Wrapper

from continual_learning.envs.base import JittableVectorEnv, Timestep
from continual_learning.types import Action, EnvState, Observation


class BraxRandomDelayWrapper(Wrapper):
    """Random observation + action delay wrapper for vectorised Brax envs.

    Parameters
    ----------
    env:
        An env that has already been wrapped with
        ``brax.envs.training.wrap`` (so VmapWrapper/EpisodeWrapper/AutoResetWrapper
        are inside this wrapper).
    num_envs:
        Number of parallel environments — the leading batch dim of ``state``.
    obs_delay_range, act_delay_range:
        Per-task sub-intervals (Python ``range`` objects, exclusive stop).
        ``alpha = jax.random.randint(..., obs_delay_range.start, obs_delay_range.stop)``
        each step.
    overall_obs_delay_max, overall_act_delay_max:
        Exclusive upper bounds for the obs/action delay one-hots and buffer
        sizes — kept constant across tasks so the augmented obs dim is stable.
    base_obs_dim, action_dim:
        Dimensions of the underlying env's observation and action.
    seed:
        Seed for the wrapper's per-step delay-sampling RNG (independent of
        the env reset RNG).
    """

    def __init__(
        self,
        env: Wrapper,
        *,
        num_envs: int,
        obs_delay_range: range,
        act_delay_range: range,
        overall_obs_delay_max: int,
        overall_act_delay_max: int,
        base_obs_dim: int,
        action_dim: int,
        seed: int,
        action_clip: float | None = 5.0,
    ):
        super().__init__(env)
        if obs_delay_range.start < 0 or obs_delay_range.stop > overall_obs_delay_max:
            raise ValueError(
                f"obs_delay_range {obs_delay_range} must lie inside "
                f"[0, {overall_obs_delay_max})"
            )
        if act_delay_range.start < 0 or act_delay_range.stop > overall_act_delay_max:
            raise ValueError(
                f"act_delay_range {act_delay_range} must lie inside "
                f"[0, {overall_act_delay_max})"
            )

        self.num_envs = num_envs
        self.obs_delay_min = int(obs_delay_range.start)
        self.obs_delay_max = int(obs_delay_range.stop)
        self.act_delay_min = int(act_delay_range.start)
        self.act_delay_max = int(act_delay_range.stop)
        self.overall_obs_delay_max = int(overall_obs_delay_max)
        self.overall_act_delay_max = int(overall_act_delay_max)
        self.base_obs_dim = int(base_obs_dim)
        self.action_dim = int(action_dim)
        # Buffers must accommodate the worst-case delay across all tasks.
        self.obs_buf_size = max(self.overall_obs_delay_max, 1)
        self.act_buf_size = max(self.overall_act_delay_max, 1)
        self._seed = int(seed)
        # Loose clip on the policy action before storing/applying. Breaks the
        # action-history feedback loop in the augmented obs (huge action →
        # huge act_buffer entry → huge obs to policy → larger action) without
        # materially restricting normal training, where a Gaussian policy with
        # std≈1 essentially never samples beyond a few sigma. ``None`` disables
        # clipping entirely.
        self.action_clip = None if action_clip is None else float(action_clip)

    @property
    def observation_size(self) -> int:
        return (
            self.base_obs_dim
            + self.act_buf_size * self.action_dim
            + self.overall_obs_delay_max
            + self.overall_act_delay_max
        )

    def _sample_alpha_kappa(self, alpha_keys: jax.Array, kappa_keys: jax.Array):
        alphas = jax.vmap(
            lambda k: jax.random.randint(k, (), self.obs_delay_min, self.obs_delay_max)
        )(alpha_keys)
        kappas = jax.vmap(
            lambda k: jax.random.randint(k, (), self.act_delay_min, self.act_delay_max)
        )(kappa_keys)
        return alphas, kappas

    def _build_augmented_obs(
        self,
        obs_buffer: jax.Array,
        act_buffer: jax.Array,
        alphas: jax.Array,
        kappas: jax.Array,
    ) -> jax.Array:
        obs_indices = self.obs_buf_size - 1 - alphas
        delayed_obs = jnp.take_along_axis(
            obs_buffer, obs_indices[:, None, None], axis=1
        ).squeeze(1)
        act_buffer_flat = act_buffer.reshape(self.num_envs, -1)
        alpha_one_hot = jax.nn.one_hot(alphas, self.overall_obs_delay_max)
        kappa_one_hot = jax.nn.one_hot(kappas, self.overall_act_delay_max)
        return jnp.concatenate(
            [delayed_obs, act_buffer_flat, alpha_one_hot, kappa_one_hot], axis=-1
        )

    def _interval_arrays(self) -> tuple[jax.Array, jax.Array]:
        obs_lohi = jnp.broadcast_to(
            jnp.array([self.obs_delay_min, self.obs_delay_max - 1], dtype=jnp.int32),
            (self.num_envs, 2),
        )
        act_lohi = jnp.broadcast_to(
            jnp.array([self.act_delay_min, self.act_delay_max - 1], dtype=jnp.int32),
            (self.num_envs, 2),
        )
        return obs_lohi, act_lohi

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        raw_obs = state.obs

        obs_buffer = jnp.tile(raw_obs[:, None, :], (1, self.obs_buf_size, 1))
        act_buffer = jnp.zeros(
            (self.num_envs, self.act_buf_size, self.action_dim), dtype=jnp.float32
        )

        per_env_master = jax.random.split(jax.random.PRNGKey(self._seed), self.num_envs)
        keys = jax.vmap(lambda k: jax.random.split(k, 3))(per_env_master)
        next_rng = keys[:, 0]
        alpha_keys = keys[:, 1]
        kappa_keys = keys[:, 2]
        alphas, kappas = self._sample_alpha_kappa(alpha_keys, kappa_keys)

        augmented_obs = self._build_augmented_obs(obs_buffer, act_buffer, alphas, kappas)

        obs_lohi, act_lohi = self._interval_arrays()
        new_info = dict(state.info)
        new_info["delay_obs_buffer"] = obs_buffer
        new_info["delay_act_buffer"] = act_buffer
        new_info["delay_rng"] = next_rng
        new_info["realised_obs_delay"] = alphas
        new_info["realised_act_delay"] = kappas
        new_info["current_obs_interval"] = obs_lohi
        new_info["current_act_interval"] = act_lohi
        return state.replace(obs=augmented_obs, info=new_info)

    def step(self, state: State, action: jax.Array) -> State:
        if self.action_clip is not None:
            action = jnp.clip(action, -self.action_clip, self.action_clip)

        # Reset our buffers BEFORE pushing the current action. The reset
        # signal is the previous step's done flag (preserved by
        # EpisodeWrapper as info["episode_done"]). On the step that follows
        # an episode end, we want fresh buffers (zeroed actions, first_obs
        # tiled into the obs buffer) — otherwise the agent's first action
        # of a new episode would be zeroed when we belatedly clear stale
        # entries.
        just_reset = state.info["episode_done"].astype(bool)
        first_obs = state.info["first_obs"]
        obs_buffer_in = state.info["delay_obs_buffer"]
        act_buffer_in = state.info["delay_act_buffer"]
        obs_buffer_reset = jnp.tile(first_obs[:, None, :], (1, self.obs_buf_size, 1))
        obs_buffer_pre_push = jnp.where(
            just_reset[:, None, None], obs_buffer_reset, obs_buffer_in
        )
        act_buffer_pre_push = jnp.where(
            just_reset[:, None, None], jnp.zeros_like(act_buffer_in), act_buffer_in
        )

        # Push current action onto the right end of the buffer.
        act_buffer = jnp.concatenate(
            [act_buffer_pre_push[:, 1:, :], action[:, None, :]], axis=1
        )

        per_env_rng = state.info["delay_rng"]
        keys = jax.vmap(lambda k: jax.random.split(k, 3))(per_env_rng)
        next_rng = keys[:, 0]
        alpha_keys = keys[:, 1]
        kappa_keys = keys[:, 2]
        alphas, kappas = self._sample_alpha_kappa(alpha_keys, kappa_keys)

        kappa_indices = self.act_buf_size - 1 - kappas
        delayed_action = jnp.take_along_axis(
            act_buffer, kappa_indices[:, None, None], axis=1
        ).squeeze(1)

        # Replace state.obs with the *raw* current obs before stepping the
        # inner wrap chain. EpisodeWrapper.step uses jax.lax.scan whose carry
        # must have stable shapes, so the obs handed in must match what
        # Ant.step produces (raw, shape (num_envs, base_obs_dim)). Our
        # augmented obs (with action history + one-hots) is restored on the
        # way out.
        raw_current_obs = obs_buffer_pre_push[:, -1, :]
        state_for_inner = state.replace(obs=raw_current_obs)
        next_state = self.env.step(state_for_inner, delayed_action)
        raw_next_obs = next_state.obs

        obs_buffer = jnp.concatenate(
            [obs_buffer_pre_push[:, 1:, :], raw_next_obs[:, None, :]], axis=1
        )

        augmented_obs = self._build_augmented_obs(obs_buffer, act_buffer, alphas, kappas)

        obs_lohi, act_lohi = self._interval_arrays()
        new_info = dict(next_state.info)
        new_info["delay_obs_buffer"] = obs_buffer
        new_info["delay_act_buffer"] = act_buffer
        new_info["delay_rng"] = next_rng
        new_info["realised_obs_delay"] = alphas
        new_info["realised_act_delay"] = kappas
        new_info["current_obs_interval"] = obs_lohi
        new_info["current_act_interval"] = act_lohi
        return next_state.replace(obs=augmented_obs, info=new_info)


class DelayedJittableVectorEnv(JittableVectorEnv):
    """Adapter that exposes a ``BraxRandomDelayWrapper``-wrapped env via the
    repo's ``JittableVectorEnv`` interface (``init`` / ``step`` returning a
    ``Timestep``)."""

    def __init__(
        self,
        seed: int,
        env: BraxRandomDelayWrapper,
        num_envs: int,
        env_checkpoint: EnvState | None = None,
        reward_gain: float = 1.0,
    ):
        self.envs = env
        self.key = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self.reward_gain = reward_gain
        self.checkpoint = env_checkpoint

    def init(self) -> tuple[State, Observation]:
        if self.checkpoint is not None:
            state = self.checkpoint
        else:
            state = jax.jit(self.envs.reset)(self.key)
        obs = state.obs
        assert isinstance(obs, jax.Array)
        return state, obs

    def step(self, state: State, action: Action) -> tuple[State, Timestep]:
        assert isinstance(action, jax.Array)
        next_state = self.envs.step(state, action)
        assert isinstance(next_state.obs, jax.Array)
        return next_state, Timestep(
            next_observation=next_state.obs,
            reward=self.reward_gain * next_state.reward,
            terminated=(next_state.done * (1 - next_state.info["truncation"])),
            truncated=next_state.info["truncation"],
            info=next_state.info,
        )
