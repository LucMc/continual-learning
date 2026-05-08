"""Tests for the JIT-compatible Brax random-delay wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax import envs as brax_envs
from brax.envs.ant import Ant

from continual_learning.envs.brax_delay_wrapper import BraxRandomDelayWrapper


def _make_delay_env(
    *,
    obs_delay_range: range,
    act_delay_range: range,
    overall_obs_delay_max: int,
    overall_act_delay_max: int,
    num_envs: int = 4,
    episode_length: int = 200,
    seed: int = 0,
    delay_info_mode: str = "one_hot",
):
    base = Ant(backend="generalized")
    wrapped = brax_envs.training.wrap(
        base, episode_length=episode_length, action_repeat=1
    )
    delayed = BraxRandomDelayWrapper(
        wrapped,
        num_envs=num_envs,
        obs_delay_range=obs_delay_range,
        act_delay_range=act_delay_range,
        overall_obs_delay_max=overall_obs_delay_max,
        overall_act_delay_max=overall_act_delay_max,
        base_obs_dim=int(base.observation_size),
        action_dim=int(base.action_size),
        seed=seed,
        delay_info_mode=delay_info_mode,
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    reset = jax.jit(delayed.reset)
    step = jax.jit(delayed.step)
    return delayed, reset, step, keys, int(base.observation_size), int(base.action_size)


def test_identity_zero_delay():
    """alpha=kappa=0 → delayed_obs equals the most recent raw obs each step."""
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(0, 1),
        act_delay_range=range(0, 1),
        overall_obs_delay_max=1,
        overall_act_delay_max=1,
        num_envs=4,
    )
    state = reset(keys)

    # Reset alpha/kappa should be 0 (only choice).
    assert jnp.all(state.info["realised_obs_delay"] == 0)
    assert jnp.all(state.info["realised_act_delay"] == 0)

    for t in range(20):
        action = jnp.zeros((4, action_dim), dtype=jnp.float32)
        state = step(state, action)
        raw = state.info["delay_obs_buffer"][:, -1, :]  # most-recent raw obs
        delayed_obs = state.obs[:, :obs_dim]
        np.testing.assert_allclose(np.asarray(delayed_obs), np.asarray(raw))
        assert jnp.all(state.info["realised_obs_delay"] == 0)
        assert jnp.all(state.info["realised_act_delay"] == 0)


def test_fixed_two_step_obs_delay():
    """obs_delay fixed at 2 → after warmup, delayed_obs[t] == raw_obs[t-2]."""
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(2, 3),  # fixed alpha = 2
        act_delay_range=range(0, 1),  # kappa = 0 to isolate obs delay
        overall_obs_delay_max=3,
        overall_act_delay_max=1,
        num_envs=4,
    )
    state = reset(keys)

    raw_history: list[jnp.ndarray] = []
    delayed_history: list[jnp.ndarray] = []
    rng = jax.random.PRNGKey(123)
    for t in range(15):
        rng, subkey = jax.random.split(rng)
        action = jax.random.uniform(subkey, (4, action_dim), minval=-1.0, maxval=1.0)
        state = step(state, action)
        raw_history.append(state.info["delay_obs_buffer"][:, -1, :])
        delayed_history.append(state.obs[:, :obs_dim])
        assert jnp.all(state.info["realised_obs_delay"] == 2)
        assert jnp.all(state.info["realised_act_delay"] == 0)

    # For t >= 2, delayed_obs at history index t must equal raw_obs at history t-2.
    for t in range(2, len(delayed_history)):
        np.testing.assert_allclose(
            np.asarray(delayed_history[t]), np.asarray(raw_history[t - 2])
        )


def test_blind_mode_omits_action_buffer_and_delay_info():
    """blind mode exposes only the delayed raw observation to the policy."""
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(2, 3),
        act_delay_range=range(3, 4),
        overall_obs_delay_max=5,
        overall_act_delay_max=5,
        num_envs=4,
        delay_info_mode="blind",
    )
    assert delayed.observation_size == obs_dim

    state = reset(keys)
    assert state.obs.shape == (4, obs_dim)

    raw_history: list[jnp.ndarray] = []
    delayed_history: list[jnp.ndarray] = []
    for _ in range(15):
        action = jnp.zeros((4, action_dim), dtype=jnp.float32)
        state = step(state, action)
        assert state.obs.shape == (4, obs_dim)
        raw_history.append(state.info["delay_obs_buffer"][:, -1, :])
        delayed_history.append(state.obs)

    for t in range(2, len(delayed_history)):
        np.testing.assert_allclose(
            np.asarray(delayed_history[t]), np.asarray(raw_history[t - 2])
        )


def test_episode_boundary_buffer_reset():
    """On the first step of a new episode, older buffer slots are cleared
    (filled with first_obs / zero) — only the just-pushed action and just-
    received obs occupy the rightmost slot."""
    episode_length = 5  # tiny episode → triggers reset within the test
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(0, 1),
        act_delay_range=range(0, 1),
        overall_obs_delay_max=4,
        overall_act_delay_max=4,
        num_envs=4,
        episode_length=episode_length,
    )
    state = reset(keys)

    saw_reset = False
    rng = jax.random.PRNGKey(7)
    for t in range(episode_length * 3):
        rng, subkey = jax.random.split(rng)
        action = jax.random.uniform(subkey, (4, action_dim), minval=-1.0, maxval=1.0)
        state = step(state, action)
        steps_per_env = state.info["steps"]
        if jnp.any(steps_per_env == 1) and t > 0:
            first_obs = state.info["first_obs"]
            obs_buffer = state.info["delay_obs_buffer"]
            act_buffer = state.info["delay_act_buffer"]
            mask = steps_per_env == 1  # (num_envs,)
            for env_idx in jnp.where(mask)[0].tolist():
                # Older obs slots (excluding the just-pushed raw_next_obs)
                # equal first_obs for that env.
                buf = obs_buffer[env_idx, :-1, :]
                expected = jnp.broadcast_to(first_obs[env_idx], buf.shape)
                np.testing.assert_allclose(
                    np.asarray(buf), np.asarray(expected), atol=1e-6
                )
                # Older action slots equal zero; the just-pushed slot equals
                # the action we sent this step.
                np.testing.assert_allclose(
                    np.asarray(act_buffer[env_idx, :-1, :]),
                    np.zeros((act_buffer.shape[1] - 1, action_dim), dtype=np.float32),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    np.asarray(act_buffer[env_idx, -1, :]),
                    np.asarray(action[env_idx]),
                    atol=1e-6,
                )
            saw_reset = True

    assert saw_reset, "expected at least one episode reset within 3*episode_length steps"


def test_action_clip_breaks_feedback_loop():
    """Huge raw actions are clipped to the configured magnitude before being
    stored in act_buffer or applied to the env."""
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(0, 1),
        act_delay_range=range(0, 1),
        overall_obs_delay_max=2,
        overall_act_delay_max=2,
        num_envs=4,
    )
    state = reset(keys)
    huge = jnp.full((4, action_dim), 1e10, dtype=jnp.float32)
    state = step(state, huge)
    # The action stored in the buffer (rightmost slot) must be clipped to
    # action_clip, NOT 1e10 — otherwise the augmented-obs feedback loop is
    # still open.
    stored = state.info["delay_act_buffer"][:, -1, :]
    assert jnp.all(stored == 5.0), f"expected stored action = 5.0, got {stored}"
    # The policy still gets a finite obs (no inf / nan from a runaway env).
    assert jnp.all(jnp.isfinite(state.obs))


def test_action_clip_loose_enough_for_normal_actions():
    """Normal policy outputs (within ±3) pass through unchanged."""
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(0, 1),
        act_delay_range=range(0, 1),
        overall_obs_delay_max=2,
        overall_act_delay_max=2,
        num_envs=4,
    )
    state = reset(keys)
    rng = jax.random.PRNGKey(0)
    for _ in range(20):
        rng, subkey = jax.random.split(rng)
        # Sample from N(0, 1) — almost all values within ±3, well below clip=5.
        action = jax.random.normal(subkey, (4, action_dim))
        state = step(state, action)
        stored = state.info["delay_act_buffer"][:, -1, :]
        np.testing.assert_allclose(np.asarray(stored), np.asarray(action), atol=1e-6)


def test_jit_compiles_and_runs():
    """End-to-end JIT smoke test for many steps."""
    delayed, reset, step, keys, obs_dim, action_dim = _make_delay_env(
        obs_delay_range=range(0, 4),
        act_delay_range=range(0, 4),
        overall_obs_delay_max=4,
        overall_act_delay_max=4,
        num_envs=8,
    )
    state = reset(keys)
    rng = jax.random.PRNGKey(42)
    for _ in range(50):
        rng, subkey = jax.random.split(rng)
        action = jax.random.uniform(subkey, (8, action_dim), minval=-1.0, maxval=1.0)
        state = step(state, action)
    # Sanity: obs is finite, has the augmented dim, and per-step delays sit in range.
    assert jnp.all(jnp.isfinite(state.obs))
    expected_dim = obs_dim + 4 * action_dim + 4 + 4
    assert state.obs.shape == (8, expected_dim)
    assert jnp.all(state.info["realised_obs_delay"] >= 0)
    assert jnp.all(state.info["realised_obs_delay"] < 4)
    assert jnp.all(state.info["realised_act_delay"] >= 0)
    assert jnp.all(state.info["realised_act_delay"] < 4)
