from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class ReplayBatch(NamedTuple):
    observations: Float[Array, "batch obs_dim"]
    actions: Float[Array, "batch action_dim"]
    rewards: Float[Array, "batch 1"]
    next_observations: Float[Array, "batch obs_dim"]
    dones: Bool[Array, "batch 1"]


@struct.dataclass
class ReplayBufferState(struct.PyTreeNode):
    observations: Float[Array, "capacity obs_dim"]
    actions: Float[Array, "capacity action_dim"]
    rewards: Float[Array, "capacity 1"]
    next_observations: Float[Array, "capacity obs_dim"]
    dones: Bool[Array, "capacity 1"]

    position: int = 0
    size: int = 0


class ReplayBuffer:
    """Circular replay buffer with pre-allocated arrays for JIT compatibility. """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def init(self) -> ReplayBufferState:
        return ReplayBufferState(
            observations=jnp.zeros((self.capacity, self.obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((self.capacity, self.action_dim), dtype=jnp.float32),
            rewards=jnp.zeros((self.capacity, 1), dtype=jnp.float32),
            next_observations=jnp.zeros((self.capacity, self.obs_dim), dtype=jnp.float32),
            dones=jnp.zeros((self.capacity, 1), dtype=bool),
            position=0,
            size=0,
        )

    @staticmethod
    @jax.jit
    def add(
        state: ReplayBufferState,
        obs: Float[Array, "... obs_dim"],
        action: Float[Array, "... action_dim"],
        reward: Float[Array, "... 1"],
        next_obs: Float[Array, "... obs_dim"],
        done: Bool[Array, "... 1"],
    ) -> ReplayBufferState:
        """Add a transition (or batch of transitions) to the buffer. """
        # Handle both single transitions and batches
        obs = jnp.atleast_2d(obs)
        action = jnp.atleast_2d(action)
        reward = jnp.atleast_2d(reward)
        next_obs = jnp.atleast_2d(next_obs)
        done = jnp.atleast_2d(done)

        batch_size = obs.shape[0]
        capacity = state.observations.shape[0]

        # Compute indices for insertion (circular buffer)
        indices = (jnp.arange(batch_size) + state.position) % capacity

        # Update arrays
        new_observations = state.observations.at[indices].set(obs)
        new_actions = state.actions.at[indices].set(action)
        new_rewards = state.rewards.at[indices].set(reward)
        new_next_observations = state.next_observations.at[indices].set(next_obs)
        new_dones = state.dones.at[indices].set(done)

        # Update position and size
        new_position = (state.position + batch_size) % capacity
        new_size = jnp.minimum(state.size + batch_size, capacity)

        return ReplayBufferState(
            observations=new_observations,
            actions=new_actions,
            rewards=new_rewards,
            next_observations=new_next_observations,
            dones=new_dones,
            position=new_position,
            size=new_size,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(
        state: ReplayBufferState,
        key: PRNGKeyArray,
        batch_size: int,
    ) -> ReplayBatch:
        """Sample a batch of transitions from the buffer. """
        # Sample random indices from valid range
        indices = jax.random.randint(key, (batch_size,), 0, state.size)  # pyright: ignore[reportArgumentType]

        return ReplayBatch(
            observations=state.observations[indices],
            actions=state.actions[indices],
            rewards=state.rewards[indices],
            next_observations=state.next_observations[indices],
            dones=state.dones[indices],
        )

    @staticmethod
    def is_ready(state: ReplayBufferState, min_size: int) -> bool:
        return state.size >= min_size

