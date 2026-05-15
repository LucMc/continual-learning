from typing import Generator

import jax
import jax.numpy as jnp
import numpy as np

from continual_learning.configs.envs import EnvConfig
from continual_learning.envs.base import ContinualLearningEnv, Timestep, VectorEnv
from continual_learning.types import Action, Observation

# (minatar game name, number of obs channels)
TASK_SPECS: list[tuple[str, int]] = [
    ("space_invaders", 6),
    ("asterix", 4),
    ("seaquest", 10),
]

PADDED_CHANNELS = 10
PADDED_OBS_DIM = 10 * 10 * PADDED_CHANNELS  # 1000


class MinatarVectorEnv(VectorEnv):
    """Vectorized official MinAtar environment. """

    def __init__(
        self,
        task_name: str,
        n_actions: int,
        num_channels: int,
        num_envs: int,
        seed: int = 0,
        episode_length: int = 1000,
    ):
        from minatar import Environment

        self.task_name = task_name
        self.n_actions = n_actions
        self._num_channels = num_channels
        self._num_envs = num_envs
        self._max_episode_length = episode_length
        self._step_counts = np.zeros(num_envs, dtype=int)

        self._envs: list[Environment] = [
            Environment(task_name) for _ in range(num_envs)
        ]
        for i, env in enumerate(self._envs):
            env.seed(seed + i)

    def init(self) -> Observation:
        for env in self._envs:
            env.reset()
        self._step_counts[:] = 0
        obs = np.stack([e.state() for e in self._envs])  # (N, 10, 10, C)
        return self._pad_and_flatten(obs)

    def step(self, action: Action) -> Timestep:
        actions_np = np.asarray(action).ravel()

        rewards = np.zeros(self._num_envs, dtype=np.float32)
        dones = np.zeros(self._num_envs, dtype=bool)
        truncations = np.zeros(self._num_envs, dtype=bool)

        for i, (env, a) in enumerate(zip(self._envs, actions_np)):
            r, terminal = env.act(int(a))
            self._step_counts[i] += 1
            truncated = self._step_counts[i] >= self._max_episode_length
            rewards[i] = float(r)
            dones[i] = bool(terminal)
            truncations[i] = bool(truncated) and not bool(terminal)
            if terminal or truncated:
                env.reset()
                self._step_counts[i] = 0

        obs = np.stack([e.state() for e in self._envs])  # (N, 10, 10, C)

        return Timestep(
            next_observation=self._pad_and_flatten(obs),
            reward=jnp.array(rewards).reshape(self._num_envs, 1),
            terminated=jnp.array(dones).reshape(self._num_envs, 1),
            truncated=jnp.array(truncations).reshape(self._num_envs, 1),
            info={},
        )

    def _pad_and_flatten(self, obs: np.ndarray) -> jax.Array:
        """Pad obs channels to PADDED_CHANNELS and flatten to 1D."""
        # obs: (num_envs, 10, 10, num_channels)
        pad_channels = PADDED_CHANNELS - self._num_channels
        if pad_channels > 0:
            pad = np.zeros((*obs.shape[:-1], pad_channels), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=-1)
        return jnp.array(obs.reshape(self._num_envs, PADDED_OBS_DIM), dtype=jnp.float32)

    def save(self) -> dict:
        return {}

    def load(self, checkpoint: dict):
        pass


class MinatarContinualEnv(ContinualLearningEnv):
    """Continual learning benchmark chaining 3 MinAtar tasks.

    Tasks (in order): space_invaders → asterix → seaquest.
    All observations padded to (10, 10, 10) and flattened to 1000 dims.
    """

    def __init__(self, seed: int, config: EnvConfig):
        self._seed = seed
        self._num_envs = config.num_envs
        self._episode_length = config.episode_length
        self._current_task_idx = 0

        # All official MinAtar games have 6 actions
        from minatar import Environment
        self._max_n_actions = max(
            Environment(name).num_actions() for name, _ in TASK_SPECS
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def tasks(self) -> Generator[MinatarVectorEnv, None, None]:
        from minatar import Environment

        for idx, (task_name, num_channels) in enumerate(TASK_SPECS):
            self._current_task_idx = idx
            n_actions = Environment(task_name).num_actions()
            yield MinatarVectorEnv(
                task_name=task_name,
                n_actions=n_actions,
                num_channels=num_channels,
                num_envs=self._num_envs,
                seed=self._seed,
                episode_length=self._episode_length,
            )

    @property
    def observation_spec(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape=(PADDED_OBS_DIM,), dtype=jnp.float32)

    @property
    def action_dim(self) -> int:
        return self._max_n_actions

    def evaluate(self, agent, forgetting: bool = False) -> None:
        return None

    def save(self) -> dict:
        return {"current_task_idx": self._current_task_idx}

    def load(self, checkpoint: dict, envs_checkpoint=None):
        self._current_task_idx = checkpoint["current_task_idx"]
