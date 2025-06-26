import numpy as np
import numpy.typing as npt
from jaxtyping import Float

from continual_learning_2.envs.base import ContinualLearningEnv
from continual_learning_2.types import Action, Observation, Rollout


class RolloutBuffer:
    num_rollout_steps: int
    num_envs: int
    pos: int

    observations: Float[Observation, "timestep env"]
    actions: Float[Action, "timestep env"]
    rewards: Float[npt.NDArray, "timestep env 1"]
    episode_starts: Float[npt.NDArray, "timestep env 1"]

    values: Float[npt.NDArray, "timestep env 1"]
    log_probs: Float[npt.NDArray, "timestep env 1"]
    means: Float[Action, "timestep env"]
    stds: Float[Action, "timestep env"]

    def __init__(
        self,
        num_rollout_steps: int,
        benchmark: ContinualLearningEnv,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        self.num_rollout_steps = num_rollout_steps
        self.num_envs = benchmark.num_envs
        self._obs_shape = benchmark.observation_spec.shape[-1]
        self._action_shape = benchmark.action_dim
        self.dtype = dtype
        self.reset()  # Init buffer

    def reset(self) -> None:
        """Reinitialize the buffer."""
        self.observations = np.zeros(
            (self.num_rollout_steps, self.num_envs, self._obs_shape), dtype=self.dtype
        )
        self.actions = np.zeros(
            (self.num_rollout_steps, self.num_envs, self._action_shape),
            dtype=self.dtype,
        )
        self.rewards = np.zeros((self.num_rollout_steps, self.num_envs, 1), dtype=self.dtype)
        self.episode_starts = np.zeros(
            (self.num_rollout_steps, self.num_envs, 1), dtype=self.dtype
        )

        self.log_probs = np.zeros((self.num_rollout_steps, self.num_envs, 1), dtype=self.dtype)
        self.values = np.zeros_like(self.rewards)

        self.pos = 0

    @property
    def ready(self) -> bool:
        return self.pos == self.num_rollout_steps

    def save(self) -> dict:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "episode_starts": self.episode_starts,
            "log_probs": self.log_probs,
            "values": self.values,
            "pos": self.pos,
        }

    def load(self, state: dict):
        self.observations = state["observations"]
        self.actions = state["actions"]
        self.rewards = state["rewards"]
        self.episode_starts = state["episode_starts"]
        self.log_probs = state["log_probs"]
        self.values = state["values"]
        self.pos = state["pos"]

    def add(
        self,
        obs: Float[Observation, " env"],
        action: Float[Action, " env"],
        reward: Float[npt.NDArray, " env"],
        episode_start: Float[npt.NDArray, " env"],
        value: Float[npt.NDArray, " env"] | None = None,
        log_prob: Float[npt.NDArray, " env"] | None = None,
    ):
        # NOTE: assuming batch dim = env dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and episode_start.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == episode_start.shape[0]
            == self.num_envs
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.episode_starts[self.pos] = episode_start.copy().reshape(-1, 1)

        if value is not None:
            self.values[self.pos] = value.copy()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.reshape(-1, 1).copy()

        self.pos += 1

    def get(
        self,
    ) -> Rollout:
        return Rollout(
            self.observations,
            self.actions,
            self.rewards,
            self.episode_starts,
            self.log_probs,
            self.values,
        )
