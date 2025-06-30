import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Bool, Float

from continual_learning_2.envs.base import ContinualLearningEnv
from continual_learning_2.types import Action, Done, Observation, Reward, Rollout


class RolloutBuffer:
    num_rollout_steps: int
    num_envs: int
    pos: int

    observations: Float[Observation, "timestep env"]
    actions: Float[Action, "timestep env"]
    rewards: Float[npt.NDArray, "timestep env 1"]
    done: Float[npt.NDArray, "timestep env 1"]

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
        self.done = np.zeros((self.num_rollout_steps, self.num_envs, 1), dtype=self.dtype)

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
            "dones": self.done,
            "log_probs": self.log_probs,
            "values": self.values,
            "pos": self.pos,
        }

    def load(self, state: dict):
        self.observations = state["observations"]
        self.actions = state["actions"]
        self.rewards = state["rewards"]
        self.done = state["dones"]
        self.log_probs = state["log_probs"]
        self.values = state["values"]
        self.pos = state["pos"]

    def add(
        self,
        obs: Float[Observation, " env"],
        action: Float[Action, " env"],
        reward: Float[Reward, " env"],
        done: Bool[Done, " env"],
        value: Float[Reward, " env"] | None = None,
        log_prob: Float[Reward, " env"] | None = None,
    ):
        # NOTE: assuming batch dim = env dim
        assert obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_envs
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.done[self.pos] = done.copy().reshape(-1, 1)

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
            self.done,
            self.log_probs,
            self.values,
        )


def compute_gae(
    rollouts: Rollout,
    gamma: float,
    gae_lambda: float,
    last_values: Float[Array, "... 1"],
) -> Rollout:
    assert rollouts.values is not None

    advantages = np.zeros_like(rollouts.rewards)

    # Adapted from https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py
    # Changed to actually use the done flag
    last_gae_lamda = 0
    num_rollout_steps = rollouts.observations.shape[0]
    assert last_values is not None

    for timestep in reversed(range(num_rollout_steps)):
        next_nonterminal = 1.0 - rollouts.dones[timestep]
        delta = (
            rollouts.rewards[timestep]
            + next_nonterminal * gamma * last_values
            - rollouts.values[timestep]
        )
        advantages[timestep] = last_gae_lamda = (
            delta + next_nonterminal * gamma * gae_lambda * last_gae_lamda
        )
        last_values = rollouts.values[timestep]

    return rollouts._replace(
        advantages=advantages,
        returns=advantages + rollouts.values,
    )


@jax.jit
def compute_gae_scan(
    rollouts: Rollout, last_values: Float[Array, "... 1"], gamma: float, gae_lambda: float
) -> Rollout:
    """Adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py#L142"""

    def get_advantages(gae_and_next_value: tuple[jax.Array, jax.Array], rollout: Rollout):
        assert rollout.values is not None

        gae, next_value = gae_and_next_value
        next_nonterminal = 1.0 - rollout.dones
        delta = (rollout.rewards + next_nonterminal * gamma * next_value) - rollout.values
        gae = delta + next_nonterminal * gamma * gae_lambda * gae
        return (gae, rollout.values), gae

    _, advantages = jax.lax.scan(
        get_advantages,  # pyright: ignore[reportArgumentType]
        (jnp.zeros_like(last_values), last_values),
        rollouts,
        reverse=True,
        unroll=16,
    )
    return rollouts._replace(
        advantages=advantages,
        returns=advantages + rollouts.values,
    )
