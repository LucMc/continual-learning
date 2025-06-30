# pyright: reportAttributeAccessIssue=false
from typing import Generator

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.ant import Ant
from brax.envs.base import PipelineEnv, State
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper, Wrapper
from brax.io import mjcf
from etils import epath

from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.envs.base import (
    Agent,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    Timestep,
)
from continual_learning_2.types import EnvState, Observation


class SlipperyAnt(Ant):
    """The Ant-v5 environment but with customisable friction"""

    def __init__(
        self,
        friction: float,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        **kwargs,
    ):
        #### This whole loop is copypasted from the parent class, except
        path = epath.resource_path("brax") / "envs/assets/ant.xml"
        sys = mjcf.load(path)

        #### We set the friction programmatically
        model = sys.mj_model
        model.geom_friction[:] = np.array([friction, 0.5, 0.5])
        sys = sys.replace(mj_model=model)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear))
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        PipelineEnv.__init__(self, sys=sys, backend=backend, **kwargs)  # pyright: ignore[reportArgumentType]

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")


class AutoResetWrapper(Wrapper):
    """Custom AutoResetWrapper that's more like Gymnasium in behaviour.

    The default Brax AutoResetWrapper loses the final observation.
    """

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["final_episode_returns"] = jnp.zeros_like(state.reward)
        state.info["final_episode_lengths"] = jnp.zeros_like(state.reward)

        assert isinstance(state.obs, jax.Array)
        state.info["final_observation"] = jnp.zeros_like(state.obs)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Get rid of any done flags from previous steps
        state = state.replace(done=jnp.zeros_like(state.done))

        state = self.env.step(state, action)

        # The actual autoreset
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )

        # Save final obs
        assert isinstance(state.obs, jax.Array)
        state.info.update(
            final_observation=jnp.where(state.done, state.obs, jnp.zeros_like(state.obs))
        )

        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        if "episode_metrics" in state.info:
            returns = state.info["episode_metrics"]["sum_reward"]
            lengths = state.info["episode_metrics"]["length"]
            state.info.update(
                final_episode_returns=jnp.where(state.done, returns, jnp.zeros_like(returns)),
                final_episode_lengths=jnp.where(state.done, lengths, jnp.zeros_like(lengths)),
            )

        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class JittableVectorEnvWrapper(JittableVectorEnv):
    def __init__(
        self,
        seed: int,
        env: PipelineEnv,
        num_envs: int,
        episode_length: int,
        env_checkpoint: EnvState | None,
    ):
        self.envs = VmapWrapper(env, batch_size=num_envs)
        self.envs = EpisodeWrapper(self.envs, episode_length=episode_length, action_repeat=1)
        self.envs = AutoResetWrapper(self.envs)
        self.key = jax.random.PRNGKey(seed)

        self.checkpoint = env_checkpoint

    def init(self) -> tuple[State, Observation]:
        if self.checkpoint is not None:
            state = self.checkpoint
        else:
            self.key, reset_key = jax.random.split(self.key)
            state = self.envs.reset(reset_key)

        obs = state.obs

        assert state.pipeline_state is not None
        assert isinstance(obs, jax.Array)

        return state, obs

    def step(self, state: State, action) -> tuple[State, Timestep]:
        assert isinstance(action, jax.Array)
        state = self.envs.step(state, action)

        assert isinstance(state.obs, jax.Array)

        return state, Timestep(
            next_observation=state.obs,
            reward=state.reward,
            terminated=state.done - state.info["truncation"],
            truncated=state.info["truncation"],
            final_episode_returns=state.info["final_episode_returns"],
            final_episode_lengths=state.info["final_episode_lengths"],
            final_observation=state.info["final_observation"],
        )


class ContinualAnt(JittableContinualLearningEnv):
    def __init__(self, seed: int, config: EnvConfig):
        self._num_envs = config.num_envs

        rng = np.random.default_rng(seed)
        self.seed = seed
        self.frictions = rng.uniform(low=0.1, high=2.0, size=config.num_tasks)
        self.current_task = 0
        self.saved_envs: JittableVectorEnv | None = None

    @property
    def tasks(self) -> Generator[JittableVectorEnv, None, None]:
        for task in range(self.current_task, len(self.frictions)):
            self.current_task = task
            yield self._get_task(task, self.saved_envs)
            self.saved_envs = None

    def save(self, env_state: EnvState) -> dict:
        return {"current_task": self.current_task, "env_state": env_state}

    def load(self, checkpoint: dict):
        self.current_task = checkpoint["current_task"]
        self.saved_env_state = checkpoint["env_state"]

    def _get_task(self, task_id: int, env_checkpoint: EnvState) -> JittableVectorEnv:
        friction = self.frictions[task_id]
        return self._make_envs(friction, env_checkpoint)

    def _make_envs(self, friction: float, env_checkpoint: EnvState) -> JittableVectorEnv:
        return JittableVectorEnvWrapper(
            seed=self.seed,
            env=SlipperyAnt(friction=friction),
            num_envs=self.num_envs,
            episode_length=1_000,
            env_checkpoint=env_checkpoint,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def evaluate(self, agent: Agent, forgetting: bool = False) -> dict[str, float] | None:
        del agent, forgetting
        return None

    @property
    def observation_spec(self) -> jax.ShapeDtypeStruct:
        env = SlipperyAnt(friction=self.frictions[0])
        return jax.ShapeDtypeStruct((1, env.observation_size), jnp.float32)

    @property
    def action_dim(self) -> int:
        env = SlipperyAnt(friction=self.frictions[0])
        return env.action_size
