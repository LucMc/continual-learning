# pyright: reportAttributeAccessIssue=false
from typing import Generator

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import envs as brax_envs
from brax.envs.ant import Ant
from brax.envs.humanoid import Humanoid
from brax.envs.half_cheetah import Halfcheetah
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath

from continual_learning.configs.envs import EnvConfig
from continual_learning.envs.base import (
    Agent,
    JittableContinualLearningEnv,
    JittableVectorEnv,
    Timestep,
)
from continual_learning.types import Action, EnvState, Observation


class SlipperyHumanoid(Humanoid):
    """The Humanoid environment but with customisable friction"""

    def __init__(
        self,
        friction: float,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.01,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        **kwargs,
    ):
        #### This whole loop is copypasted from the parent class, except
        path = epath.resource_path("brax") / "envs/assets/humanoid.xml"
        sys = mjcf.load(path)

        #### We set the friction programmatically
        model = sys.mj_model
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_geom_id == -1:
            raise ValueError("Humanoid MJCF missing geom named 'floor'.")
        model.geom_friction[floor_geom_id] = np.array([friction, 0.1, 0.1])
        sys = sys.replace(mj_model=model)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.0015})
            n_frames = 10
            # fmt: off
            gear = jnp.array([
                350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
                350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))
            # fmt: on

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        PipelineEnv.__init__(self, sys=sys, backend=backend, **kwargs)  # pyright: ignore[reportArgumentType]

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )


class SlipperyAnt(Ant):
    """The Ant environment but with customisable friction"""

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


class SlipperyCheetah(Halfcheetah):
    """The Humanoid environment but with customisable friction"""

    def __init__(
        self,
        friction: float,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        **kwargs,
    ):
        #### This whole loop is copypasted from the parent class, except
        path = epath.resource_path("brax") / "envs/assets/half_cheetah.xml"
        sys = mjcf.load(path)

        #### We set the friction programmatically
        model = sys.mj_model
        model.geom_friction[:] = np.array([friction, 0.1, 0.1])
        sys = sys.replace(mj_model=model)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.003125})
            n_frames = 16
            gear = jnp.array([120, 90, 60, 120, 100, 100])
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )


class JittableVectorEnvWrapper(JittableVectorEnv):
    def __init__(
        self,
        seed: int,
        env: PipelineEnv,
        num_envs: int,
        episode_length: int,
        env_checkpoint: EnvState | None = None,
        reward_gain: float = 1.0,
    ):
        envs = brax_envs.training.wrap(
            env,
            episode_length=episode_length,
            action_repeat=1,
        )

        self.envs = envs
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
            terminated=(next_state.done * (1 - state.info["truncation"])),
            truncated=next_state.info["truncation"],
            info=next_state.info,
        )


class ContinualAnt(JittableContinualLearningEnv):
    def __init__(self, seed: int, config: EnvConfig):
        self._num_envs = config.num_envs
        self._episode_length = config.episode_length

        rng = np.random.default_rng(seed)
        self.seed = seed
        low, high = np.log10(0.02), np.log10(2.0)
        self.frictions = np.pow(10, rng.uniform(low=low, high=high, size=config.num_tasks))
        self.current_task = 0
        self.saved_envs: JittableVectorEnv | None = None
        self.reward_gain = 1.0
        self.backend = "mjx"

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
            env=SlipperyAnt(friction=friction, backend=self.backend),
            num_envs=self.num_envs,
            episode_length=self._episode_length,
            env_checkpoint=env_checkpoint,
            reward_gain=self.reward_gain,
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


class ContinualHumanoid(JittableContinualLearningEnv):
    def __init__(self, seed: int, config: EnvConfig):
        self._num_envs = config.num_envs
        self._episode_length = config.episode_length

        rng = np.random.default_rng(seed)
        self.seed = seed
        low, high = np.log10(0.02), np.log10(2.0)
        self.frictions = np.pow(10, rng.uniform(low=low, high=high, size=config.num_tasks))
        self.current_task = 0
        self.saved_envs: JittableVectorEnv | None = None
        self.reward_gain = 0.1
        self.backend = "mjx"

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
            env=SlipperyHumanoid(friction=friction, backend=self.backend),
            num_envs=self.num_envs,
            episode_length=self._episode_length,
            env_checkpoint=env_checkpoint,
            reward_gain=self.reward_gain,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def evaluate(self, agent: Agent, forgetting: bool = False) -> dict[str, float] | None:
        del agent, forgetting
        return None

    @property
    def observation_spec(self) -> jax.ShapeDtypeStruct:
        env = SlipperyHumanoid(friction=self.frictions[0])
        return jax.ShapeDtypeStruct((1, env.observation_size), jnp.float32)

    @property
    def action_dim(self) -> int:
        env = SlipperyHumanoid(friction=self.frictions[0])
        return env.action_size


class ContinualCheetah(JittableContinualLearningEnv):
    def __init__(self, seed: int, config: EnvConfig):
        self._num_envs = config.num_envs
        self._episode_length = config.episode_length

        rng = np.random.default_rng(seed)
        self.seed = seed
        low, high = np.log10(0.02), np.log10(2.0)
        self.frictions = np.pow(10, rng.uniform(low=low, high=high, size=config.num_tasks))
        self.current_task = 0
        self.saved_envs: JittableVectorEnv | None = None
        self.reward_gain = 1.0
        self.backend = "mjx"

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
            env=SlipperyAnt(friction=friction, backend=self.backend),
            num_envs=self.num_envs,
            episode_length=self._episode_length,
            env_checkpoint=env_checkpoint,
            reward_gain=self.reward_gain,
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
