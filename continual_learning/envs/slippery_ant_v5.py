from typing import Optional, Tuple, Any
from functools import partial
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv
import jax.random as random
import jax.numpy as jnp

import continual_learning.envs

# Register the environment such that it can easily be used w/ gym.make
gym.register(
    id="ContinualAnt-v0",
    entry_point=f"{__name__}:ContinualAntEnv",
    max_episode_steps=1000,
    kwargs={"change_every": 2_000_000},
)


def gen_xml_file(friction, xml_file):
    old_file = os.path.join(
        os.path.dirname(gym.envs.mujoco.ant_v5.__file__), "assets", "ant.xml"
    )
    tree = ET.parse(old_file)
    root = tree.getroot()
    root[3][1].attrib["friction"] = str(friction) + " 0.5 0.5"
    tree.write(xml_file)


class SlipperyAntEnv(AntEnv):
    """
    SlipperyAnt-v5 (defaults from https://github.com/shibhansh/loss-of-plasticity/tree/main/)
    """

    def __init__(
        self,
        friction=1,
        xml_file: str = str(Path(__file__).parent / "ant.xml"),  # This dir by default
        **kwargs,
    ):
        self.xml_file = xml_file
        gen_xml_file(friction, xml_file)
        super().__init__(**kwargs, xml_file=xml_file)
        self.print_friction()

    def print_friction(self):
        root = ET.parse(self.xml_file).getroot()
        print("XML:", root[3][1].attrib["friction"])


class ContinualAntEnv(gym.Env):
    """This continual learning Ant-v5 environment changes the friction every `change_every` timesteps"""
    def __init__(
        self,
        min_friction: float = 0.1,
        max_friction: float = 2,
        change_every=int(2e6),  # 2M How often to change friction
        xml_file: str = str(Path(__file__).parent / "ant.xml"),  # This dir by default
        seed=random.PRNGKey(0),
        render_mode=None,
        **kwargs,
    ):
        self.min_friction = min_friction
        self.max_friction = max_friction
        self.change_every = change_every
        self.seed = random.PRNGKey(seed) if type(seed) == int else seed
        self.local_time_steps = 0
        self.render_mode = render_mode
        self.env_init_kwargs = kwargs
        self.env = SlipperyAntEnv(
            friction=self.gen_random_friction(),
            render_mode=render_mode,
            **self.env_init_kwargs,
        )
        super().__init__()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata  # Enables rendering

    def gen_random_friction(self):
        self.seed, f_key = random.split(self.seed)
        friction = float(
            random.uniform(f_key, minval=self.min_friction, maxval=self.max_friction)
        )
        print("friction", friction)
        return friction

    def step(self, *args, **kwargs):
        self.local_time_steps += 1
        return self.env.step(*args, **kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        if (self.local_time_steps / self.change_every) > 1:
            print(f"[Randomising] @ {self.local_time_steps}")
            self.env = SlipperyAntEnv(
                friction=self.gen_random_friction(),
                render_mode=self.render_mode,
                **self.env_init_kwargs,
            )
            # Figure out how to not do this as to keep incrementing for stats
            self.local_time_steps = 0

        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()


if __name__ == "__main__":
    env = ContinualAnt(change_every=1000)
    obs, _ = env.reset()
