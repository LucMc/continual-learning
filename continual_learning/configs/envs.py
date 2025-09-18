from dataclasses import dataclass


@dataclass
class EnvConfig:
    name: str
    num_envs: int
    num_tasks: int
    episode_length: int
