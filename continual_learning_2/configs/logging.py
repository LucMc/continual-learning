from dataclasses import dataclass
from tokenize import group
from typing import Literal


@dataclass
class LoggingConfig:
    run_name: str
    wandb_entity: str
    wandb_project: str
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    group: str | None = None

    interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000
    eval_during_training: bool = False
    catastrophic_forgetting: bool = False
