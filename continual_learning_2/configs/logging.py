from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class LoggingConfig:
    run_name: str
    wandb_entity: str
    wandb_project: str
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    checkpoint_dir: Path = Path("./checkpoints").absolute()
    best_metric: str = "metrics/eval_score"

    interval: int = 100
    save: bool = True
    save_interval: int = 1000
    eval_interval: int = 1000
    eval_during_training: bool = False
    catastrophic_forgetting: bool = False
