from dataclasses import dataclass


@dataclass
class LoggingConfig:
    run_name: str
    wandb_entity: str
    wandb_project: str

    interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000
