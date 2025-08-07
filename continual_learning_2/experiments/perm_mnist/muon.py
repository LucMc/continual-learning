import time
from chex import dataclass
from typing import Literal
from pathlib import Path
import tyro
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
)
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs import (
    MuonConfig,
    DatasetConfig,
    LoggingConfig,
    TrainingConfig,
)


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def muon_mnist_experiment() -> None:
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    start = time.time()
    optim_conf = MuonConfig(learning_rate=3e-4)

    trainer = HeadResetClassificationCSLTrainer(
        seed=args.seed,
        model_config=MLPConfig(output_size=10),
        optim_cfg=optim_conf,
        data_cfg=DatasetConfig(
            name="permuted_mnist",
            seed=args.seed,
            batch_size=64,
            num_tasks=40,
            num_epochs_per_task=2,
            num_workers=0,  # (os.cpu_count() or 0) // 2,
        ),
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"muon_{args.seed}",
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            group="perm_mnist",
            wandb_mode=args.wandb_mode,
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    muon_mnist_experiment()
