import time
from chex import dataclass
from typing import Literal
from pathlib import Path
import tyro
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
)
from continual_learning_2.configs.models import CNNConfig
from continual_learning_2.configs import (
    AdamConfig,
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


def adam_split_cifar10_experiment() -> None:
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    start = time.time()
    optim_conf = tx = AdamConfig(learning_rate=1e-3)

    trainer = HeadResetClassificationCSLTrainer(
        seed=args.seed,
        model_config=CNNConfig(output_size=10),
        optim_cfg=optim_conf,
        data_cfg=DatasetConfig(
            name="split_cifar10",
            seed=args.seed,
            batch_size=64,
            num_tasks=10,
            num_epochs_per_task=2,
            # num_workers=0,  # (os.cpu_count() or 0) // 2,
            dataset_kwargs = {
                "flatten" : False
            }
        ),
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"adam_{args.seed}",
            wandb_entity="lucmc",
            wandb_project="crl_experiments",
            group="split_cifar10",
            wandb_mode="online",
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    adam_split_cifar10_experiment()
