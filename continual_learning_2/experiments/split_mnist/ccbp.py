import os
import time
from chex import dataclass
from typing import Literal
from pathlib import Path
import tyro
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
)
from continual_learning_2.configs import (
    CcbpConfig,
    AdamConfig,
    MLPConfig,
    TrainingConfig,
    DatasetConfig,
    LoggingConfig,
)


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def ccbp_mnist_experiment():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None
    start = time.time()
    optim_conf = CcbpConfig(
        tx=AdamConfig(learning_rate=1e-3),
        seed=args.seed,
        decay_rate=0.9,
        replacement_rate=0.5,
        maturity_threshold=20,
    )

    # Add validation to say what the available options are for dataset etc
    trainer = HeadResetClassificationCSLTrainer(
        seed=args.seed,
        model_config=MLPConfig(output_size=10),
        optim_cfg=optim_conf,
        data_cfg=DatasetConfig(
            name="split_mnist",
            seed=args.seed,
            batch_size=64,
            num_tasks=10,
            num_epochs_per_task=20,
            num_workers=0,  # (os.cpu_count() or 0) // 2,
        ),
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"ccbp_{args.seed}",
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            group="split_mnist",
            wandb_mode=args.wandb_mode,
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    ccbp_mnist_experiment()
