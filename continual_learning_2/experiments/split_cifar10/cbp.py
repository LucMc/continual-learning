import os
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
    DatasetConfig,
    LoggingConfig,
    TrainingConfig,
)
from continual_learning_2.configs import CBPConfig, AdamConfig
from continual_learning_2.configs.models import CNNConfig

# @dataclass(frozen=True)
# class Args:
#     seed: int = 1
#     wandb_mode: Literal["online", "offline", "disabled"] = "online"
#     wandb_project: str | None = None
#     wandb_entity: str | None = None
#     data_dir: Path = Path("./experiment_results")
#     resume: bool = False

def cbp_split_cifar10_experiment():
    SEED = 42
    start = time.time()
    optim_conf = CBPConfig(
        tx=AdamConfig(learning_rate=1e-3),
        decay_rate=0.9,
        replacement_rate=0.5,
        maturity_threshold=20,
    )

    # Add validation to say what the available options are for dataset etc
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=CNNConfig(output_size=10),
        optim_cfg=optim_conf,
        data_cfg=DatasetConfig(
            name="split_cifar10",
            seed=SEED,
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
            run_name=f"cbp_{SEED}",
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
    cbp_split_cifar10_experiment()
