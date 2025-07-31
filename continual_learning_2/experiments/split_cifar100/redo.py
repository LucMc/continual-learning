import os
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
    DatasetConfig,
    LoggingConfig,
    TrainingConfig,
)
from continual_learning_2.configs import RedoConfig, AdamConfig
from continual_learning_2.configs.models import CNNConfig


def redo_split_cifar100_experiment():
    SEED = 42
    start = time.time()
    optim_conf = RedoConfig(
        tx=AdamConfig(learning_rate=1e-3), update_frequency=100, score_threshold=0.1
    )

    # Add validation to say what the available options are for dataset etc
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=CNNConfig(output_size=10),
        optim_cfg=optim_conf,
        data_cfg=DatasetConfig(
            name="split_cifar100",
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
            run_name=f"redo_{SEED}",
            wandb_entity="lucmc",
            wandb_project="crl_experiments",
            group="split_cifar100",
            wandb_mode="online",
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    redo_split_cifar100_experiment()
