import os
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
)
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs import (
    CBPConfig,
    AdamConfig,
    DatasetConfig,
    LoggingConfig,
    TrainingConfig,
)


def adam_mnist_experiment():
    SEED = 42
    start = time.time()
    optim_conf = tx = AdamConfig(learning_rate=1e-3)

    # Add validation to say what the available options are for dataset etc
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=MLPConfig(output_size=10),
        optim_cfg=optim_conf,
        data_cfg=DatasetConfig(
            name="split_mnist",
            seed=SEED,
            batch_size=64,
            num_tasks=10,
            num_epochs_per_task=20,
            num_workers=0,  # (os.cpu_count() or 0) // 2,
        ),
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name="adam",
            wandb_entity="lucmc",
            wandb_project="crl_experiments",
            group="split_mnist",
            wandb_mode="online",
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    adam_mnist_experiment()
