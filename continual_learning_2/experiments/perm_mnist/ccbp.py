import os
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
    DatasetConfig,
    LoggingConfig
)
from continual_learning_2.configs import CCBPConfig, AdamConfig, MLPConfig, TrainingConfig


def ccbp_mnist_experiment():
    SEED = 42
    start = time.time()
    optim_conf = CCBPConfig(
        tx=AdamConfig(learning_rate=1e-3),
        seed=SEED,
        decay_rate=0.9,
        replacement_rate=0.5,
        maturity_threshold=20,
    )

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
            run_name=f"ccbp_{SEED}",
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
    ccbp_mnist_experiment()
