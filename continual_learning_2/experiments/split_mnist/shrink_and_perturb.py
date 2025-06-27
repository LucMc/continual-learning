import os
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
    DatasetConfig,
    LoggingConfig,
    MLPConfig,
)
from continual_learning_2.configs import ShrinkAndPerterbConfig, AdamConfig


def shrink_and_perturb_mnist_experiment():
    SEED = 42

    start = time.time()

    optim_conf = ShrinkAndPerterbConfig(
        tx=AdamConfig(learning_rate=1e-3),
        param_noise_fn=x,
        seed=SEED,
        shrink=0.8,
        perturb=0.01,
        every_n=1
    )

    # Add validation to say what the available options are for dataset etc
    trainer = HeadResetClassificationCSLTrainer(
        seed=SEED,
        model_config=MLPConfig(output_size=10),
        optimizer_config=optim_conf,
        dataset_config=DatasetConfig(
            name="split_mnist",
            seed=SEED,
            batch_size=64,
            num_tasks=10,
            num_epochs_per_task=20,
            num_workers=0,  # (os.cpu_count() or 0) // 2,
        ),
        logging_config=LoggingConfig(
            run_name=optim_conf.__class__.__name__,
            wandb_entity="lucmc",
            wandb_project="split_mnist",
            wandb_mode="online",
            interval=100,
            eval_during_training=True,
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    shrink_and_perturb_mnist_experiment()
