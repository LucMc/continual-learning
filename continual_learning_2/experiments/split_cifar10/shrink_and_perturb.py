import jax
import os
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
    DatasetConfig,
    LoggingConfig
)
from continual_learning_2.configs import ShrinkAndPerterbConfig, AdamConfig, MLPConfig, TrainingConfig
from continual_learning_2.configs.models import CNNConfig


def shrink_and_perturb_split_cifar10_experiment():
    SEED = 42
    start = time.time()
    optim_conf = ShrinkAndPerterbConfig(
        tx=AdamConfig(learning_rate=1e-3),
        param_noise_fn=jax.nn.initializers.he_uniform(),
        seed=SEED,
        shrink=0.8,
        perturb=0.01,
        every_n=1,
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
            num_tasks=5,
            num_epochs_per_task=1,
            # num_workers=0,  # (os.cpu_count() or 0) // 2,
            dataset_kwargs = {
                "flatten" : False
            }
        ),
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"shrink_and_perturb_{SEED}",
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
    shrink_and_perturb_split_cifar10_experiment()
