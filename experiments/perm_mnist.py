import os
import time
from dataclasses import field
from typing import Literal

import jax
import tyro
from chex import dataclass

from continual_learning.configs import (
    AdamConfig,
    AdamwConfig,
    CbpConfig,
    CcbpConfig,
    DatasetConfig,
    LoggingConfig,
    MuonConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
    TrainingConfig,
)
from continual_learning.configs.models import MLPConfig
from continual_learning.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
)


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = ""
    wandb_entity: str = ""
    # data_dir: Path = Path("./experiment_results")
    resume: bool = False
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    postfix: str | None = None  # Postfix name tag
    base_optim: Literal["adam", "adamw", "muon"] = "adam"


def run_all_perm_mnist():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    base_optimizers = {
        "adam": AdamConfig(learning_rate=1e-3),
        "muon": MuonConfig(learning_rate=1e-3),
        "adamw": AdamwConfig(learning_rate=1e-3),
    }

    base_optim = base_optimizers[args.base_optim]

    optimizers = {
        "standard": base_optim,
        "regrama": RegramaConfig(
            tx=base_optim,
            update_frequency=1000,
            score_threshold=0.25,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "ccbp": CcbpConfig(
            tx=base_optim,
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.01,
            update_frequency=100,
        ),
        "redo": RedoConfig(
            tx=base_optim,
            update_frequency=1000,
            score_threshold=0.5,
            # score_threshold=0.001,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "cbp": CbpConfig(
            tx=base_optim,
            decay_rate=0.99,
            replacement_rate=1e-5,
            maturity_threshold=100,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=base_optim,
            param_noise_fn=jax.nn.initializers.he_uniform(),
            seed=args.seed,
            shrink=1 - 1e-5,
            perturb=1e-5,
            every_n=1,
        ),
    }

    if args.include:
        optimizers = {
            name: config for name, config in optimizers.items() if name in args.include
        }

    for algorithm in args.exclude:
        optimizers.pop(algorithm)

    print(f"Running algorithms: {list(optimizers.keys())}")

    exp_start = time.time()
    for opt_name, opt_conf in optimizers.items():
        print(f"Config: {opt_conf}")
        run_name = f"{opt_name}_{args.seed}"
        if args.postfix:
            run_name += f"_{args.postfix}"

        batch_size = 256
        # batch_size = 8

        start = time.time()
        trainer = HeadResetClassificationCSLTrainer(
            seed=args.seed,
            model_config=MLPConfig(output_size=10, hidden_size=128),
            optim_cfg=opt_conf,
            data_cfg=DatasetConfig(
                name="permuted_mnist",
                seed=args.seed,
                batch_size=batch_size,
                num_tasks=150,
                num_epochs_per_task=1,
                num_workers=(os.cpu_count() or 0) // 2,
            ),
            train_cfg=TrainingConfig(
                resume=args.resume,
            ),
            logs_cfg=LoggingConfig(
                run_name=run_name,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group="perm_mnist",
                wandb_mode=args.wandb_mode,
                interval=100,
                eval_during_training=True,
                eval_interval=16_000 // batch_size,
                sl_slow_metrics_batch_size=16_000,
            ),
        )
        trainer.train()
        print(f"Training time: {time.time() - start:.2f} seconds")
        del trainer
    print(f"Total training time: {time.time() - exp_start:.2f} seconds")


if __name__ == "__main__":
    run_all_perm_mnist()
