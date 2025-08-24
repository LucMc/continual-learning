import jax
import tyro
import time
from chex import dataclass
from typing import Literal
from continual_learning_2.trainers.continual_supervised_learning import (
    HeadResetClassificationCSLTrainer,
)
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs import (
    AdamConfig,
    CbpConfig,
    RedoConfig,
    RegramaConfig,
    CcbpConfig,
    ShrinkAndPerterbConfig,
    DatasetConfig,
    LoggingConfig,
    TrainingConfig,
)

from dataclasses import field


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    # data_dir: Path = Path("./experiment_results")
    resume: bool = False
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)


def run_all_perm_mnist():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    optimizers = {
        "adam": AdamConfig(learning_rate=1e-3),
        "regrama": RegramaConfig(
            tx=AdamConfig(learning_rate=1e-3),
            update_frequency=1000,
            score_threshold=0.0095,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "ccbp": CcbpConfig(
            tx=AdamConfig(learning_rate=1e-3),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.01,
            update_frequency=100,
        ),
        "redo": RedoConfig(
            tx=AdamConfig(learning_rate=1e-3),
            update_frequency=1000,
            score_threshold=0.025,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "cbp": CbpConfig(
            tx=AdamConfig(learning_rate=1e-3),
            decay_rate=0.99,
            replacement_rate=1e-5,
            maturity_threshold=100,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=AdamConfig(learning_rate=1e-3),
            param_noise_fn=jax.nn.initializers.he_uniform(),
            seed=args.seed,
            shrink=1-1e-5,
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
        start = time.time()
        trainer = HeadResetClassificationCSLTrainer(
            seed=args.seed,
            model_config=MLPConfig(output_size=10, hidden_size=128),
            optim_cfg=opt_conf,
            data_cfg=DatasetConfig(
                name="permuted_mnist",
                seed=args.seed,
                batch_size=1,
                num_tasks=250,
                num_epochs_per_task=1,
                num_workers=0,  # (os.cpu_count() or 0) // 2,
            ),
            train_cfg=TrainingConfig(
                resume=args.resume,
            ),
            logs_cfg=LoggingConfig(
                run_name=f"{opt_name}_{args.seed}",
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
        del trainer
    print(f"Total training time: {time.time() - exp_start:.2f} seconds")


if __name__ == "__main__":
    run_all_perm_mnist()
