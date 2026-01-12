"""MetaWorld MT10 experiment with BRO algorithm.

This script runs the full BRO (Bigger, Regularized, Optimistic) algorithm
on MetaWorld MT10 in a continual learning setting.

BRO features:
- Distributional critic with quantile regression (100 quantiles)
- Conservative + Optimistic dual actor system
- Learnable temperature, optimism, and regularizer coefficients
- BroNet architecture with LayerNorm and residual connections
- Fixed reset schedule for high replay ratio training
- Support for reset methods (REDO, ReGrAMA, CBP, CCBP, ShrinkAndPerturb)

Usage:
    # Quick smoke test with AdamW (default)
    python -m experiments.metaworld_mt10 --wandb_mode disabled

    # Run with a specific optimizer/reset method
    python -m experiments.metaworld_mt10 --wandb_mode disabled --optimizer redo

    # Run with wandb logging
    python -m experiments.metaworld_mt10 --wandb_project metaworld_bro --wandb_entity your_entity
"""

import time
from typing import Literal

import tyro
from chex import dataclass

from continual_learning.configs import (
    AdamwConfig,
    CbpConfig,
    CcbpConfig,
    LoggingConfig,
    RedoConfig,
    RegramaConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.models.rl import orthogonal_init
from continual_learning.trainers.bro import BROTrainer
from continual_learning.trainers.bro_learner import BROConfig


@dataclass(frozen=True)
class Args:
    """Command line arguments for MetaWorld MT10 experiment."""

    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = ""
    wandb_entity: str = ""

    # Optimizer selection (supports reset methods)
    optimizer: Literal["adamw", "redo", "regrama", "cbp", "ccbp"] = "adamw"

    # BRO hyperparameters
    updates_per_step: int = 10  # High replay ratio (BRO default)
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    steps_per_task: int = 500_000  # Steps per MetaWorld task

    # Network architecture
    hidden_dims: int = 256
    depth: int = 1  # BRO uses depth 1 by default

    # BRO-specific
    distributional: bool = True
    n_quantiles: int = 100
    init_temperature: float = 1.0
    init_optimism: float = 1.0
    init_regularizer: float = 0.25
    kl_target: float = 0.05
    std_multiplier: float = 0.75


def get_optimizer_config(name: str, seed: int):
    """Get optimizer config by name."""
    lr = 3e-4
    weight_init = orthogonal_init()

    optimizers = {
        "adamw": AdamwConfig(learning_rate=lr),
        "redo": RedoConfig(
            tx=AdamwConfig(learning_rate=lr),
            update_frequency=5000,
            score_threshold=0.01,
            max_reset_frac=0.02,
            seed=seed,
            weight_init_fn=weight_init,
        ),
        "regrama": RegramaConfig(
            tx=AdamwConfig(learning_rate=lr),
            update_frequency=5000,
            score_threshold=0.01,
            max_reset_frac=0.02,
            seed=seed,
            weight_init_fn=weight_init,
        ),
        "cbp": CbpConfig(
            tx=AdamwConfig(learning_rate=lr),
            decay_rate=0.99,
            replacement_rate=0.0002,
            maturity_threshold=1000,
            seed=seed,
            weight_init_fn=weight_init,
        ),
        "ccbp": CcbpConfig(
            tx=AdamwConfig(learning_rate=lr),
            seed=seed,
            decay_rate=0.9,
            sharpness=10,
            threshold=0.5,
            update_frequency=5000,
            transform_type="linear",
        ),
    }
    return optimizers[name]


def run_metaworld_mt10():
    """Run BRO on MetaWorld MT10."""
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project, "wandb_project required when wandb is enabled"
        assert args.wandb_entity, "wandb_entity required when wandb is enabled"

    print("=" * 60)
    print("BRO (Bigger, Regularized, Optimistic) on MetaWorld MT10")
    print("=" * 60)
    print(f"Optimizer: {args.optimizer}")
    print(f"Updates per step: {args.updates_per_step}")
    print(f"Steps per task: {args.steps_per_task}")
    print(f"Distributional: {args.distributional}")
    print(f"N quantiles: {args.n_quantiles}")
    print(f"Hidden dims: {args.hidden_dims}, Depth: {args.depth}")
    print()

    start = time.time()

    # Get optimizer config
    opt_cfg = get_optimizer_config(args.optimizer, args.seed)

    # Create BRO config
    bro_config = BROConfig(
        # Optimizer configs (supports reset methods)
        actor_optimizer=opt_cfg,
        critic_optimizer=opt_cfg,
        # Learning rates for scalar params
        temp_lr=3e-4,
        adj_lr=3e-5,  # Lower for adjustment coefficients
        # SAC parameters
        discount=0.99,
        tau=0.005,
        # BRO-specific
        init_temperature=args.init_temperature,
        init_optimism=args.init_optimism,
        init_regularizer=args.init_regularizer,
        pessimism=0.0,
        kl_target=args.kl_target,
        std_multiplier=args.std_multiplier,
        # Distributional RL
        distributional=args.distributional,
        n_quantiles=args.n_quantiles,
        # Network architecture
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        # Training
        updates_per_step=args.updates_per_step,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        # Reset schedule (BRO default)
        reset_steps=(15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001),
    )

    # Create trainer
    trainer = BROTrainer(
        seed=args.seed,
        bro_config=bro_config,
        env_cfg=EnvConfig(
            name="metaworld_mt10",
            num_envs=1,  # MetaWorld typically uses single env
            num_tasks=10,  # MT10 has 10 tasks
            episode_length=500,  # MetaWorld default
        ),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=args.steps_per_task,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"bro_{args.optimizer}_{args.seed}",
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            group="metaworld_mt10_bro",
            save=False,
            wandb_mode=args.wandb_mode,
        ),
    )

    # Train
    trainer.train()

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.1f}s")


if __name__ == "__main__":
    run_metaworld_mt10()
