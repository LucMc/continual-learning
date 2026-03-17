"""MetaWorld MT10 experiment with SAC and reset methods.

This script runs SAC on MetaWorld MT10 with various reset methods for comparison:
- Adam (baseline)
- REDO (Reset Dead Outputs)
- ReGrAMA (Reset via Gradient Moving Average)
- CBP (Continual Backpropagation)
- CCBP (Continuous CBP)
- ShrinkAndPerturb

Usage:
    # Quick smoke test with Adam only
    python -m experiments.metaworld_sac --wandb_mode disabled --include adam

    # Run all optimizers
    python -m experiments.metaworld_sac --wandb_project metaworld_sac --wandb_entity your_entity

    # Run specific optimizers
    python -m experiments.metaworld_sac --include adam redo regrama
"""

import time
from dataclasses import field
from typing import Literal

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

from continual_learning.configs import (
    AdamConfig,
    CbpConfig,
    CcbpConfig,
    LoggingConfig,
    MuonConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.sac_trainer import SACTrainer
from continual_learning.types import Activation, StdType


@dataclass(frozen=True)
class Args:
    """Command line arguments for MetaWorld SAC experiment."""

    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = ""
    wandb_entity: str = ""
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)

    # SAC settings
    replay_ratio: int = 4  # Gradient updates per env step
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    steps_per_task: int = 1_000_000
    num_envs: int = 10  # Match MT1 async envs for more transitions per step

    # Network architecture
    hidden_size: int = 512
    num_layers: int = 3


def run_metaworld_sac():
    """Run SAC on MetaWorld MT10 with various reset methods."""
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project, "wandb_project required when wandb is enabled"
        assert args.wandb_entity, "wandb_entity required when wandb is enabled"

    lr = 3e-4
    muon_lr = 0.01

    # Define optimizers with conservative hyperparameters
    optimizers = {
        "adam": AdamConfig(learning_rate=lr),
        "redo": RedoConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=100000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "regrama": RegramaConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=100000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "cbp": CbpConfig(
            tx=AdamConfig(learning_rate=lr),
            replacement_rate=1e-5,
            decay_rate=0.999,
            maturity_threshold=1000,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "ccbp": CcbpConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.015,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "ccbpl": CcbpConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.0015,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "ccbph": CcbpConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.15,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=args.seed,
            shrink=0.9999,
            perturb=0.001,
            every_n=1000,
            param_noise_fn=jax.nn.initializers.lecun_normal(),
        ),
        # Muon-based optimizers
        "muon": MuonConfig(learning_rate=muon_lr),
        "muon_redo": RedoConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            update_frequency=100000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "muon_regrama": RegramaConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            update_frequency=100000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "muon_cbp": CbpConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            replacement_rate=1e-5,
            decay_rate=0.999,
            maturity_threshold=1000,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "muon_ccbp": CcbpConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.015,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "muon_ccbpl": CcbpConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.0015,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "muon_ccbph": CcbpConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.15,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "muon_shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=MuonConfig(learning_rate=muon_lr),
            seed=args.seed,
            shrink=0.9999,
            perturb=0.001,
            every_n=1000,
            param_noise_fn=jax.nn.initializers.lecun_normal(),
        ),
    }

    # Filter optimizers
    if args.include:
        optimizers = {
            name: cfg for name, cfg in optimizers.items() if name in args.include
        }
    for name in args.exclude:
        optimizers.pop(name, None)

    print(f"Running optimizers: {list(optimizers.keys())}")
    print(f"Replay ratio: {args.replay_ratio}")
    print(f"Steps per task: {args.steps_per_task}")

    exp_start = time.time()

    for opt_name, opt_cfg in optimizers.items():
        print(f"\n{'='*60}")
        print(f"Starting experiment with optimizer: {opt_name}")
        print(f"{'='*60}")

        start = time.time()

        # Network configs
        actor_network = MLPConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            output_size=4,  # MetaWorld action dim
            activation_fn=Activation.ReLU,
            kernel_init=jax.nn.initializers.he_uniform(),
            bias_init=jax.nn.initializers.zeros,
            dtype=jnp.float32,
        )

        critic_network = MLPConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            output_size=1,
            activation_fn=Activation.ReLU,
            kernel_init=jax.nn.initializers.he_uniform(),
            bias_init=jax.nn.initializers.zeros,
            dtype=jnp.float32,
        )

        sac_config = SACConfig(
            actor_config=PolicyNetworkConfig(
                optimizer=opt_cfg,
                network=actor_network,
                min_std=1e-6,
                var_scale=1.0,
                std_type=StdType.MLP_HEAD,
            ),
            critic_config=QNetworkConfig(
                optimizer=opt_cfg,
                network=critic_network,
            ),
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_entropy=True,
            replay_ratio=args.replay_ratio,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            reset_interval=None,
            use_layer_norm=True,
        )

        trainer = SACTrainer(
            seed=args.seed,
            sac_config=sac_config,
            env_cfg=EnvConfig(
                name="metaworld_mt10",
                num_envs=args.num_envs,
                num_tasks=10,
                episode_length=500,
            ),
            train_cfg=RLTrainingConfig(
                resume=False,
                steps_per_task=args.steps_per_task,
            ),
            logs_cfg=LoggingConfig(
                run_name=f"sac_{opt_name}_{args.seed}",
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group="metaworld_mt10_sac",
                save=False,
                wandb_mode=args.wandb_mode,
            ),
        )

        trainer.train()

        elapsed = time.time() - start
        print(f"\nTraining with {opt_name} completed in {elapsed:.1f}s")

        del trainer

    total_time = time.time() - exp_start
    print(f"\n{'='*60}")
    print(f"All experiments completed in {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_metaworld_sac()
