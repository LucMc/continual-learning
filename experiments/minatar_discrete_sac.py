"""Discrete SAC on MinAtar continual learning benchmark.

Chains 3 tasks: SpaceInvaders-MinAtar → Asterix-MinAtar → Seaquest-MinAtar
Observations padded to (10, 10, 10), flattened to 1000 dims.
Total training: 4.5M steps = 1.5M per task (default, matches reference).

Single-task mode: pass --task <game> to train on one game only.
"""

import time
from dataclasses import field
from typing import Literal

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

import continual_learning.envs.minatar as minatar_module
from continual_learning.configs import (
    AdamConfig,
    CbpConfig,
    CprConfig,
    LoggingConfig,
    MuonConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.models import CNNConfig, MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.discrete_sac_trainer import DiscreteSACTrainer
from continual_learning.types import Activation

TASK_LOOKUP = {name: (name, ch) for name, ch in minatar_module.TASK_SPECS}


@dataclass(frozen=True)
class Args:
    """Command line arguments for MinAtar Discrete SAC experiment."""

    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = ""
    wandb_entity: str = ""
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)

    # Single-task mode: set to a MinAtar game name (space_invaders, asterix, seaquest)
    task: str = ""

    # SAC settings
    replay_ratio: int = 4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    # 3 tasks × 1.5M = 4.5M total (matches reference budget)
    steps_per_task: int = 1_500_000
    num_envs: int = 12

    # Network architecture
    network: Literal["mlp", "cnn"] = "cnn"
    hidden_size: int = 256
    num_layers: int = 2


def run_minatar_discrete_sac():
    """Run Discrete SAC on MinAtar continual benchmark with various optimizers."""
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project, "wandb_project required when wandb is enabled"
        assert args.wandb_entity, "wandb_entity required when wandb is enabled"

    # Single-task mode: patch TASK_SPECS before any env is created
    if args.task:
        assert args.task in TASK_LOOKUP, (
            f"Unknown task {args.task!r}, choose from {list(TASK_LOOKUP)}"
        )
        minatar_module.TASK_SPECS = [TASK_LOOKUP[args.task]]
        num_tasks = 1
    else:
        num_tasks = 3

    lr = 3e-4
    muon_lr = 1e-4

    optimizers = {
        "adam": AdamConfig(learning_rate=lr),
        "redo": RedoConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=50_000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "regrama": RegramaConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=50_000,
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
        "cpr": CprConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.005,
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
        "muon": MuonConfig(learning_rate=muon_lr),
    }

    if args.include:
        optimizers = {k: v for k, v in optimizers.items() if k in args.include}
    for name in args.exclude:
        optimizers.pop(name, None)

    print(f"Running optimizers: {list(optimizers.keys())}")
    print(f"Replay ratio: {args.replay_ratio}")
    print(f"Tasks: {[name for name, _ in minatar_module.TASK_SPECS]}")
    print(f"Steps per task: {args.steps_per_task} ({num_tasks} tasks × {args.steps_per_task} = "
          f"{num_tasks * args.steps_per_task:,} total)")

    exp_start = time.time()

    for opt_name, opt_cfg in optimizers.items():
        print(f"\n{'='*60}")
        print(f"Starting experiment with optimizer: {opt_name}")
        print(f"{'='*60}")

        start = time.time()

        # output_size is overridden inside CategoricalPolicy/DiscreteQNetwork
        # to n_actions/MAX_N_ACTIONS at init time — set to 1 as placeholder
        if args.network == "cnn":
            actor_network = CNNConfig(
                output_size=1,
                features=(16,),
                num_convs_per_layer=1,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                use_max_pooling=False,
                dense_hidden_size=128,
                num_dense_layers=1,
                activation_fn=Activation.ReLU,
                kernel_init=jax.nn.initializers.he_uniform(),
                bias_init=jax.nn.initializers.zeros,
                dtype=jnp.float32,
            )
            critic_network = CNNConfig(
                output_size=1,
                features=(16,),
                num_convs_per_layer=1,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                use_max_pooling=False,
                dense_hidden_size=128,
                num_dense_layers=1,
                activation_fn=Activation.ReLU,
                kernel_init=jax.nn.initializers.he_uniform(),
                bias_init=jax.nn.initializers.zeros,
                dtype=jnp.float32,
            )
        else:
            actor_network = MLPConfig(
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                output_size=1,
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

        task_tag = f"_{args.task}" if args.task else ""
        group = f"minatar_single_{args.task}" if args.task else "minatar_discrete_sac"

        sac_config = SACConfig(
            actor_config=PolicyNetworkConfig(
                optimizer=opt_cfg,
                network=actor_network,
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
        )

        trainer = DiscreteSACTrainer(
            seed=args.seed,
            sac_config=sac_config,
            env_cfg=EnvConfig(
                name="minatar",
                num_envs=args.num_envs,
                num_tasks=num_tasks,
                episode_length=1000,
            ),
            train_cfg=RLTrainingConfig(
                resume=False,
                steps_per_task=args.steps_per_task,
            ),
            logs_cfg=LoggingConfig(
                run_name=f"discrete_sac_minatar{task_tag}_{opt_name}_{args.network}_LOW_{args.seed}",
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group=group,
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
    run_minatar_discrete_sac()
