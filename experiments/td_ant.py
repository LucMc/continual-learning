"""Time-delayed Ant continual-learning experiment.

Mirrors ``experiments/slippery_ant.py`` (JIT PPO, optimizer sweep) but
replaces the friction-varied tasks with a Brax-native time-delay wrapper
whose obs/act delay sub-interval is sampled per task. See
``continual_learning/envs/brax_delay_wrapper.py`` for the wrapper and
``ContinualDelayedAnt`` in ``continual_learning/envs/slippery_mujoco.py`` for
the benchmark.

Examples
--------
0,0 fixed (sanity baseline)::

    python experiments/td_ant.py \
        --overall-max-obs-delay 1 --overall-max-act-delay 1 \
        --fixed-obs-delay 0 --fixed-act-delay 0 \
        --delay-mode fixed --num-tasks 2 --steps-per-task 20000000

1,1 fixed (degradation check)::

    python experiments/td_ant.py \
        --overall-max-obs-delay 2 --overall-max-act-delay 2 \
        --fixed-obs-delay 1 --fixed-act-delay 1 \
        --delay-mode fixed --num-tasks 2 --steps-per-task 20000000

Continual [0,8] (plasticity)::

    python experiments/td_ant.py \
        --overall-max-obs-delay 9 --overall-max-act-delay 9 \
        --delay-mode task_boundary --num-tasks 20 --steps-per-task 20000000
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
    CprConfig,
    LoggingConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import (
    PolicyNetworkConfig,
    PPOConfig,
    ValueFunctionConfig,
)
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.envs.slippery_mujoco import ContinualDelayedAnt
from continual_learning.trainers.ppo_trainer import JittedContinualPPOTrainer
from continual_learning.types import Activation, StdType


OPTIMIZER_NAMES = ("adam", "regrama", "cpr", "redo", "cbp", "shrink_and_perturb")


@dataclass(frozen=True)
class Args:
    """CLI args for the time-delayed Ant experiment."""

    seed: int = 42

    # Delay schedule
    overall_max_obs_delay: int = 1  # exclusive upper bound for obs delays
    overall_max_act_delay: int = 1  # exclusive upper bound for action delays
    delay_mode: Literal["fixed", "task_boundary"] = "fixed"
    fixed_obs_delay: int = 0  # used iff mode == "fixed"; alpha is sampled in {fixed_obs_delay}
    fixed_act_delay: int = 0  # used iff mode == "fixed"; kappa is sampled in {fixed_act_delay}

    # Training schedule
    num_tasks: int = 2
    steps_per_task: int = 20_000_000
    num_envs: int = 2048
    episode_length: int = 1000

    # Optimizers — comma-separated names from OPTIMIZER_NAMES; default "adam"
    optimizers: list[str] = field(default_factory=lambda: ["adam"])

    # W&B
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = ""
    wandb_entity: str = ""


def _get_optimizer_config(name: str, seed: int, lr: float = 1e-3):
    if name == "adam":
        return AdamConfig(learning_rate=lr)
    if name == "regrama":
        return RegramaConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=1000,
            score_threshold=0.0095,
            max_reset_frac=0.05,
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        )
    if name == "cpr":
        return CprConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=seed,
            decay_rate=0.9,
            sharpness=10,
            threshold=0.5,
            update_frequency=1000,
            transform_type="linear",
        )
    if name == "redo":
        return RedoConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=1000,
            score_threshold=0.055,
            max_reset_frac=0.05,
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        )
    if name == "cbp":
        return CbpConfig(
            tx=AdamConfig(learning_rate=lr),
            decay_rate=0.99,
            replacement_rate=0.0002,
            maturity_threshold=100,
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        )
    if name == "shrink_and_perturb":
        return ShrinkAndPerterbConfig(
            param_noise_fn=jax.nn.initializers.lecun_normal(),
            tx=AdamConfig(learning_rate=lr),
            seed=seed,
            shrink=1 - 0.001,
            perturb=0.005,
            every_n=1000,
        )
    raise ValueError(f"Unknown optimizer name: {name!r}")


def run():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project, "--wandb-project is required unless --wandb-mode disabled"
        assert args.wandb_entity, "--wandb-entity is required unless --wandb-mode disabled"

    for name in args.optimizers:
        if name not in OPTIMIZER_NAMES:
            raise ValueError(
                f"Unknown optimizer {name!r}; choose from {OPTIMIZER_NAMES}"
            )

    overall_obs = range(0, args.overall_max_obs_delay)
    overall_act = range(0, args.overall_max_act_delay)
    fixed_obs = range(args.fixed_obs_delay, args.fixed_obs_delay + 1)
    fixed_act = range(args.fixed_act_delay, args.fixed_act_delay + 1)

    if args.delay_mode == "fixed":
        if not (overall_obs.start <= fixed_obs.start and fixed_obs.stop <= overall_obs.stop):
            raise ValueError(
                f"fixed_obs_delay={args.fixed_obs_delay} not in [0, "
                f"{args.overall_max_obs_delay})"
            )
        if not (overall_act.start <= fixed_act.start and fixed_act.stop <= overall_act.stop):
            raise ValueError(
                f"fixed_act_delay={args.fixed_act_delay} not in [0, "
                f"{args.overall_max_act_delay})"
            )

    # W&B group: matches the td_mt1 convention so 0,0 / 1,1 / continual auto-cluster
    if args.delay_mode == "fixed":
        group = f"{args.fixed_act_delay}act{args.fixed_obs_delay}obs_fixed"
    else:
        group = (
            f"obs{args.overall_max_obs_delay - 1}_act{args.overall_max_act_delay - 1}"
            f"_task_boundary"
        )

    env_cfg = EnvConfig(
        name="delayed_ant",
        num_envs=args.num_envs,
        num_tasks=args.num_tasks,
        episode_length=args.episode_length,
    )

    exp_start = time.time()
    for opt_name in args.optimizers:
        start = time.time()
        opt_conf = _get_optimizer_config(opt_name, args.seed)

        benchmark = ContinualDelayedAnt(
            seed=args.seed,
            config=env_cfg,
            overall_obs_delay_range=overall_obs,
            overall_act_delay_range=overall_act,
            delay_mode=args.delay_mode,
            fixed_obs_delay_range=fixed_obs if args.delay_mode == "fixed" else None,
            fixed_act_delay_range=fixed_act if args.delay_mode == "fixed" else None,
        )

        trainer = JittedContinualPPOTrainer(
            seed=args.seed,
            ppo_config=PPOConfig(
                policy_config=PolicyNetworkConfig(
                    optimizer=opt_conf,
                    network=MLPConfig(
                        num_layers=4,
                        hidden_size=32,
                        output_size=8,
                        activation_fn=Activation.Swish,
                        kernel_init=jax.nn.initializers.lecun_normal(),
                        dtype=jnp.float32,
                    ),
                    std_type=StdType.MLP_HEAD,
                ),
                vf_config=ValueFunctionConfig(
                    optimizer=opt_conf,
                    network=MLPConfig(
                        num_layers=5,
                        hidden_size=256,
                        output_size=1,
                        activation_fn=Activation.Swish,
                        kernel_init=jax.nn.initializers.lecun_normal(),
                        dtype=jnp.float32,
                    ),
                ),
                num_rollout_steps=2048 * 32 * 3,
                num_epochs=4,
                num_gradient_steps=32,
                gamma=0.97,
                gae_lambda=0.95,
                entropy_coefficient=1e-3,
                clip_eps=0.2,
                vf_coefficient=0.5,
                normalize_advantages=True,
            ),
            env_cfg=env_cfg,
            train_cfg=RLTrainingConfig(
                resume=False,
                steps_per_task=args.steps_per_task,
            ),
            logs_cfg=LoggingConfig(
                run_name=(
                    f"td_ant_{opt_name}_obs{args.overall_max_obs_delay - 1}"
                    f"_act{args.overall_max_act_delay - 1}_{args.delay_mode}_s{args.seed}"
                ),
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group=group,
                save=False,
                wandb_mode=args.wandb_mode,
            ),
            benchmark=benchmark,
        )
        trainer.train()
        print(f"[{opt_name}] training time: {time.time() - start:.2f}s")
        del trainer

    print(f"Total experiment time: {time.time() - exp_start:.2f}s")


if __name__ == "__main__":
    run()
