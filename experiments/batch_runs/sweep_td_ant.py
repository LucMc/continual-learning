"""Hyperparameter sweep for time-delayed Ant continual learning.

Usage:
    # List configs for an algorithm:
    python sweep_td_ant.py --algo cpr --list-configs

    # Get SLURM array range:
    python sweep_td_ant.py --algo cpr --get-count

    # Run a single config:
    python sweep_td_ant.py --algo cpr --config-id 0 --seed 0 \
        --wandb-entity myteam --wandb-project sweeps

    # Run all configs sequentially:
    python sweep_td_ant.py --algo cpr --run-all --seed 0 \
        --wandb-entity myteam --wandb-project sweeps
"""

import gc
import itertools
import sys
from dataclasses import field
from typing import Any, Dict, Literal, Optional

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
from continual_learning.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.envs.slippery_mujoco import ContinualDelayedAnt
from continual_learning.trainers.ppo_trainer import JittedContinualPPOTrainer
from continual_learning.types import Activation, StdType


FREQUENCIES = [100, 1000, 10000]


SWEEP_RANGES = {
    # CPR defaults for TD Ant: lr=1e-3, decay_rate=0.99, sharpness=16,
    # threshold=1, transform_type="sigmoid"; replacement rates centered on 0.015.
    "cpr": {
        "tx_lr": [1e-3],
        "decay_rate": [0.99],
        "replacement_rate": [0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03],
        "update_frequency": FREQUENCIES,
    },
    # CBP keeps decay fixed and sweeps replacement around the TD Ant starting point.
    "cbp": {
        "tx_lr": [1e-3],
        "decay_rate": [0.99],
        "replacement_rate": [0.0003, 0.001, 0.003, 0.01],
        "maturity_threshold": FREQUENCIES,
    },
    # ReDo/ReGraMa thresholds include the TD Ant defaults and nearby values.
    "redo": {
        "tx_lr": [1e-3],
        "update_frequency": FREQUENCIES,
        "score_threshold": [0.05, 0.075, 0.1, 0.15],
        "max_reset_frac": [None, 0.05],
    },
    "regrama": {
        "tx_lr": [1e-3],
        "update_frequency": FREQUENCIES,
        "score_threshold": [0.1, 0.15, 0.2, 0.25],
        "max_reset_frac": [None, 0.05],
    },
    "shrink_and_perturb": {
        "tx_lr": [1e-3],
        "shrink": [0.995, 0.999, 0.9999],
        "perturb": [0.001, 0.005, 0.01],
        "every_n": FREQUENCIES,
    },
}


def _rewrite_legacy_argv():
    """Support older queue scripts that pass positional sweep arguments."""
    argv = ["--list-configs" if arg == "--list" else arg for arg in sys.argv[1:]]
    if argv and not argv[0].startswith("-") and argv[0] in SWEEP_RANGES:
        algo = argv.pop(0)
        rewritten = ["--algo", algo]
        positional_flags = ["--config-id", "--seed", "--wandb-entity", "--wandb-project"]

        pos_idx = 0
        while argv and not argv[0].startswith("-") and pos_idx < len(positional_flags):
            rewritten.extend([positional_flags[pos_idx], argv.pop(0)])
            pos_idx += 1

        argv = rewritten + argv

    sys.argv[1:] = argv


def _all_configs_for(algo: str):
    """Return all parameter combinations in ``SWEEP_RANGES[algo]``."""
    grid = itertools.product(
        *[[(key, value) for value in values] for key, values in SWEEP_RANGES[algo].items()]
    )
    return [dict(cfg) for cfg in grid]


def _format_tag(params: Dict[str, Any]) -> str:
    return ",".join(
        f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in params.items()
    )


def _alternating_ramp_schedule(num_increments: int) -> list[tuple[int, int]]:
    schedule = [(0, 0)]
    obs, act = 0, 0
    for idx in range(num_increments):
        if idx % 2 == 0:
            obs += 1
        else:
            act += 1
        schedule.append((obs, act))
    return schedule


def _build_delay_config(
    *,
    delay_mode: str,
    overall_max_obs_delay: int,
    overall_max_act_delay: int,
    fixed_obs_delay: int,
    fixed_act_delay: int,
    ramp_num_increments: int,
    num_tasks: int,
):
    fixed_obs = range(fixed_obs_delay, fixed_obs_delay + 1)
    fixed_act = range(fixed_act_delay, fixed_act_delay + 1)
    ramp_schedule: list[tuple[int, int]] | None = None

    if delay_mode == "ramp":
        ramp_schedule = _alternating_ramp_schedule(ramp_num_increments)
        max_obs = max(obs for obs, _ in ramp_schedule)
        max_act = max(act for _, act in ramp_schedule)
        overall_obs = range(0, max_obs + 1)
        overall_act = range(0, max_act + 1)
        resolved_num_tasks = len(ramp_schedule)
    else:
        overall_obs = range(0, overall_max_obs_delay)
        overall_act = range(0, overall_max_act_delay)
        resolved_num_tasks = num_tasks

    if delay_mode == "fixed":
        if not (overall_obs.start <= fixed_obs.start and fixed_obs.stop <= overall_obs.stop):
            raise ValueError(
                f"fixed_obs_delay={fixed_obs_delay} not in [0, {overall_max_obs_delay})"
            )
        if not (overall_act.start <= fixed_act.start and fixed_act.stop <= overall_act.stop):
            raise ValueError(
                f"fixed_act_delay={fixed_act_delay} not in [0, {overall_max_act_delay})"
            )

    return overall_obs, overall_act, fixed_obs, fixed_act, ramp_schedule, resolved_num_tasks


def _delay_group(
    *,
    delay_mode: str,
    delay_info_mode: str,
    overall_obs: range,
    overall_act: range,
    fixed_obs_delay: int,
    fixed_act_delay: int,
    ramp_schedule: list[tuple[int, int]] | None,
    ramp_num_increments: int,
) -> str:
    info_suffix = "" if delay_info_mode == "one_hot" else f"_info-{delay_info_mode}"

    if delay_mode == "fixed":
        return f"{fixed_act_delay}act{fixed_obs_delay}obs_fixed{info_suffix}"
    if delay_mode == "ramp":
        assert ramp_schedule is not None
        return (
            f"ramp{ramp_num_increments}"
            f"_obs{max(obs for obs, _ in ramp_schedule)}"
            f"_act{max(act for _, act in ramp_schedule)}"
            f"{info_suffix}"
        )
    if delay_mode == "task_boundary_constant":
        return (
            f"obs{overall_obs.stop - 1}_act{overall_act.stop - 1}"
            f"_task_boundary_const{info_suffix}"
        )
    return f"obs{overall_obs.stop - 1}_act{overall_act.stop - 1}_task_boundary{info_suffix}"


def build_optimizer(algo: str, params: Dict[str, Any], seed: int):
    tx = AdamConfig(learning_rate=params.get("tx_lr", 1e-3))

    configs = {
        "cpr": lambda: CprConfig(
            tx=tx,
            decay_rate=params["decay_rate"],
            replacement_rate=params["replacement_rate"],
            sharpness=16,
            threshold=1,
            update_frequency=params["update_frequency"],
            transform_type="sigmoid",
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "cbp": lambda: CbpConfig(
            tx=tx,
            decay_rate=params["decay_rate"],
            replacement_rate=params["replacement_rate"],
            maturity_threshold=params["maturity_threshold"],
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "redo": lambda: RedoConfig(
            tx=tx,
            update_frequency=params["update_frequency"],
            score_threshold=params["score_threshold"],
            max_reset_frac=params.get("max_reset_frac"),
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "regrama": lambda: RegramaConfig(
            tx=tx,
            update_frequency=params["update_frequency"],
            score_threshold=params["score_threshold"],
            max_reset_frac=params.get("max_reset_frac"),
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "shrink_and_perturb": lambda: ShrinkAndPerterbConfig(
            tx=tx,
            param_noise_fn=jax.nn.initializers.lecun_normal(),
            seed=seed,
            shrink=params["shrink"],
            perturb=params["perturb"],
            every_n=params["every_n"],
        ),
    }
    return configs[algo]()


def run_config(
    algo: str,
    config_id: int,
    seed: int = 0,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_mode: str = "online",
    overall_max_obs_delay: int = 11,
    overall_max_act_delay: int = 11,
    delay_mode: str = "task_boundary",
    fixed_obs_delay: int = 0,
    fixed_act_delay: int = 0,
    ramp_num_increments: int = 20,
    delay_info_mode: str = "one_hot",
    num_tasks: int = 10,
    steps_per_task: int = 40_000_000,
    num_envs: int = 2048,
    episode_length: int = 1000,
):
    configs = _all_configs_for(algo)
    if config_id >= len(configs):
        print(f"Config ID {config_id} out of range for {algo} (max: {len(configs) - 1})")
        return

    params = configs[config_id]
    tag = _format_tag(params)
    opt_config = build_optimizer(algo, params, seed)

    (
        overall_obs,
        overall_act,
        fixed_obs,
        fixed_act,
        ramp_schedule,
        resolved_num_tasks,
    ) = _build_delay_config(
        delay_mode=delay_mode,
        overall_max_obs_delay=overall_max_obs_delay,
        overall_max_act_delay=overall_max_act_delay,
        fixed_obs_delay=fixed_obs_delay,
        fixed_act_delay=fixed_act_delay,
        ramp_num_increments=ramp_num_increments,
        num_tasks=num_tasks,
    )

    group = _delay_group(
        delay_mode=delay_mode,
        delay_info_mode=delay_info_mode,
        overall_obs=overall_obs,
        overall_act=overall_act,
        fixed_obs_delay=fixed_obs_delay,
        fixed_act_delay=fixed_act_delay,
        ramp_schedule=ramp_schedule,
        ramp_num_increments=ramp_num_increments,
    )

    env_cfg = EnvConfig(
        name="delayed_ant",
        num_envs=num_envs,
        num_tasks=resolved_num_tasks,
        episode_length=episode_length,
    )
    benchmark = ContinualDelayedAnt(
        seed=seed,
        config=env_cfg,
        overall_obs_delay_range=overall_obs,
        overall_act_delay_range=overall_act,
        delay_mode=delay_mode,
        fixed_obs_delay_range=fixed_obs if delay_mode == "fixed" else None,
        fixed_act_delay_range=fixed_act if delay_mode == "fixed" else None,
        ramp_schedule=ramp_schedule,
        delay_info_mode=delay_info_mode,
    )

    trainer = JittedContinualPPOTrainer(
        seed=seed,
        ppo_config=PPOConfig(
            policy_config=PolicyNetworkConfig(
                optimizer=opt_config,
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
                optimizer=opt_config,
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
            steps_per_task=steps_per_task,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"sweep_td_ant_{algo}_{tag}_{group}_s{seed}",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=f"td_ant_{group}_{algo}_sweep",
            save=False,
            wandb_mode=wandb_mode,
        ),
        benchmark=benchmark,
    )
    trainer.train()


def list_configs(algo: str):
    configs = _all_configs_for(algo)
    for idx, params in enumerate(configs):
        print(f"{idx}: {_format_tag(params)}")
    print(f"Total configs: {len(configs)}")


def get_count(algo: str):
    configs = _all_configs_for(algo)
    total = len(configs)
    max_index = total - 1

    print(f"Algorithm: {algo}")
    print(f"Total configurations: {total}")
    print(f"SLURM array range: 0-{max_index}")
    print("")
    print("To submit the sweep, run:")
    print(f"sbatch --array=0-{max_index} slurm_hyperparameter_sweep.sh {algo} td_ant")


def run_all_configs(
    algo: str,
    seed: int = 0,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_mode: str = "online",
    config_start: Optional[int] = None,
    config_end: Optional[int] = None,
    overall_max_obs_delay: int = 11,
    overall_max_act_delay: int = 11,
    delay_mode: str = "task_boundary",
    fixed_obs_delay: int = 0,
    fixed_act_delay: int = 0,
    ramp_num_increments: int = 20,
    delay_info_mode: str = "one_hot",
    num_tasks: int = 10,
    steps_per_task: int = 40_000_000,
    num_envs: int = 2048,
    episode_length: int = 1000,
):
    configs = _all_configs_for(algo)
    total = len(configs)
    start = 0 if config_start is None else max(0, config_start)
    end = total - 1 if config_end is None else min(total - 1, config_end)
    if start > end:
        print(f"Empty range: start ({start}) > end ({end}). Nothing to do.")
        return

    print(f"Running {algo} configs {start}..{end} (total {end - start + 1} / {total})")
    for config_id in range(start, end + 1):
        tag = _format_tag(configs[config_id])
        print(f"\n=== [{config_id}/{total - 1}] {algo} :: {tag} ===")
        try:
            run_config(
                algo=algo,
                config_id=config_id,
                seed=seed,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                wandb_mode=wandb_mode,
                overall_max_obs_delay=overall_max_obs_delay,
                overall_max_act_delay=overall_max_act_delay,
                delay_mode=delay_mode,
                fixed_obs_delay=fixed_obs_delay,
                fixed_act_delay=fixed_act_delay,
                ramp_num_increments=ramp_num_increments,
                delay_info_mode=delay_info_mode,
                num_tasks=num_tasks,
                steps_per_task=steps_per_task,
                num_envs=num_envs,
                episode_length=episode_length,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user; stopping sweep.")
            break
        except Exception as exc:
            print(f"Config {config_id} failed with error: {exc}. Continuing.")
        finally:
            try:
                jax.clear_caches()
            except Exception:
                pass
            gc.collect()


@dataclass
class Args:
    algo: Literal[*list(SWEEP_RANGES.keys())]
    config_id: Optional[int] = None
    seed: int = 0
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    # TD Ant defaults: delays in {0, ..., 10}, sampled at task boundaries.
    overall_max_obs_delay: int = 11
    overall_max_act_delay: int = 11
    delay_mode: Literal["fixed", "task_boundary", "task_boundary_constant", "ramp"] = (
        "task_boundary"
    )
    fixed_obs_delay: int = 0
    fixed_act_delay: int = 0
    ramp_num_increments: int = 20
    delay_info_mode: Literal["one_hot", "scalar", "none", "blind"] = "one_hot"

    num_tasks: int = 10
    steps_per_task: int = 40_000_000
    num_envs: int = 2048
    episode_length: int = 1000

    list_configs: bool = False
    get_count: bool = False

    run_all: bool = False
    config_start: Optional[int] = None
    config_end: Optional[int] = None

    # Compatibility aliases for older local queue scripts.
    list: bool = field(default=False, metadata={"help": "Alias for --list-configs."})


if __name__ == "__main__":
    _rewrite_legacy_argv()
    args = tyro.cli(Args)

    if args.list_configs or args.list:
        list_configs(args.algo)
    elif args.get_count:
        get_count(args.algo)
    elif args.run_all:
        run_all_configs(
            algo=args.algo,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
            config_start=args.config_start,
            config_end=args.config_end,
            overall_max_obs_delay=args.overall_max_obs_delay,
            overall_max_act_delay=args.overall_max_act_delay,
            delay_mode=args.delay_mode,
            fixed_obs_delay=args.fixed_obs_delay,
            fixed_act_delay=args.fixed_act_delay,
            ramp_num_increments=args.ramp_num_increments,
            delay_info_mode=args.delay_info_mode,
            num_tasks=args.num_tasks,
            steps_per_task=args.steps_per_task,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    else:
        if args.config_id is None:
            print(
                "Error: config_id is required when not listing configs or running all "
                "configs"
            )
            print("Use --list-configs to see available configurations, or pass --run-all")
            raise SystemExit(1)
        run_config(
            algo=args.algo,
            config_id=args.config_id,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
            overall_max_obs_delay=args.overall_max_obs_delay,
            overall_max_act_delay=args.overall_max_act_delay,
            delay_mode=args.delay_mode,
            fixed_obs_delay=args.fixed_obs_delay,
            fixed_act_delay=args.fixed_act_delay,
            ramp_num_increments=args.ramp_num_increments,
            delay_info_mode=args.delay_info_mode,
            num_tasks=args.num_tasks,
            steps_per_task=args.steps_per_task,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
