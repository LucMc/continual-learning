"""Hyperparameter sweep for MetaWorld MT10 experiments.

This script supports sweeping hyperparameters for both SAC with reset methods
and BRO algorithm on MetaWorld MT10.

Usage:
    # List all configs for an algorithm
    python sweep_metaworld.py --algo regrama --list-configs

    # Get count for SLURM array
    python sweep_metaworld.py --algo regrama --get-count

    # Run a specific config
    python sweep_metaworld.py --algo regrama --config-id 0 --seed 42

    # Run all configs sequentially
    python sweep_metaworld.py --algo regrama --run-all --seed 42
"""

import gc
import itertools
from typing import Any, Dict, Literal, Optional

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

from continual_learning.configs import (
    AdamConfig,
    MuonConfig,
    CbpConfig,
    CcbpConfig,
    LoggingConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.sac_trainer import SACTrainer
from continual_learning.trainers.bro import BROTrainer
from continual_learning.trainers.bro_learner import BROConfig
from continual_learning.types import Activation, StdType


# Sweep ranges for each algorithm
SWEEP_RANGES = {
    "adam": {
        "seeds": [0, 1, 2, 3, 4],
        "learning_rate": [3e-4, 1e-4, 1e-3],
    },
    "muon": {
        "seeds": [0, 1, 2],
        "learning_rate": [1e-4, 1e-3, 1e-2],
    },
    "redo": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [3e-4],
        "update_frequency": [1000, 5000, 10000],
        "score_threshold": [0.01, 0.05, 0.1],
        "max_reset_frac": [0.02, 0.05, None],
    },
    "regrama": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [3e-4],
        "update_frequency": [1000, 5000, 10000],
        "score_threshold": [0.01, 0.05, 0.1],
        "max_reset_frac": [0.02, 0.05, None],
    },
    "cbp": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [3e-4],
        "decay_rate": [0.99, 0.999],
        "replacement_rate": [1e-6, 1e-5, 1e-4],
        "maturity_threshold": [100, 1000, 10000],
    },
    "ccbp": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [3e-4],
        "decay_rate": [0.9, 0.99],
        "replacement_rate": [0.001, 0.01],
        "sharpness": [8, 16],
        "threshold": [0.5, 0.9],
        "update_frequency": [1000, 5000],
        "transform_type": ["linear"],
    },
    "shrink_and_perturb": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [3e-4],
        "shrink": [0.999, 0.9999],
        "perturb": [0.001, 0.005],
        "every_n": [1000, 5000],
    },
    "bro": {
        "seeds": [0, 1, 2, 3, 4],
        "updates_per_step": [4, 10],
        "hidden_dims": [256],
        "depth": [1, 2],
        "n_quantiles": [100],
        "init_optimism": [0.5, 1.0, 2.0],
    },
}


def _all_configs_for(algo: str):
    """Return list of param dicts for all combinations in SWEEP_RANGES[algo]."""
    def _normalize_key(key: str) -> str:
        return "seed" if key == "seeds" else key

    grid = list(
        itertools.product(
            *[[(_normalize_key(k), v) for v in vals] for k, vals in SWEEP_RANGES[algo].items()]
        )
    )
    return [dict(cfg) for cfg in grid]


def _format_tag(params: Dict[str, Any]) -> str:
    return ",".join(
        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
    )


def build_sac_optimizer(algo: str, params: Dict[str, Any], seed: int):
    """Build optimizer config for SAC-based experiments."""
    if algo == "adam":
        return AdamConfig(learning_rate=params["learning_rate"])

    if algo == "muon":
        return MuonConfig(learning_rate=params["learning_rate"])

    tx = AdamConfig(learning_rate=params.get("tx_lr", 3e-4))

    configs = {
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
        "cbp": lambda: CbpConfig(
            tx=tx,
            decay_rate=params["decay_rate"],
            replacement_rate=params["replacement_rate"],
            maturity_threshold=params["maturity_threshold"],
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "ccbp": lambda: CcbpConfig(
            tx=tx,
            decay_rate=params["decay_rate"],
            replacement_rate=params.get("replacement_rate", 0.01),
            sharpness=params["sharpness"],
            threshold=params["threshold"],
            update_frequency=params["update_frequency"],
            transform_type=params.get("transform_type", "linear"),
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


def run_sac_config(
    algo: str,
    params: Dict[str, Any],
    seed: int,
    wandb_entity: str,
    wandb_project: str,
    wandb_mode: str = "online",
):
    """Run SAC experiment with given optimizer config."""
    opt_cfg = build_sac_optimizer(algo, params, seed)
    tag = _format_tag(params)

    actor_network = MLPConfig(
        num_layers=3,
        hidden_size=256,
        output_size=4,
        activation_fn=Activation.ReLU,
        kernel_init=jax.nn.initializers.he_uniform(),
        bias_init=jax.nn.initializers.zeros,
        dtype=jnp.float32,
    )

    critic_network = MLPConfig(
        num_layers=3,
        hidden_size=256,
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
        replay_ratio=4,
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=5000,
        reset_interval=None,
        use_layer_norm=True,
    )

    trainer = SACTrainer(
        seed=seed,
        sac_config=sac_config,
        env_cfg=EnvConfig(
            name="metaworld_mt10",
            num_envs=1,
            num_tasks=10,
            episode_length=500,
        ),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=1_000_000,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"sac_{algo}_{tag}_s{seed}",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=f"metaworld_sac_{algo}_sweep",
            save=False,
            wandb_mode=wandb_mode,
        ),
    )
    trainer.train()


def run_bro_config(
    params: Dict[str, Any],
    seed: int,
    wandb_entity: str,
    wandb_project: str,
    wandb_mode: str = "online",
):
    """Run BRO experiment with given config."""
    tag = _format_tag(params)

    bro_config = BROConfig(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        adj_lr=3e-5,
        discount=0.99,
        tau=0.005,
        init_temperature=1.0,
        init_optimism=params.get("init_optimism", 1.0),
        init_regularizer=0.25,
        pessimism=0.0,
        kl_target=0.05,
        std_multiplier=0.75,
        distributional=True,
        n_quantiles=params.get("n_quantiles", 100),
        hidden_dims=params.get("hidden_dims", 256),
        depth=params.get("depth", 1),
        updates_per_step=params.get("updates_per_step", 10),
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=5000,
        reset_steps=(15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001),
    )

    trainer = BROTrainer(
        seed=seed,
        bro_config=bro_config,
        env_cfg=EnvConfig(
            name="metaworld_mt10",
            num_envs=1,
            num_tasks=10,
            episode_length=500,
        ),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=500_000,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"bro_{tag}_s{seed}",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group="metaworld_bro_sweep",
            save=False,
            wandb_mode=wandb_mode,
        ),
    )
    trainer.train()


def run_config(
    algo: str,
    config_id: int,
    seed: int = 0,
    wandb_entity: str = None,
    wandb_project: str = None,
    wandb_mode: str = "online",
):
    """Run experiment for given algorithm and config ID."""
    configs = _all_configs_for(algo)
    if config_id >= len(configs):
        print(f"Config ID {config_id} out of range for {algo} (max: {len(configs) - 1})")
        return

    params = configs[config_id]
    base_seed = params.get("seed", 0)
    run_seed = base_seed + seed

    if algo == "bro":
        run_bro_config(params, run_seed, wandb_entity, wandb_project, wandb_mode)
    else:
        run_sac_config(algo, params, run_seed, wandb_entity, wandb_project, wandb_mode)


def list_configs(algo: str):
    """Print all configurations for an algorithm."""
    configs = _all_configs_for(algo)
    for i, params in enumerate(configs):
        tag = _format_tag(params)
        print(f"{i}: {tag}")
    print(f"Total configs: {len(configs)}")


def get_count(algo: str):
    """Print config count and SLURM array command."""
    configs = _all_configs_for(algo)
    total = len(configs)
    max_index = total - 1

    print(f"Algorithm: {algo}")
    print(f"Total configurations: {total}")
    print(f"SLURM array range: 0-{max_index}")
    print("")
    print("To submit the sweep, run:")
    print(f"sbatch --array=0-{max_index} slurm_hyperparameter_sweep.sh {algo} metaworld")


def run_all_configs(
    algo: str,
    seed: int = 0,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    config_start: Optional[int] = None,
    config_end: Optional[int] = None,
    wandb_mode: str = "online",
):
    """Run all configurations sequentially."""
    cfgs = _all_configs_for(algo)
    total = len(cfgs)
    start = 0 if config_start is None else max(0, config_start)
    end = total - 1 if config_end is None else min(total - 1, config_end)
    if start > end:
        print(f"Empty range: start ({start}) > end ({end}).")
        return

    print(f"Running {algo} configs {start}..{end} (total {end - start + 1} / {total})")
    for cid in range(start, end + 1):
        tag = _format_tag(cfgs[cid])
        print(f"\n=== [{cid}/{total - 1}] {algo} :: {tag} ===")
        try:
            run_config(algo, cid, seed, wandb_entity, wandb_project, wandb_mode)
        except KeyboardInterrupt:
            print("\nInterrupted by user; stopping sweep.")
            break
        except Exception as e:
            print(f"Config {cid} failed with error: {e}. Continuing.")
        finally:
            try:
                jax.clear_caches()
            except Exception:
                pass
            gc.collect()


@dataclass
class Args:
    algo: Literal["adam", "muon", "redo", "regrama", "cbp", "ccbp", "shrink_and_perturb", "bro"]
    config_id: Optional[int] = None
    seed: int = 0
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_mode: str = "online"
    list_configs: bool = False
    get_count: bool = False
    run_all: bool = False
    config_start: Optional[int] = None
    config_end: Optional[int] = None


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.list_configs:
        list_configs(args.algo)
    elif args.get_count:
        get_count(args.algo)
    elif args.run_all:
        run_all_configs(
            algo=args.algo,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            config_start=args.config_start,
            config_end=args.config_end,
            wandb_mode=args.wandb_mode,
        )
    else:
        if args.config_id is None:
            print("Error: config_id required when not listing or running all")
            print("Use --list-configs to see configurations, or --run-all")
            exit(1)
        run_config(
            args.algo, args.config_id, args.seed, args.wandb_entity, args.wandb_project, args.wandb_mode
        )
