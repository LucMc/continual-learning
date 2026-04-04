"""Hyperparameter sweep for MinAtar Discrete SAC continual learning.

Usage:
    # List configs for an algorithm:
    python sweep_minatar.py --algo cpr --list-configs

    # Get SLURM array range:
    python sweep_minatar.py --algo cpr --get-count

    # Run a single config:
    python sweep_minatar.py --algo cpr --config-id 0 --seed 0 --wandb-entity myteam --wandb-project sweeps

    # Run all configs sequentially:
    python sweep_minatar.py --algo cpr --run-all --seed 0 --wandb-entity myteam --wandb-project sweeps
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
    CbpConfig,
    CprConfig,
    MuonConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.logging import LoggingConfig
from continual_learning.configs.models import CNNConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.discrete_sac_trainer import DiscreteSACTrainer
from continual_learning.types import Activation

# ---------------------------------------------------------------------------
# Sweep ranges — centred on defaults from experiments/minatar_discrete_sac.py
# ---------------------------------------------------------------------------
SWEEP_RANGES = {
    # adam default: lr=3e-4
    # "adam": {
    #     "learning_rate": [1e-3, 3e-4, 1e-4, 3e-5],
    # },  # 4 configs
    # muon default: lr=1e-4
    # "muon": {
    #     "learning_rate": [3e-4, 1e-4, 3e-5],
    # },  # 3 configs
    # redo defaults: lr=3e-4, update_frequency=50000, score_threshold=0.0001, max_reset_frac=0.02
    "redo": {
        "tx_lr": [3e-4],
        "update_frequency": [1_000, 10_000, 100_000],
        "score_threshold": [0.00005, 0.0001, 0.0005],
        "max_reset_frac": [None, 0.02],
    },  # 27 configs
    # regrama defaults: same as redo
    "regrama": {
        "tx_lr": [3e-4],
        "update_frequency": [1000, 10_000, 100_000],
        "score_threshold": [0.00005, 0.0001, 0.0005],
        "max_reset_frac": [None, 0.02],
    },  # 27 configs
    # cbp defaults: lr=3e-4, replacement_rate=1e-5, decay_rate=0.999, maturity_threshold=1000
    "cbp": {
        "tx_lr": [3e-4],
        "replacement_rate": [1e-6, 1e-5, 1e-4],
        "decay_rate": [0.99],
        "maturity_threshold": [1000, 10000, 100000],
    },  # 27 configs
    # cpr defaults: lr=3e-4, replacement_rate=0.005, decay_rate=0.99,
    #   sharpness=16 (FIXED), threshold=1.0 (FIXED), update_frequency=1000,
    #   transform_type="sigmoid" (FIXED)
    # More replacement_rate values as requested
    "cpr": {
        "tx_lr": [3e-4],
        "replacement_rate": [0.001, 0.003, 0.005, 0.01, 0.015, 0.02],
        "decay_rate": [0.99],
        "update_frequency": [1000, 5000, 10000, 100000],
        "transform_type": "sigmoid"
    },  # 30 configs
    # shrink_and_perturb defaults: lr=3e-4, shrink=0.9999, perturb=0.001, every_n=1000
    "shrink_and_perturb": {
        "tx_lr": [3e-4],
        "shrink": [0.999, 0.9999, 0.99999],
        "perturb": [0.0005, 0.001, 0.005],
        "every_n": [1000, 10000, 100000],
    },  # 27 configs
}


# ---------------------------------------------------------------------------
# Config enumeration
# ---------------------------------------------------------------------------
def _all_configs_for(algo: str):
    """Return list of param dicts for all combinations in SWEEP_RANGES[algo]."""
    grid = list(
        itertools.product(
            *[[(k, v) for v in vals] for k, vals in SWEEP_RANGES[algo].items()]
        )
    )
    return [dict(cfg) for cfg in grid]


def _format_tag(params: Dict[str, Any]) -> str:
    return ",".join(
        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
    )


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------
def build_optimizer(algo: str, params: Dict[str, Any], seed: int):
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
            replacement_rate=params["replacement_rate"],
            decay_rate=params["decay_rate"],
            maturity_threshold=params["maturity_threshold"],
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "cpr": lambda: CprConfig(
            tx=tx,
            replacement_rate=params["replacement_rate"],
            decay_rate=params["decay_rate"],
            update_frequency=params["update_frequency"],
            sharpness=16,
            threshold=1.0,
            transform_type="sigmoid",
            seed=seed,
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


# ---------------------------------------------------------------------------
# Run a single config
# ---------------------------------------------------------------------------
def run_config(
    algo: str,
    config_id: int,
    seed: int = 0,
    wandb_entity: str = None,
    wandb_project: str = None,
    wandb_mode: str = "online",
):
    configs = _all_configs_for(algo)
    if config_id >= len(configs):
        print(f"Config ID {config_id} out of range for {algo} (max: {len(configs) - 1})")
        return

    params = configs[config_id]
    tag = _format_tag(params)
    opt_config = build_optimizer(algo, params, seed)

    # CNN networks — matches experiments/minatar_discrete_sac.py
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

    sac_config = SACConfig(
        actor_config=PolicyNetworkConfig(optimizer=opt_config, network=actor_network),
        critic_config=QNetworkConfig(optimizer=opt_config, network=critic_network),
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True,
        replay_ratio=4,
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=5_000,
    )

    trainer = DiscreteSACTrainer(
        seed=seed,
        sac_config=sac_config,
        env_cfg=EnvConfig(
            name="minatar",
            num_envs=12,
            num_tasks=3,
            episode_length=1000,
        ),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=1_500_000,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"sweep_{algo}_{tag}_s{seed}",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=f"minatar_sweep_{algo}",
            save=False,
            wandb_mode=wandb_mode,
        ),
    )
    trainer.train()


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------
def list_configs(algo: str):
    configs = _all_configs_for(algo)
    for i, params in enumerate(configs):
        tag = _format_tag(params)
        print(f"{i}: {tag}")
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
    print(f"sbatch --array=0-{max_index} slurm_hyperparameter_sweep.sh {algo}")


def run_all_configs(
    algo: str,
    seed: int = 0,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_mode: str = "online",
    config_start: Optional[int] = None,
    config_end: Optional[int] = None,
):
    cfgs = _all_configs_for(algo)
    total = len(cfgs)
    start = 0 if config_start is None else max(0, config_start)
    end = total - 1 if config_end is None else min(total - 1, config_end)
    if start > end:
        print(f"Empty range: start ({start}) > end ({end}). Nothing to do.")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Args:
    algo: Literal[*list(SWEEP_RANGES.keys())]
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
            wandb_mode=args.wandb_mode,
            config_start=args.config_start,
            config_end=args.config_end,
        )
    else:
        if args.config_id is None:
            print("Error: config_id is required when not listing configs or running all configs")
            print("Use --list-configs to see available configurations, or pass --run-all")
            exit(1)
        run_config(
            args.algo, args.config_id, args.seed,
            args.wandb_entity, args.wandb_project, args.wandb_mode,
        )
