import itertools
import gc
from typing import Any, Dict, Literal, Optional

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.logging import LoggingConfig
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.continual_rl import JittedContinualPPOTrainer
from continual_learning.types import Activation, StdType

SWEEP_RANGES = {
    "adam": {"learning_rate": [1e-3, 3e-4, 1e-4]},
    "regrama": {
        "tx_lr": [1e-3],
        "seeds": [0, 1, 2, 3, 4],
        # "update_frequency": [100, 1000, 5000, 10_000],
        "update_frequency": [100, 1000, 5000],
        "max_reset_frac": [None, 0.1],
        # "score_threshold": [0.001, 0.002, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.09, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75], # fmt: skip
        "score_threshold": [0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75],  # fmt: skip
    },
    # "regrama": {"tx_lr": [1e-3], "update_frequency": [100, 1000, 10_000, 100_000], "score_threshold": [0.003, 0.003]} # Added ones
    "redo": {
        "tx_lr": [1e-3],
        "seeds": [0, 1, 2, 3, 4, 5, 6],
        # "update_frequency": [100, 1000, 5000, 10_000],
        "update_frequency": [100],
        "max_reset_frac": [None],
        # "score_threshold": [0.001, 0.002, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.09, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75], # fmt: skip
        "score_threshold": [0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 0.75],  # fmt: skip
    },
    # "redo": {"tx_lr": [1e-3], "update_frequency": [100], "score_threshold": [0.000001, 0.00001, 0.0001, 0.002, 0.003, 0.004, 0.005, 0.02, 0.3]}, # Added regrama ones plus a few inbetweens
    "cbp": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [1e-3],
        "decay_rate": [0.95, 0.99],
        "replacement_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
        "maturity_threshold": [10, 100, 1000, 10_000],
    },

 "ccbp_exp": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [1e-3],
        "decay_rate": [0.9],
        "sharpness": [5.0, 15.0, 20.0],
        "threshold": [0.01, 0.1, 0.5, 0.95],
        "update_frequency": [1000],
        "replacement_rate": [0.01, 0.1],
        "transform_type": ["exp"],
    },
    "ccbp_sigmoid": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [1e-3],
        "decay_rate": [0.9],
        "sharpness": [12.0, 16.0, 24.0, 32.0],
        "threshold": [0.9, 0.92, 0.95, 0.96],
        "update_frequency": [1000],
        "replacement_rate": [0.01],
        "transform_type": ["sigmoid"],
    },
    "ccbp_softplus": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [1e-3],
        "decay_rate": [0.9],
        "sharpness": [10.0, 14.0, 18.0, 22.0],
        "threshold": [0.9, 0.93, 0.95, 0.97],
        "update_frequency": [1000],
        "replacement_rate": [0.01],
        "transform_type": ["softplus"],
    },
    "ccbp_linear": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [1e-3],
        "decay_rate": [0.9],
        "sharpness": [8.0, 12.0, 16.0],
        "threshold": [0.93, 0.95, 0.97],
        "update_frequency": [1000],
        "replacement_rate": [0.012, 0.015, 0.020],
        "transform_type": ["linear"],
    },
    "shrink_and_perturb": {
        "seeds": [0, 1, 2, 3, 4],
        "tx_lr": [1e-3],
        "shrink": [0.995, 0.999],
        "perturb": [0.002, 0.005, 0.01],
        "every_n": [10, 1000],
    },
}


def _all_configs_for(algo: str):
    """Return list of param dicts for all combinations in SWEEP_RANGES[algo]."""

    def _maybe_normalize_key(key: str) -> str:
        return "seed" if key == "seeds" else key

    grid = list(
        itertools.product(
            *[[(_maybe_normalize_key(k), v) for v in vals] for k, vals in SWEEP_RANGES[algo].items()]
        )
    )
    return [dict(cfg) for cfg in grid]


def _format_tag(params: Dict[str, Any]) -> str:
    return ",".join(
        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
    )


def build_optimizer(algo: str, params: Dict[str, Any], seed: int):
    if algo == "adam":
        return AdamConfig(learning_rate=params["learning_rate"])

    tx = AdamConfig(learning_rate=params.get("tx_lr", 1e-3))

    if algo in ("ccbp", "ccbp_exp", "ccbp_sigmoid", "ccbp_softplus", "ccbp_linear"):
        return CcbpConfig(
            tx=tx,
            decay_rate=params["decay_rate"],
            replacement_rate=params.get("replacement_rate"),
            sharpness=params["sharpness"],
            threshold=params["threshold"],
            update_frequency=params["update_frequency"],
            transform_type=params.get("transform_type"),
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        )

    configs = {
        "regrama": lambda: RegramaConfig(
            tx=tx,
            update_frequency=params["update_frequency"],
            score_threshold=params["score_threshold"],
            max_reset_frac=params.get("max_reset_frac"),
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
        "cbp": lambda: CbpConfig(
            tx=tx,
            decay_rate=params["decay_rate"],
            replacement_rate=params["replacement_rate"],
            maturity_threshold=params["maturity_threshold"],
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        # "ccbp": lambda: CcbpConfig(
        #     tx=tx,
        #     decay_rate=params["decay_rate"],
        #     sharpness=params["sharpness"],
        #     threshold=params["threshold"],
        #     update_frequency=params["update_frequency"],
        #     seed=seed,
        #     weight_init_fn=jax.nn.initializers.lecun_normal(),
        # ),
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
    wandb_entity: str = None,
    wandb_project: str = None,
    wandb_mode: str = "online",
):
    configs = _all_configs_for(algo)  # UPDATED
    if config_id >= len(configs):
        print(f"Config ID {config_id} out of range for {algo} (max: {len(configs) - 1})")
        return

    params = configs[config_id]
    base_seed = params.get("seed", 0)
    run_seed = base_seed + seed
    tag = _format_tag(params)
    opt_config = build_optimizer(algo, params, params["seed"] + seed)

    trainer = JittedContinualPPOTrainer(
        seed=run_seed,
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
        env_cfg=EnvConfig("slippery_ant", num_envs=2048, num_tasks=20, episode_length=1000),
            train_cfg=RLTrainingConfig(resume=False, steps_per_task=20_000_000),
        logs_cfg=LoggingConfig(
            run_name=f"{algo}_{tag}_s{run_seed}",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=f"slippery_ant_{algo}_sweep",
            save=False,
            wandb_mode=wandb_mode,
        ),
    )
    trainer.train()


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
    config_start: Optional[int] = None,
    config_end: Optional[int] = None,
    wandb_mode: str = "online",
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


@dataclass
class Args:
    algo: Literal[*list(SWEEP_RANGES.keys())]
    config_id: Optional[int] = None
    seed: int = 42
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
    else:
        if args.run_all:
            run_all_configs(
                algo=args.algo,
                seed=args.seed,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                config_start=args.config_start,
                config_end=args.config_end,
            )
        else:
            if args.config_id is None:
                print(
                    "Error: config_id is required when not listing configs or running all configs"
                )
                print("Use --list-configs to see available configurations, or pass --run-all")
                exit(1)
            run_config(
                args.algo, args.config_id, args.seed, args.wandb_entity, args.wandb_project
            )
