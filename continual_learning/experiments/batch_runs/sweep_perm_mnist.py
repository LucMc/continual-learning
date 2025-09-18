import itertools
import gc 
from typing import Any, Dict, Literal, Optional

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

from continual_learning_2.configs import *
from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.trainers.continual_supervised_learning import HeadResetClassificationCSLTrainer

# SWEEP_RANGES = {
#     "adam": {"learning_rate": [1e-3, 3e-4, 1e-4]},
#     "adamw": {"learning_rate": [1e-3, 3e-4, 1e-4]},
#     "muon": {"learning_rate": [1e-3, 3e-4, 1e-4]},
#
#     "regrama": {"tx_lr": [1e-3], "max_reset_frac": [None, 0.1], "update_frequency": [50, 100, 1000, 10_000], "score_threshold": [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,]},
#     "redo":    {"tx_lr": [1e-3], "max_reset_frac": [None, 0.1], "update_frequency": [50, 100, 1000, 10_000], "score_threshold": [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,]},
#     "cbp": {"tx_lr": [1e-3], "decay_rate": [0.95, 0.99], "replacement_rate": [1e-6, 1e-5, 1e-4], "maturity_threshold": [100, 1000]},
#     "ccbp": {"tx_lr": [1e-3], "decay_rate": [0., 0.99], "replacement_rate": [0.01, 0.05, 0.2], "update_frequency": [100, 1000]},
#     "shrink_and_perturb": {"tx_lr": [1e-3], "shrink": [1-1e-3, 1-1e-4, 1-1e-5], "perturb": [1e-3, 1e-4, 1e-5], "every_n": [1, 10, 100]},
# }


SWEEP_RANGES = {
    "adam": {"learning_rate": [1e-3, 3e-4, 1e-4]},
    "adamw": {"learning_rate": [1e-3, 3e-4, 1e-4]},
    "muon": {"learning_rate": [1e-3, 3e-4, 1e-4]},

    "regrama": {"tx_lr": [1e-3], "max_reset_frac": [None], "update_frequency": [100, 1000, 10_000], "score_threshold": [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,]},
    "redo":    {"tx_lr": [1e-3], "max_reset_frac": [None], "update_frequency": [100, 1000, 10_000], "score_threshold": [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,]},
    "cbp": {"tx_lr": [1e-3], "decay_rate": [0.95, 0.99], "replacement_rate": [1e-6, 1e-5, 1e-4], "maturity_threshold": [100, 1000]},
    "ccbp": {"tx_lr": [1e-3], "decay_rate": [0., 0.99], "replacement_rate": [0.01, 0.05, 0.2], "update_frequency": [100, 1000]},
    "shrink_and_perturb": {"tx_lr": [1e-3], "shrink": [1-1e-3, 1-1e-4, 1-1e-5], "perturb": [1e-3, 1e-4, 1e-5], "every_n": [1, 10, 100]},
}

def _all_configs_for(algo: str):
    """Return list of param dicts for all combinations in SWEEP_RANGES[algo]."""
    grid = list(itertools.product(*[[(k, v) for v in vals] for k, vals in SWEEP_RANGES[algo].items()]))
    return [dict(cfg) for cfg in grid]

def _format_tag(params: Dict[str, Any]) -> str:
    return ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())

def build_optimizer(algo: str, params: Dict[str, Any], seed: int):
    if algo == "adam":
        return AdamConfig(learning_rate=params["learning_rate"])
    elif algo == "adamw":
        return AdamwConfig(learning_rate=params["learning_rate"])
    elif algo == "muon":
        return MuonConfig(learning_rate=params["learning_rate"])
    
    tx = AdamConfig(learning_rate=params.get("tx_lr", 1e-3))
    configs = {
        "regrama": lambda: RegramaConfig(tx=tx, update_frequency=params["update_frequency"], score_threshold=params["score_threshold"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
        "redo": lambda: RedoConfig(tx=tx, update_frequency=params["update_frequency"], score_threshold=params["score_threshold"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
        "cbp": lambda: CbpConfig(tx=tx, decay_rate=params["decay_rate"], replacement_rate=params["replacement_rate"], maturity_threshold=params["maturity_threshold"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
        "ccbp": lambda: CcbpConfig(tx=tx, decay_rate=params["decay_rate"], replacement_rate=params["replacement_rate"], update_frequency=params["update_frequency"], seed=seed),
        "shrink_and_perturb": lambda: ShrinkAndPerterbConfig(tx=tx, param_noise_fn=jax.nn.initializers.he_uniform(), seed=seed, shrink=params["shrink"], perturb=params["perturb"], every_n=params["every_n"]),
    }
    return configs[algo]()

def run_config(algo: str, config_id: int, seed: int = 42, wandb_entity: str = None, wandb_project: str = None):
    configs = _all_configs_for(algo)  # UPDATED
    if config_id >= len(configs):
        print(f"Config ID {config_id} out of range for {algo} (max: {len(configs)-1})")
        return
    
    params = configs[config_id]
    tag = _format_tag(params)  # UPDATED
    
    opt_config = build_optimizer(algo, params, seed)
    
    trainer = HeadResetClassificationCSLTrainer(
        seed=seed,
        model_config=MLPConfig(output_size=10, hidden_size=128),
        optim_cfg=opt_config,
        data_cfg=DatasetConfig(
            name="permuted_mnist",
            seed=seed,
            batch_size=16,
            num_tasks=50,
            num_epochs_per_task=1,
            num_workers=0,
        ),
        train_cfg=TrainingConfig(
            resume=False,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"{algo}_{tag}_s{seed}",
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=f"perm_mnist_{algo}_sweep_16_50_2",
            wandb_mode="online" if wandb_project else "disabled",
            interval=100,
            eval_during_training=True,
        ),
    )
    trainer.train()

def list_configs(algo: str):
    configs = _all_configs_for(algo)  # UPDATED
    for i, params in enumerate(configs):
        tag = _format_tag(params)
        print(f"{i}: {tag}")
    print(f"Total configs: {len(configs)}")

def run_all_configs(
    algo: str,
    seed: int = 42,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
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
        print(f"\n=== [{cid}/{total-1}] {algo} :: {tag} ===")
        try:
            run_config(algo, cid, seed, wandb_entity, wandb_project)
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
    algo: Literal["adam", "adamw", "muon", "regrama", "redo", "cbp", "ccbp", "shrink_and_perturb"]
    config_id: Optional[int] = None
    seed: int = 42
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    list_configs: bool = False

    run_all: bool = False
    config_start: Optional[int] = None
    config_end: Optional[int] = None
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.list_configs:
        list_configs(args.algo)
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
                print("Error: config_id is required when not listing configs or running all configs")
                print("Use --list-configs to see available configurations, or pass --run-all")
                exit(1)
            run_config(args.algo, args.config_id, args.seed, args.wandb_entity, args.wandb_project)
