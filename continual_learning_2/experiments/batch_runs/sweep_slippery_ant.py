# sweep_slippery_ant.py
import itertools
import gc  # NEW
from typing import Any, Dict, Literal, Optional

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

from continual_learning_2.configs import *
from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning_2.configs.training import RLTrainingConfig
from continual_learning_2.trainers.continual_rl import JittedContinualPPOTrainer
from continual_learning_2.types import Activation, StdType

SWEEP_RANGES = {
    "adam": {"learning_rate": [1e-3, 3e-4, 1e-4]},
    "regrama": {"tx_lr": [1e-3], "update_frequency": [1000, 10_000], "score_threshold": [0.008, 0.0095, 0.011]},
    "redo": {"tx_lr": [1e-3], "update_frequency": [1000, 10_000], "score_threshold": [0.01, 0.05, 0.075, 0.1]},
    "cbp": {"tx_lr": [1e-3], "decay_rate": [0.95, 0.99], "replacement_rate": [1e-6, 1e-5, 1e-4], "maturity_threshold": [100, 1000]},
    "ccbp": {"tx_lr": [1e-3], "decay_rate": [0., 0.99], "sharpness": [5.0, 10.0, 15.0, 20.0], "threshold": [0.3, 0.5, 0.7, 1.0], "update_frequency": [1000], "replacement_rate": [0.01, 0.05, 0.2, 0.5, 1.0]},
    "shrink_and_perturb": {"tx_lr": [1e-3], "shrink": [0.995, 0.999], "perturb": [0.002, 0.005, 0.01], "every_n": [10, 1000]},
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
    
    tx = AdamConfig(learning_rate=params.get("tx_lr", 1e-3))
    configs = {
        "regrama": lambda: RegramaConfig(tx=tx, update_frequency=params["update_frequency"], score_threshold=params["score_threshold"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
        "redo": lambda: RedoConfig(tx=tx, update_frequency=params["update_frequency"], score_threshold=params["score_threshold"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
        "cbp": lambda: CbpConfig(tx=tx, decay_rate=params["decay_rate"], replacement_rate=params["replacement_rate"], maturity_threshold=params["maturity_threshold"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
        "ccbp": lambda: CcbpConfig(tx=tx, decay_rate=params["decay_rate"], sharpness=params["sharpness"], threshold=params["threshold"], update_frequency=params["update_frequency"], seed=seed, weight_init_fn=jax.nn.initializers.he_uniform()),
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
    
    trainer = JittedContinualPPOTrainer(
        seed=seed,
        ppo_config=PPOConfig(
            policy_config=PolicyNetworkConfig(
                optimizer=opt_config,
                network=MLPConfig(num_layers=4, hidden_size=32, output_size=8, activation_fn=Activation.Swish, kernel_init=jax.nn.initializers.lecun_normal(), dtype=jnp.float32),
                std_type=StdType.MLP_HEAD,
            ),
            vf_config=ValueFunctionConfig(
                optimizer=opt_config,
                network=MLPConfig(num_layers=5, hidden_size=256, output_size=1, activation_fn=Activation.Swish, kernel_init=jax.nn.initializers.lecun_normal(), dtype=jnp.float32),
            ),
            num_rollout_steps=2048 * 32 * 3, num_epochs=4, num_gradient_steps=32, gamma=0.97, gae_lambda=0.95,
            entropy_coefficient=1e-3, clip_eps=0.2, vf_coefficient=0.5, normalize_advantages=True,
        ),
        env_cfg=EnvConfig("slippery_ant", num_envs=2048, num_tasks=20, episode_length=1000),
        train_cfg=RLTrainingConfig(resume=False, steps_per_task=20_000_000),
        logs_cfg=LoggingConfig(
            run_name=f"{algo}_{tag}_s{seed}",
            wandb_entity=wandb_entity, wandb_project=wandb_project,
            group=f"slippery_ant_{algo}_sweep", save=False, wandb_mode="online" if wandb_project else "disabled",
        ),
    )
    trainer.train()

def list_configs(algo: str):
    configs = _all_configs_for(algo)  # UPDATED
    for i, params in enumerate(configs):
        tag = _format_tag(params)
        print(f"{i}: {tag}")
    print(f"Total configs: {len(configs)}")

# NEW: run a whole sweep sequentially for one algorithm
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
            # Help avoid memory buildup between long runs
            try:
                jax.clear_caches()
            except Exception:
                pass
            gc.collect()


@dataclass
class Args:
    algo: Literal["adam", "regrama", "redo", "cbp", "ccbp", "shrink_and_perturb"]
    config_id: Optional[int] = None
    seed: int = 42
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    list_configs: bool = False

    # NEW flags:
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
