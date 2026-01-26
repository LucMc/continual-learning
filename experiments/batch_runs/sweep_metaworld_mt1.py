"""Hyperparameter sweep for MetaWorld MT1 single-task experiments.

This script sweeps reset method hyperparameters for single-task RL on MetaWorld.
The ranges are LESS aggressive than MT10 sweeps since single-task RL doesn't
require the same level of plasticity maintenance as continual learning.

Paper-informed sweep ranges:
- CBP: Lower replacement rates (1e-6 to 1e-4), higher maturity (1000-10000)
- ReDo/ReGrAMA: Higher thresholds (0.05-0.5), less frequent updates (5000-20000)
- Shrink & Perturb: Minimal shrink (0.9999+), small perturb, infrequent
- CCBP: High threshold (0.9+), low replacement (0.001-0.01)

Usage:
    # Get config count for SLURM
    python sweep_metaworld_mt1.py --algo cbp --get-count

    # Run specific config
    python sweep_metaworld_mt1.py --algo cbp --config-id 0 --task reach-v3 --seed 0

    # List all configs
    python sweep_metaworld_mt1.py --algo regrama --list-configs
"""

import itertools
import time
from typing import Any, Dict, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from chex import dataclass

from continual_learning.configs import (
    AdamConfig,
    CbpConfig,
    CcbpConfig,
    LoggingConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, SACConfig

# Backwards compatibility: Use NetworkConfig if QNetworkConfig not available
try:
    from continual_learning.configs.rl import QNetworkConfig
except ImportError:
    from continual_learning.configs.rl import NetworkConfig as QNetworkConfig
from continual_learning.envs.metaworld import MetaWorldSingleTaskEnv
from continual_learning.trainers.sac import SAC
from continual_learning.types import Activation, StdType
from continual_learning.utils.monitoring import Logger, prefix_dict
from continual_learning.utils.replay_buffer import ReplayBuffer


MT10_TASKS = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
]

# Sweep ranges based on paper defaults (comprehensive analysis)
#
# Reference values from original papers:
#
# ReDo (Sokar et al. 2023):
#   - τ (dormancy threshold): {0, 0.01, 0.025, 0.05, 0.1, 0.2}
#   - Optimal: 0.1 (Atari/discrete), 0.025-0.05 (MuJoCo/continuous)
#   - Check interval: {100, 1000, 2000, 5000, 10000}, typical 1000-2000
#   - CRITICAL: Must reset optimizer moments for recycled neurons!
#
# ReGraMa (Liu et al. 2025):
#   - α (gradient threshold): {1e-5, 1e-4, 0.001, 0.01}, optimal 1e-4
#   - reset_rate (max_reset_frac): {0.001, 0.01, 0.1}, optimal 0.01
#   - More effective than ReDo on ResNets/deep architectures
#
# CBP (Dohare et al. 2024, Nature):
#   - ρ (replacement rate): {0, 1e-5, 1e-4, 1e-3}, optimal 1e-4
#   - Maturity threshold m: {50, 100, 200, 500, 1000, 2000}
#   - Optimal for RL: 1000-2000 (due to sparse/noisy gradients)
#   - Utility decay η: 0.99 or 0.999
#
SWEEP_RANGES = {
    # Baseline - just learning rate
    "adam": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
    },
    # ReDo (Sokar et al. 2023): Activation-based dormancy detection
    # For single-task RL: less frequent resets, conservative thresholds
    # max_reset_frac caps neurons reset per cycle to prevent instability
    "redo": {
        "update_frequency": [10000, 100000],
        "score_threshold": [0.0001, 0.001, 0.01],
        "max_reset_frac": [0.02, 0.05],
    },
    # ReGraMa (Liu et al. 2025): Gradient magnitude-based resets
    # Uses gradient norm instead of activation - better for deep nets
    # max_reset_frac caps neurons reset per cycle to prevent instability
    "regrama": {
        "update_frequency": [10000, 100000],
        "score_threshold": [0.0001, 0.001, 0.01],
        "max_reset_frac": [0.02, 0.05],
    },
    # CBP (Dohare et al. 2024): Stochastic continuous replacement
    # ρ (replacement_rate): optimal 1e-4 for RL
    # Maturity: 1000-2000 for RL (protects neurons from noisy gradients)
    "cbp": {
        "replacement_rate": [1e-5, 1e-4, 1e-3],  # Paper: optimal 1e-4
        "decay_rate": [0.99, 0.999],  # Utility decay η
        "maturity_threshold": [100, 500, 1000, 2000],  # Paper values
    },
    # CCBP: Threshold-based CBP variant with utility scoring
    "ccbp": {
        "decay_rate": [0.9],
        "replacement_rate": [0.0001, 0.001, 0.01, 0.1, 0.125, 0.15],
        "sharpness": [16],  # Fixed at paper optimal
        "threshold": [1.0],  # Fixed
        "update_frequency": [1, 1000, 5000],
        "transform_type": ["sigmoid"],
    },
    # Shrink and Perturb (Ash & Adams 2020)
    # Paper: noise σ ∈ {0.01, 0.1} for S&P baseline
    "shrink_and_perturb": {
        "shrink": [0.99, 0.999, 0.9999],
        "perturb": [0.001, 0.01, 0.1],  # Paper σ values
        "every_n": [1000, 5000, 10000],
    },
}


def _all_configs_for(algo: str):
    """Return list of param dicts for all combinations in SWEEP_RANGES[algo]."""
    grid = list(
        itertools.product(
            *[[(k, v) for v in vals] for k, vals in SWEEP_RANGES[algo].items()]
        )
    )
    return [dict(cfg) for cfg in grid]


def _format_tag(params: Dict[str, Any]) -> str:
    """Format params as a compact string tag."""
    parts = []
    for k, v in params.items():
        if v is None:
            parts.append(f"{k}=None")
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return ",".join(parts)


def build_optimizer(algo: str, params: Dict[str, Any], seed: int):
    """Build optimizer config from algorithm name and parameters."""
    if algo == "adam":
        return AdamConfig(learning_rate=params["learning_rate"])

    # Fixed learning rate for all reset methods
    tx = AdamConfig(learning_rate=3e-4)

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
            replacement_rate=params["replacement_rate"],
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


def make_sac_config(opt_cfg, hidden_size: int = 256, num_layers: int = 3) -> SACConfig:
    """Create SACConfig with given optimizer."""
    actor_network = MLPConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        output_size=4,  # MetaWorld action dim
        activation_fn=Activation.ReLU,
        kernel_init=jax.nn.initializers.he_uniform(),
        bias_init=jax.nn.initializers.zeros,
        dtype=jnp.float32,
    )

    critic_network = MLPConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        output_size=1,
        activation_fn=Activation.ReLU,
        kernel_init=jax.nn.initializers.he_uniform(),
        bias_init=jax.nn.initializers.zeros,
        dtype=jnp.float32,
    )

    return SACConfig(
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


def evaluate(env, sac_state, key, num_episodes: int = 10):
    """Evaluate the policy deterministically."""
    obs = env.init()
    episode_returns = []
    episode_successes = []

    current_return = np.zeros(env.num_envs)

    while len(episode_returns) < num_episodes:
        dist = sac_state.actor.apply_fn(sac_state.actor.params, obs)
        try:
            action = dist.mode()
        except NotImplementedError:
            base_mean = dist.distribution.loc
            action = jnp.tanh(base_mean)
        action = jnp.clip(action, -1.0, 1.0)

        timestep = env.step(action)

        current_return += np.asarray(timestep.reward).squeeze(-1)

        dones = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)
        for i, done in enumerate(dones):
            if done:
                episode_returns.append(float(current_return[i]))
                success_list = timestep.info.get("success", [False] * env.num_envs)
                episode_successes.append(success_list[i] if i < len(success_list) else False)
                current_return[i] = 0

        obs = timestep.next_observation

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
    }


def run_config(
    algo: str,
    config_id: int,
    task: str,
    seed: int = 0,
    total_steps: int = 5_000_000,
    num_envs: int = 10,
    async_envs: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_mode: str = "online",
):
    """Run single MT1 experiment with given config."""
    configs = _all_configs_for(algo)
    if config_id >= len(configs):
        print(f"Config ID {config_id} out of range for {algo} (max: {len(configs) - 1})")
        return

    params = configs[config_id]
    tag = _format_tag(params)

    print(f"{'='*60}")
    print(f"MT1 Sweep: {algo} config {config_id}")
    print(f"Task: {task}, Seed: {seed}")
    print(f"Num envs: {num_envs} (async={async_envs})")
    print(f"Total steps: {total_steps}")
    print(f"Params: {tag}")
    print(f"{'='*60}")

    # Build optimizer and SAC config
    opt_cfg = build_optimizer(algo, params, seed)
    sac_config = make_sac_config(opt_cfg)

    # Create environment
    print("Initializing environment...")
    env = MetaWorldSingleTaskEnv(
        task_name=task, num_envs=num_envs, seed=seed, async_envs=async_envs
    )
    print(f"  Obs dim: {env.obs_dim}, Action dim: {env.action_dim}")

    # Initialize logger
    run_name = f"mt1_{task}_{algo}_cfg{config_id}_s{seed}"
    logger = Logger(
        LoggingConfig(
            run_name=run_name,
            wandb_entity=wandb_entity or "",
            wandb_project=wandb_project or "MT1_sweep",
            group=f"mt1_{algo}_sweep",
            save=False,
            wandb_mode=wandb_mode,
        ),
        run_config={
            "algorithm": "sac",
            "task": task,
            "optimizer": algo,
            "config_id": config_id,
            "seed": seed,
            "total_steps": total_steps,
            "num_envs": num_envs,
            "async_envs": async_envs,
            **params,
        },
    )

    # Initialize SAC
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    sac_state = SAC.init_state(
        key=init_key,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        cfg=sac_config,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        capacity=sac_config.buffer_size,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
    )
    buffer_state = replay_buffer.init()

    target_entropy = -float(env.action_dim)

    # Training loop
    total_env_steps = 0
    total_episodes = 0
    total_gradient_steps = 0

    episode_rewards: list[float] = []
    episode_successes: list[bool] = []
    current_episode_reward = np.zeros(num_envs)
    current_episode_length = np.zeros(num_envs, dtype=int)

    obs = env.init()
    start_time = time.time()
    last_log_step = 0
    last_eval_step = 0

    print("Starting training...")

    while total_env_steps < total_steps:
        # Select action
        key, action_key = jax.random.split(key)
        dist = sac_state.actor.apply_fn(sac_state.actor.params, obs)
        action = dist.sample(seed=action_key)
        action = jnp.clip(action, -1.0, 1.0)
        action = jnp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # Step environment
        timestep = env.step(action)

        # Add to buffer
        buffer_state = ReplayBuffer.add(
            buffer_state,
            obs=obs,
            action=action,
            reward=timestep.reward,
            next_obs=timestep.next_observation,
            done=timestep.terminated | timestep.truncated,
        )

        # Track episode stats
        rewards_np = np.asarray(timestep.reward).squeeze(-1)
        dones_np = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)

        current_episode_reward += rewards_np
        current_episode_length += 1

        for i, done in enumerate(dones_np):
            if done:
                episode_rewards.append(float(current_episode_reward[i]))
                success_list = timestep.info.get("success", [False] * num_envs)
                episode_successes.append(success_list[i] if i < len(success_list) else False)
                current_episode_reward[i] = 0
                current_episode_length[i] = 0
                total_episodes += 1

        obs = timestep.next_observation
        total_env_steps += num_envs  # Count all env steps

        # Update SAC
        all_logs = []
        if total_env_steps >= sac_config.learning_starts:
            for _ in range(sac_config.replay_ratio):
                key, sample_key = jax.random.split(key)
                batch = ReplayBuffer.sample(buffer_state, sample_key, sac_config.batch_size)
                sac_state, logs = SAC.update(sac_state, batch, sac_config, target_entropy)
                all_logs.append(logs)
                total_gradient_steps += 1

        # Log metrics
        if total_env_steps - last_log_step >= 1000:
            elapsed = time.time() - start_time
            sps = total_env_steps / max(elapsed, 1e-6)

            log_dict = {
                "charts/total_steps": total_env_steps,
                "charts/total_episodes": total_episodes,
                "charts/total_gradient_steps": total_gradient_steps,
                "charts/SPS": sps,
            }

            if episode_rewards:
                log_dict["charts/mean_episode_return"] = float(np.mean(episode_rewards[-100:]))

            if episode_successes:
                log_dict["charts/success_rate"] = float(np.mean(episode_successes[-100:]))

            if all_logs:
                avg_logs = {}
                for log_key in all_logs[0].keys():
                    values = [float(l[log_key]) for l in all_logs if log_key in l]
                    if values:
                        avg_logs[log_key] = float(np.mean(values))
                log_dict.update(prefix_dict("train", avg_logs))

            logger.log(log_dict, step=total_env_steps)
            last_log_step = total_env_steps

            mean_return = log_dict.get("charts/mean_episode_return", 0)
            success_rate = log_dict.get("charts/success_rate", 0)
            print(
                f"Step {total_env_steps:>7}, Eps: {total_episodes:>4}, "
                f"Return: {mean_return:>7.2f}, Success: {success_rate:>5.2%}, "
                f"SPS: {sps:>6.1f}",
                flush=True,
            )

        # Periodic evaluation
        if total_env_steps - last_eval_step >= 10_000:
            eval_metrics = evaluate(env, sac_state, key, num_episodes=20)
            eval_log = {
                "eval/mean_return": eval_metrics["mean_return"],
                "eval/success_rate": eval_metrics["success_rate"],
            }
            logger.log(eval_log, step=total_env_steps)
            last_eval_step = total_env_steps
            print(
                f"  [Eval] Return: {eval_metrics['mean_return']:.2f}, "
                f"Success: {eval_metrics['success_rate']:.2%}",
                flush=True,
            )

    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = evaluate(env, sac_state, key, num_episodes=50)
    final_log = {
        "final/mean_return": final_metrics["mean_return"],
        "final/std_return": final_metrics["std_return"],
        "final/success_rate": final_metrics["success_rate"],
    }
    logger.log(final_log, step=total_env_steps)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total steps: {total_env_steps}")
    print(f"Final success rate: {final_metrics['success_rate']:.2%}")
    print(f"Final mean return: {final_metrics['mean_return']:.2f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    logger.close()
    env.close()

    return final_metrics


def list_configs(algo: str):
    """Print all configurations for an algorithm."""
    configs = _all_configs_for(algo)
    for i, params in enumerate(configs):
        tag = _format_tag(params)
        print(f"{i}: {tag}")
    print(f"\nTotal configs: {len(configs)}")


def get_count(algo: str):
    """Print config count and SLURM array command."""
    configs = _all_configs_for(algo)
    total = len(configs)
    max_index = total - 1

    # Calculate total jobs across all tasks and seeds
    num_tasks = len(MT10_TASKS)
    num_seeds = 3  # Default 3 seeds

    print(f"Algorithm: {algo}")
    print(f"Configs per (task, seed): {total}")
    print(f"Tasks: {num_tasks}")
    print(f"Seeds: {num_seeds}")
    print(f"Total jobs: {total * num_tasks * num_seeds}")
    print(f"\nConfig ID range: 0-{max_index}")
    print(f"\nTo run sweep for one task with 3 seeds:")
    print(f"  sbatch --array=0-{max_index * num_seeds - 1} slurm_mt1_sweep.sh {algo} reach-v3")


def get_total_sweep_size():
    """Print total sweep size across all algorithms."""
    print("Sweep sizes by algorithm:")
    print("-" * 40)
    total = 0
    for algo in SWEEP_RANGES:
        configs = _all_configs_for(algo)
        n = len(configs)
        total += n
        print(f"  {algo:20s}: {n:4d} configs")
    print("-" * 40)
    print(f"  {'TOTAL':20s}: {total:4d} configs")
    print(f"\nWith 10 tasks × 3 seeds = {total * 10 * 3} total jobs")


@dataclass
class Args:
    algo: Optional[Literal["adam", "redo", "regrama", "cbp", "ccbp", "shrink_and_perturb"]] = None
    task: str = "reach-v3"
    config_id: Optional[int] = None
    seed: int = 0
    total_steps: int = 5_000_000
    num_envs: int = 10
    async_envs: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_mode: str = "online"
    list_configs: bool = False
    get_count: bool = False
    get_total: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.get_total:
        get_total_sweep_size()
    elif args.list_configs:
        if args.algo is None:
            print("Error: --algo required for --list-configs")
            exit(1)
        list_configs(args.algo)
    elif args.get_count:
        if args.algo is None:
            print("Error: --algo required for --get-count")
            exit(1)
        get_count(args.algo)
    else:
        if args.algo is None:
            print("Error: --algo required")
            exit(1)
        if args.config_id is None:
            print("Error: --config-id required when running experiments")
            print("Use --list-configs to see available configurations")
            print("Use --get-count to see SLURM array command")
            exit(1)
        if args.task not in MT10_TASKS:
            print(f"Error: Unknown task '{args.task}'")
            print(f"Available tasks: {MT10_TASKS}")
            exit(1)
        run_config(
            algo=args.algo,
            config_id=args.config_id,
            task=args.task,
            seed=args.seed,
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            async_envs=args.async_envs,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
        )
