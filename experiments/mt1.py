import multiprocessing
# If you get subproccess warnings uncomment as workers don't need gpu
# if multiprocessing.current_process().name != "MainProcess":
#     import os
#     os.environ["JAX_PLATFORMS"] = "cpu"

import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
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
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.envs.metaworld import MetaWorldSingleTaskEnv
from continual_learning.trainers.sac_trainer import SAC
from continual_learning.types import Activation, StdType
from continual_learning.utils.monitoring import Logger, prefix_dict
from continual_learning.utils.replay_buffer import ReplayBuffer


# MT10 task names
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

OPTIMIZERS = ["adam", "cbp", "cpr", "redo", "regrama", "shrink_and_perturb"]


@dataclass(frozen=True)
class Args:
    """Command line arguments for MetaWorld MT1 experiment."""

    task_name: str
    optimizer: Literal["adam", "cbp", "cpr", "redo", "regrama", "shrink_and_perturb"]
    seed: int = 0
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "MT1 results"
    wandb_entity: str = ""

    # Training settings (match MT10 SAC)
    replay_ratio: int = 4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    total_steps: int = 5_000_000

    # Environment settings
    num_envs: int = 10

    # Network architecture (match MT10 SAC)
    hidden_size: int = 256
    num_layers: int = 3


def get_optimizer_config(name: str, seed: int, lr: float = 3e-4):
    """Get optimizer config by name.

    Best parameters from MT1 sweep (window-close-v3):
    - CCBP: 4414.9 return (best single run)
    - S&P: 4377.8 return
    - ReGraMa: 3466.9 return
    - ReDo: Uses same params as ReGraMa
    """
    optimizers = {
        "adam": AdamConfig(learning_rate=lr),
        # ReGraMa: Best single run 3466.9
        "regrama": RegramaConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=100000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        # ReDo: Using same params as ReGraMa (ReDo underperformed in sweep)
        "redo": RedoConfig(
            tx=AdamConfig(learning_rate=lr),
            update_frequency=100000,
            score_threshold=0.0001,
            max_reset_frac=0.02,
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "cbp": CbpConfig(
            tx=AdamConfig(learning_rate=lr),
            replacement_rate=1e-5,
            decay_rate=0.999,
            maturity_threshold=1000,
            seed=seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        # CCBP: Best single run 4414.9
        "cpr": CprConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=seed,
            decay_rate=0.99,
            replacement_rate=0.15,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        # S&P: Best single run 4377.8
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=AdamConfig(learning_rate=lr),
            seed=seed,
            shrink=0.9999,
            perturb=0.001,
            every_n=1000,
            param_noise_fn=jax.nn.initializers.lecun_normal(),
        ),
    }
    return optimizers[name]


def make_sac_config(args: Args, opt_cfg) -> SACConfig:
    """Create SACConfig from args and optimizer config."""
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
        replay_ratio=args.replay_ratio,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
    )


def run_metaworld_mt1():
    """Run SAC on a single MetaWorld task."""
    args = tyro.cli(Args)

    # Validate task name
    if args.task_name not in MT10_TASKS:
        raise ValueError(
            f"Unknown task: {args.task_name}. Must be one of: {MT10_TASKS}"
        )

    # Validate wandb settings
    if args.wandb_mode != "disabled":
        if not args.wandb_entity:
            raise ValueError("wandb_entity required when wandb is enabled")

    print(f"{'='*60}")
    print(f"MetaWorld MT1 Single-Task Experiment")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Seed: {args.seed}")
    print(f"Total steps: {args.total_steps}")
    print(f"Num envs: {args.num_envs}")
    print(f"Replay ratio: {args.replay_ratio}")
    print(f"{'='*60}")

    # Create environment
    print("Initializing environment...")
    env = MetaWorldSingleTaskEnv(
        task_name=args.task_name,
        num_envs=args.num_envs,
        seed=args.seed,
    )
    print(f"  Obs dim: {env.obs_dim}, Action dim: {env.action_dim}")

    # Get optimizer config
    opt_cfg = get_optimizer_config(args.optimizer, args.seed)

    # Create SAC config
    sac_config = make_sac_config(args, opt_cfg)

    # Initialize logger
    logger = Logger(
        LoggingConfig(
            run_name=f"sac_{args.task_name}_{args.optimizer}_{args.seed}",
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            group=args.task_name,
            save=False,
            wandb_mode=args.wandb_mode,
        ),
        run_config={
            "algorithm": "sac",
            "task_name": args.task_name,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "total_steps": args.total_steps,
            "num_envs": args.num_envs,
            "replay_ratio": args.replay_ratio,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        },
    )

    # Initialize SAC
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    sac_state = SAC.init_state(
        key=init_key,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        cfg=sac_config,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
    )
    buffer_state = replay_buffer.init()

    # Target entropy
    target_entropy = -float(env.action_dim)

    # Training state
    total_steps = 0
    total_episodes = 0
    total_gradient_steps = 0

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    current_episode_reward = np.zeros(args.num_envs)
    current_episode_length = np.zeros(args.num_envs, dtype=int)
    episode_successes: list[bool] = []

    # Initialize environment
    obs = env.init()
    start_time = time.time()
    last_log_step = 0
    last_eval_step = 0

    print("Starting training...")

    while total_steps < args.total_steps:
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
                episode_lengths.append(int(current_episode_length[i]))
                success_list = timestep.info.get("success", [False] * args.num_envs)
                episode_successes.append(success_list[i] if i < len(success_list) else False)
                current_episode_reward[i] = 0
                current_episode_length[i] = 0
                total_episodes += 1

        obs = timestep.next_observation
        total_steps += args.num_envs  # Count all env steps

        # Update SAC
        all_logs = []
        if total_steps >= args.learning_starts:
            for _ in range(args.replay_ratio):
                key, sample_key = jax.random.split(key)
                batch = ReplayBuffer.sample(buffer_state, sample_key, args.batch_size)
                sac_state, logs = SAC.update(sac_state, batch, sac_config, target_entropy)
                all_logs.append(logs)
                total_gradient_steps += 1

        # Log metrics
        if total_steps - last_log_step >= 1000:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1e-6)

            log_dict = {
                "charts/total_steps": total_steps,
                "charts/total_episodes": total_episodes,
                "charts/total_gradient_steps": total_gradient_steps,
                "charts/SPS": sps,
                "charts/replay_ratio": args.replay_ratio,
                "charts/buffer_size": int(buffer_state.size),
            }

            if episode_rewards:
                log_dict["charts/mean_episode_return"] = float(
                    np.mean(episode_rewards[-100:])
                )
                log_dict["charts/mean_episode_length"] = float(
                    np.mean(episode_lengths[-100:])
                )

            if episode_successes:
                log_dict["charts/success_rate"] = float(
                    np.mean(episode_successes[-100:])
                )

            if all_logs:
                avg_logs = {}
                for log_key in all_logs[0].keys():
                    values = [float(l[log_key]) for l in all_logs if log_key in l]
                    if values:
                        avg_logs[log_key] = float(np.mean(values))
                log_dict.update(prefix_dict("train", avg_logs))

            logger.log(log_dict, step=total_steps)
            last_log_step = total_steps

            mean_return = log_dict.get("charts/mean_episode_return", 0)
            success_rate = log_dict.get("charts/success_rate", 0)
            print(
                f"Step {total_steps:>7}, Eps: {total_episodes:>4}, "
                f"Return: {mean_return:>7.2f}, Success: {success_rate:>5.2%}, "
                f"SPS: {sps:>6.1f}",
                flush=True,
            )

        # Periodic evaluation
        if total_steps - last_eval_step >= 10_000:
            eval_metrics = evaluate(env, sac_state, key, num_episodes=20)
            eval_log = {
                "eval/mean_return": eval_metrics["mean_return"],
                "eval/success_rate": eval_metrics["success_rate"],
            }
            logger.log(eval_log, step=total_steps)
            last_eval_step = total_steps
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
    logger.log(final_log, step=total_steps)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total gradient steps: {total_gradient_steps}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final success rate: {final_metrics['success_rate']:.2%}")
    print(f"Final mean return: {final_metrics['mean_return']:.2f}")
    print(f"{'='*60}")

    logger.close()
    env.close()


def evaluate(env, sac_state, key, num_episodes: int = 10):
    """Evaluate the policy deterministically."""
    obs = env.init()
    episode_returns = []
    episode_successes = []

    current_return = np.zeros(env.num_envs)
    current_length = np.zeros(env.num_envs, dtype=int)

    while len(episode_returns) < num_episodes:
        # Deterministic action selection
        dist = sac_state.actor.apply_fn(sac_state.actor.params, obs)
        try:
            action = dist.mode()
        except NotImplementedError:
            base_mean = dist.distribution.loc
            action = jnp.tanh(base_mean)
        action = jnp.clip(action, -1.0, 1.0)

        timestep = env.step(action)

        current_return += np.asarray(timestep.reward).squeeze(-1)
        current_length += 1

        dones = np.asarray(timestep.terminated | timestep.truncated).squeeze(-1)
        for i, done in enumerate(dones):
            if done:
                episode_returns.append(float(current_return[i]))
                success_list = timestep.info.get("success", [False] * env.num_envs)
                episode_successes.append(success_list[i] if i < len(success_list) else False)
                current_return[i] = 0
                current_length[i] = 0

        obs = timestep.next_observation

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
    }


if __name__ == "__main__":
    run_metaworld_mt1()

