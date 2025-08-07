import time
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.experimental
import jax.flatten_util
import jax.numpy as jnp
import tyro
from chex import dataclass
from flax.core.scope import DenyList
from jaxtyping import PRNGKeyArray

from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs.optim import AdamConfig, MuonConfig, CcbpConfig 
from continual_learning_2.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning_2.configs.training import RLTrainingConfig
from continual_learning_2.envs import JittableContinualLearningEnv, get_benchmark
from continual_learning_2.envs.base import JittableVectorEnv
from continual_learning_2.models import get_model, get_model_cls
from continual_learning_2.models.rl import Policy
from continual_learning_2.optim import get_optimizer
from continual_learning_2.trainers.continual_rl import JittedContinualPPOTrainer
from continual_learning_2.types import (
    Activation,
    EnvState,
    LogDict,
    Observation,
    Rollout,
    StdType,
)
from continual_learning_2.utils.buffers import compute_gae_scan
from continual_learning_2.utils.monitoring import (
    Logger,
    accumulate_concatenated_metrics,
    explained_variance,
    get_logs,
    prefix_dict,
    pytree_histogram,
)
from continual_learning_2.utils.training import TrainState


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def ccbp_ant_experiment() -> None:
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    optim_conf = CcbpConfig(
        tx=MuonConfig(learning_rate=3e-4),
        decay_rate=0.95,
        replacement_rate=0.001,
        maturity_threshold=100,
    )

    start = time.time()
    trainer = JittedContinualPPOTrainer(
        seed=args.seed,
        ppo_config=PPOConfig(
            policy_config=PolicyNetworkConfig(
                optimizer=optim_conf,
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
                optimizer=optim_conf,
                network=MLPConfig(
                    num_layers=5,
                    hidden_size=256,
                    output_size=1,
                    activation_fn=Activation.Swish,
                    kernel_init=jax.nn.initializers.lecun_normal(),
                    dtype=jnp.float32,
                ),
            ),
            num_rollout_steps=2048 * 32 * 5,
            num_epochs=4,
            num_gradient_steps=32,
            gamma=0.97,
            gae_lambda=0.95,
            entropy_coefficient=1e-2,
            clip_eps=0.3,
            vf_coefficient=0.5,
            normalize_advantages=True,
        ),
        env_cfg=EnvConfig("slippery_ant", num_envs=4096, num_tasks=20, episode_length=1000),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=50_000_000,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"ccbp_{args.seed}",
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            group="slippery_ant",
            save=False,  # Disable checkpoints cause it's so fast anyway
            wandb_mode=args.wandb_mode
        ),
    )

    trainer.train()

    print(f"Training time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    ccbp_ant_experiment()
