import jax
import time
from continual_learning_2.trainers.continual_supervised_learning import (
    LoggingConfig,
)
from continual_learning_2.configs import (
    ShrinkAndPerterbConfig,
    AdamConfig,
    MLPConfig,
)
from chex import dataclass
from typing import Literal
from pathlib import Path
import tyro
import jax.numpy as jnp
from continual_learning_2.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning_2.trainers.continual_rl import JittedContinualPPOTrainer
from continual_learning_2.types import (
    Activation,
    StdType,
)
from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs.optim import AdamConfig
from continual_learning_2.configs.training import RLTrainingConfig


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def shrink_and_perturb_ant_experiment():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    optim_conf = ShrinkAndPerterbConfig(
        tx=AdamConfig(learning_rate=1e-3),
        param_noise_fn=jax.nn.initializers.he_uniform(),
        seed=args.seed,
        shrink=0.8,
        perturb=0.01,
        every_n=1,
    )

    # Add validation to say what the available options are for dataset etc
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
            gae_lambda=0.9,
            entropy_coefficient=2e-2,
            clip_eps=0.2,
            vf_coefficient=0.5,
            normalize_advantages=True,
        ),
        env_cfg=EnvConfig("slippery_ant", num_envs=4096, num_tasks=20, episode_length=1000),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=50_000_000,
        ),
        logs_cfg=LoggingConfig(
            run_name=f"shrink_and_perturb_{args.seed}",
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            group="slippery_ant",
            save=False,  # Disable checkpoints cause it's so fast anyway
            wandb_mode=args.wandb_mode,
        ),
    )

    trainer.train()
    print(f"Training time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    shrink_and_perturb_ant_experiment()
