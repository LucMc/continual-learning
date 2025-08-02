import time
from typing import Literal

import jax
import jax.numpy as jnp
import tyro
from chex import dataclass

from continual_learning_2.configs import (
    AdamConfig,
    AdamwConfig,
    CBPConfig,
    CCBPConfig,
    LoggingConfig,
    RedoConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.configs.models import MLPConfig
# from continual_learning_2.configs.optim import AdamConfig, Adamw, RedoConfig
from continual_learning_2.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning_2.configs.training import RLTrainingConfig
from continual_learning_2.trainers.continual_rl import JittedContinualPPOTrainer
from continual_learning_2.types import (
    Activation,
    StdType,
)


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    # data_dir: Path = Path("./experiment_results")
    resume: bool = False

def run_all_mnist():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    optimizers = {
        "cbp": CBPConfig(
            tx=AdamConfig(learning_rate=1e-3),
            decay_rate=0.9,
            replacement_rate=0.5,
            maturity_threshold=20,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "ccbp": CCBPConfig(
            tx=AdamConfig(learning_rate=1e-3),
            seed=args.seed,
            decay_rate=0.9,
            replacement_rate=0.05,
            maturity_threshold=20,
        ),
        "redo": RedoConfig(
            tx=AdamConfig(learning_rate=1e-3),
            update_frequency=100,
            score_threshold=0.1,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=AdamConfig(learning_rate=1e-3),
            param_noise_fn=jax.nn.initializers.he_uniform(),
            seed=args.seed,
            shrink=0.8,
            perturb=0.01,
            every_n=1,
        ),
        "adam": AdamConfig(learning_rate=1e-3),
        "adamw": AdamwConfig(learning_rate=1e-3)
    }

    exp_start = time.time()
    for opt_name, opt_conf in optimizers.items():
        start = time.time()

        trainer = JittedContinualPPOTrainer(
            seed=args.seed,
            ppo_config=PPOConfig(
                policy_config=PolicyNetworkConfig(
                    optimizer=opt_conf,
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
                    optimizer=opt_conf,
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
                entropy_coefficient=1e-3, # 1e-2
                clip_eps=0.2, # 0.3
                vf_coefficient=0.5,
                normalize_advantages=True,
            ),
            env_cfg=EnvConfig("slippery_ant", num_envs=4096, num_tasks=3, episode_length=1000),
            train_cfg=RLTrainingConfig(
                resume=False,
                steps_per_task=50_000_000,
            ),
            logs_cfg=LoggingConfig(
                run_name=f"{opt_name}_{args.seed}",
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group="slippery_ant",
                save=False,  # Disable checkpoints cause it's so fast anyway
                wandb_mode=args.wandb_mode
            ),
        )
        trainer.train()
        print(f"Training time: {time.time() - start:.2f} seconds")
        del trainer

    print(f"Total training time: {time.time() - exp_start:.2f} seconds")

if __name__ == "__main__":
    run_all_mnist()
