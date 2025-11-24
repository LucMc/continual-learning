import time
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import tyro

from continual_learning.configs import (
    AdamConfig,
    CbpConfig,
    CcbpConfig,
    LoggingConfig,
    RedoConfig,
    RegramaConfig,
    ShrinkAndPerterbConfig,
)
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.continual_rl import JittedContinualPPOTrainer
from continual_learning.types import (
    Activation,
    StdType,
)


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = ""
    wandb_entity: str = ""
    # data_dir: Path = Path("./experiment_results")
    resume: bool = False
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)

    layer_norm: bool = False
    layer_norm_type: Literal["ln", "rmsnorm"] = "ln"


def run_all_slippery_ant():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    optimizers = {
        "ccbp": CcbpConfig(
            tx=AdamConfig(learning_rate=1e-3),
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.015,
            sharpness=16,
            threshold=1.0,
            update_frequency=1000,
            transform_type="sigmoid",
            metrics_type="cbp",
        ),
    }

    if args.include:
        optimizers = {
            name: config for name, config in optimizers.items() if name in args.include
        }

    for algorithm in args.exclude:
        optimizers.pop(algorithm)

    print(f"Running algorithms: {list(optimizers.keys())}")

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
                        layer_norm=args.layer_norm,
                        layer_norm_type=args.layer_norm_type,
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
                        layer_norm=args.layer_norm,
                        layer_norm_type=args.layer_norm_type,
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
            env_cfg=EnvConfig(
                "slippery_ant", num_envs=2048, num_tasks=20, episode_length=1000
            ),
            train_cfg=RLTrainingConfig(
                resume=False,
                steps_per_task=20_000_000,
            ),
            logs_cfg=LoggingConfig(
                run_name=f"{opt_name}_new_{args.seed}",
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group="slippery_ant_full2",
                save=False,  # Disable checkpoints cause it's so fast anyway
                wandb_mode=args.wandb_mode,
            ),
        )
        trainer.train()
        print(f"Training time: {time.time() - start:.2f} seconds")
        del trainer

    print(f"Total training time: {time.time() - exp_start:.2f} seconds")


if __name__ == "__main__":
    run_all_slippery_ant()

#     num_rollout_steps=2048 * 32 * 5,
#     num_epochs=4,
#     num_gradient_steps=32,
#     gamma=0.97,
#     gae_lambda=0.95,
#     entropy_coefficient=1e-2,
#     clip_eps=0.3,
#     vf_coefficient=0.5,
#     normalize_advantages=True,
# ),
# env_cfg=EnvConfig(
#     "slippery_ant", num_envs=4096, num_tasks=20, episode_length=1000
# ),
