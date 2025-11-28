import os

os.environ["XLA_FLAGS"] = "--xla_gpu_graph_min_graph_size=1 --xla_gpu_triton_gemm_any=True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

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


def run_all_humanoid_stand():
    args = tyro.cli(Args)

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    # base_optim = AdamConfig(learning_rate=1e-3)
    base_optim = AdamConfig(learning_rate=3e-4)
    # base_optim = MuonConfig(learning_rate=1e-3)

    optimizers = {
        "standard": base_optim,
        "regrama": RegramaConfig(
            tx=base_optim,
            update_frequency=100,
            score_threshold=0.25,
            max_reset_frac=None,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "ccbp": CcbpConfig(
            tx=base_optim,
            seed=args.seed,
            replacement_rate=0.015,
            decay_rate=0.99,
            sharpness=16,
            threshold=0.95,
            update_frequency=1000,
            transform_type="sigmoid",
        ),
        "redo": RedoConfig(
            tx=base_optim,
            update_frequency=100,
            score_threshold=0.5,
            max_reset_frac=None,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "cbp": CbpConfig(
            tx=base_optim,
            decay_rate=0.99,
            replacement_rate=0.0025,
            maturity_threshold=100,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.lecun_normal(),
        ),
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            param_noise_fn=jax.nn.initializers.lecun_normal(),
            tx=base_optim,
            seed=args.seed,
            shrink=1 - 0.001,
            perturb=0.005,
            every_n=1000,
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
                        # output_size=21,
                        output_size=17,
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
                num_rollout_steps=2048 * 32,
                num_epochs=4,
                num_gradient_steps=32,
                gamma=0.97,
                gae_lambda=0.95,
                entropy_coefficient=1e-3,
                clip_eps=0.2,
                vf_coefficient=0.5,
                normalize_advantages=True,
                normalize_observations=True,
            ),
            env_cfg=EnvConfig(
                "humanoid_stand", num_envs=1024, num_tasks=1, episode_length=1000
            ),
            train_cfg=RLTrainingConfig(
                resume=False,
                steps_per_task=20_000_000,
            ),
            logs_cfg=LoggingConfig(
                run_name=f"{opt_name}_humanoid_stand_{args.seed}",
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group=f"{opt_name}_humanoid_stand",
                save=False,  # Disable checkpoints cause it's so fast anyway
                wandb_mode=args.wandb_mode,
            ),
        )
        trainer.train()
        print(f"Training time: {time.time() - start:.2f} seconds")
        del trainer

    print(f"Total training time: {time.time() - exp_start:.2f} seconds")


if __name__ == "__main__":
    run_all_humanoid_stand()

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
