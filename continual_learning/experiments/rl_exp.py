from dataclasses import dataclass
from pathlib import Path

import tyro

from continual_learning.config.networks import ContinuousActionPolicyConfig, ValueFunctionConfig
from continual_learning.config.nn import MultiHeadConfig, VanillaNetworkConfig
from continual_learning.config.optim import OptimizerConfig
from continual_learning.config.rl import OnPolicyTrainingConfig
# from continual_learning.envs import MetaworldConfig
from continual_learning.experiment import Experiment
from continual_learning.rl.algorithms import PPOConfig

import gymnasium as gym

@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    experiment = Experiment(
        exp_name="ppo_continual_learning",
        seed=args.seed,
        data_dir=args.data_dir,
        env=gym.make_vec("LunarLanderContinuous-v3", num_envs=10, vectorization_mode="async"),
        algorithm=PPOConfig(
            # num_tasks=10,
            gamma=0.99,
            policy_config=ContinuousActionPolicyConfig( # TODO: Replace these configs with directly making the thing
                # network_config=VanillaNetworkConfig(
                #     optimizer=OptimizerConfig(max_grad_norm=1.0)
                # )
            ),
            vf_config=ValueFunctionConfig(
                # network_config=VanillaNetworkConfig(
                #     optimizer=OptimizerConfig(max_grad_norm=1.0),
                # )
            ),
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=int(2e7),
            rollout_steps=10_000,
            num_epochs=16,
            num_gradient_steps=32,
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        experiment.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=experiment,
            resume="allow",
        )

    experiment.run()


if __name__ == "__main__":
    main()
