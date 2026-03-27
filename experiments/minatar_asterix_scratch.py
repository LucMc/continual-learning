"""Sanity check: train Discrete SAC on Asterix from scratch.

Uses identical hyperparameters to the continual learning run
(discrete_sac_minatar_adam_cnn_42) to verify the algorithm can
learn Asterix when starting from a fresh initialisation.
"""

import continual_learning.envs.minatar as minatar_module

# Patch task list BEFORE any benchmark is created
minatar_module.TASK_SPECS = [("asterix", 4)]

import jax
import jax.numpy as jnp

from continual_learning.configs import AdamConfig, LoggingConfig
from continual_learning.configs.envs import EnvConfig
from continual_learning.configs.models import CNNConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.configs.training import RLTrainingConfig
from continual_learning.trainers.discrete_sac_trainer import DiscreteSACTrainer
from continual_learning.types import Activation

lr = 3e-4
seed = 42

actor_network = CNNConfig(
    output_size=1,
    features=(16,),
    num_convs_per_layer=1,
    kernel_size=(3, 3),
    strides=1,
    padding="SAME",
    use_max_pooling=False,
    dense_hidden_size=128,
    num_dense_layers=1,
    activation_fn=Activation.ReLU,
    kernel_init=jax.nn.initializers.he_uniform(),
    bias_init=jax.nn.initializers.zeros,
    dtype=jnp.float32,
)
critic_network = CNNConfig(
    output_size=1,
    features=(16,),
    num_convs_per_layer=1,
    kernel_size=(3, 3),
    strides=1,
    padding="SAME",
    use_max_pooling=False,
    dense_hidden_size=128,
    num_dense_layers=1,
    activation_fn=Activation.ReLU,
    kernel_init=jax.nn.initializers.he_uniform(),
    bias_init=jax.nn.initializers.zeros,
    dtype=jnp.float32,
)

sac_config = SACConfig(
    actor_config=PolicyNetworkConfig(
        optimizer=AdamConfig(learning_rate=lr),
        network=actor_network,
    ),
    critic_config=QNetworkConfig(
        optimizer=AdamConfig(learning_rate=lr),
        network=critic_network,
    ),
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    auto_entropy=True,
    replay_ratio=4,
    buffer_size=200_000,
    batch_size=256,
    learning_starts=5_000,
)

trainer = DiscreteSACTrainer(
    seed=seed,
    sac_config=sac_config,
    env_cfg=EnvConfig(
        name="minatar",
        num_envs=4,
        num_tasks=1,
        episode_length=1000,
    ),
    train_cfg=RLTrainingConfig(
        resume=False,
        steps_per_task=500_000,
    ),
    logs_cfg=LoggingConfig(
        run_name="asterix_from_scratch_sanity_check",
        wandb_entity="lucmc",
        wandb_project="cont-minatar",
        group="sanity_check",
        save=False,
        wandb_mode="online",
    ),
)

trainer.train()
