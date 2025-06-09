"""
LayerNorm reduces grads, so increase lr, delay increases grads so decrease lr
would be nice to adjust lr automatically to get the same grad curve under each condition...

TODO:
 - Remember the total delay subtly influences things as it affects the total action buf size
 - Change from calculating mini-batches based on batch_size to using n_mini_batches a
 param directly (less to change when chaning n_envs/ more intuitive)
 - Test w/ multiple envs?
 - Test w/ changing constant delays too
 - Add logs for env delay/ friction - i.e. in info and add info
"""

import enum
from typing import Tuple, Literal, override
from chex import dataclass
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import optax
import distrax
from pprint import pprint
from functools import partial
import wandb
import tyro
import os
from pathlib import Path
import continual_learning.envs.slippery_ant_v5
from continual_learning.nn import (
    ActorNet,
    ValueNet,
    ActorNetLayerNorm,
    ValueNetLayerNorm,
)
from continual_learning.optim.base import ResettingTrainState
from continual_learning.optim.continual_backprop import continual_backprop
from continual_learning.optim.continuous_continual_backprop import continuous_continual_backprop
from continual_learning.optim.ccbp_2 import continuous_continual_backprop2
from continual_learning.optim.redo import redo


from continual_learning.utils.metrics import compute_plasticity_metrics
from continual_learning.utils.wrappers_rd import (
    # ContinualRandomIntervalDelayWrapper,
    ContinualIntervalDelayWrapper,
)
from continual_learning.rl.ppo import PPO, Config
import gymnasium_robotics

# import time
# from memory_profiler import profile
from jaxtyping import jaxtyped, TypeCheckError
from beartype import beartype as typechecker


@dataclass(frozen=True)
class ContConfig(Config):
    """Only need to specify new params, over overwrite existing defaults in ppo.py"""

    # Changed defaults
    # training_steps: int = 2_000_000  # total training time-steps
    # n_envs: int = 1  # number of parralel training envs
    # rollout_steps: int = 64 * 20  # env steps per rollout
    # env_id: str = "ContinualAnt-v0"
    # batch_size: int = 64  # minibatch size
    # clip_range: float = 0.2  # policy clip range
    # epochs: int = 10  # number of epochs for fitting mini-batches
    # max_grad_norm: float = 0.5  # maximum gradient norm
    # gamma: float = 0.99  # discount factor
    # vf_clip_range: float = np.inf  # vf clipping (typically higher than clip_range)
    # ent_coef: float = 0.0  # how much exploration?
    # gae_lambda: float = 0.95  # bias-variance tradeoff in gae
    # learning_rate: float = 3e-4  # lr for both actor and critic
    # vf_coef: float = 0.5  # balance vf loss magnitude

    # New params
    layer_norm: bool = False  # Weather or not to use LayerNorm layers after activations
    dormant_reset_method: Literal["cbp", "ccbp", "ccbp2", "redo", "none"] = (
        "none"  # Dormant neuron reactivation method
    )
    optim: Literal["adam", "adamw", "sgd", "muon", "muonw"] = "muonw"
    run_name: str = ""  # Postfix name for training run
    delay_type: Literal["none", "random", "random_incremental", "constant", "incremental"] = (
        "none"
    )
    changes: int = 10  # How many env changes for continual learning (if using ContinualAnt-v0 or delay=True)


@dataclass(frozen=True)
class ContPPO(PPO, ContConfig):
    buffer_size: int = 2048

    @partial(jax.jit, static_argnames=["self"])
    def update(
        self,
        actor_ts: TrainState,
        value_ts: TrainState,
        obss: Float[Array, "#n_minibatches #batch_size #obs_dim"],
        actions: Float[Array, "#n_minibatches #batch_size #action_dim"],
        old_values: Float[Array, "#n_minibatches #batch_size 1"],
        old_log_probs: Float[Array, "#n_minibatches #batch_size"],
        advantages: Float[Array, "#n_minibatches #batch_size"],
        returns: Float[Array, "#n_minibatches #batch_size"],
    ):
        # Shuffle idxs
        n_minibatches = obss.shape[0]

        actor_loss_total = 0
        value_loss_total = 0
        value_total = 0
        lp_total = 0
        kl_total = 0
        clip_fraction_total = 0

        for i in range(n_minibatches):
            # advantage normalisation
            adv_norm = (advantages[i] - advantages[i].mean()) / (advantages[i].std() + 1e-8)

            (
                (actor_loss_v, (lp_mean, approx_kl_mean, clip_fraction_mean, actor_features)),
                actor_grads,
            ) = jax.value_and_grad(self.actor_loss, has_aux=True)(
                actor_ts.params,
                actor_ts.apply_fn,
                obss[i],
                actions[i],
                old_log_probs[i],
                adv_norm,
            )
            # Apply updates with / without features
            if self.dormant_reset_method != "none":
                actor_ts = actor_ts.apply_gradients(grads=actor_grads, features=actor_features)
            else:
                actor_ts = actor_ts.apply_gradients(grads=actor_grads)

            actor_loss_total += actor_loss_v

            (value_loss_v, (value_mean, value_features)), value_grads = jax.value_and_grad(
                self.value_loss, has_aux=True
            )(value_ts.params, value_ts.apply_fn, obss[i], returns[i], old_values[i])

            if self.dormant_reset_method != "none":
                value_ts = value_ts.apply_gradients(grads=value_grads, features=value_features)
            else:
                value_ts = value_ts.apply_gradients(grads=value_grads)

            value_loss_total += value_loss_v
            value_total += value_mean
            lp_total += lp_mean
            kl_total += approx_kl_mean
            clip_fraction_total += clip_fraction_mean

        return (
            actor_ts,
            value_ts,
            {
                "value_loss_final": value_loss_v,
                "actor_loss_final": actor_loss_v,
                "value_loss_total": value_loss_total,
                "actor_loss_total": actor_loss_total,
                "value_pred_mean": (value_total / n_minibatches),
                "actor_log_probs_mean": (lp_total / n_minibatches),
                "approx_kl": (kl_total / n_minibatches),
                "clip_fraction": (clip_fraction_total / n_minibatches),
                "value_g_mag": jax.tree.reduce(
                    lambda acc, g: acc + jnp.sum(jnp.abs(g)),
                    value_grads,
                ),
                "actor_g_mag": jax.tree.reduce(
                    lambda acc, g: acc + jnp.sum(jnp.abs(g)),
                    actor_grads,
                ),
            },
        )

    def make_env(self: PPO, idx: int, video_folder: str = None, env_args: dict = {}):
        def thunk():
            if self.delay_type != "none":
                print(":: Added continual time delays ::")
                change_every = env_args.pop("change_every")
                env = gym.make(self.env_id, **env_args)
                # env = ContinualRandomIntervalDelayWrapper(
                env = ContinualIntervalDelayWrapper(
                    env,
                    change_every=change_every,
                    obs_delay_range=range(0, 4),
                    act_delay_range=range(0, 4),
                    delay_type=self.delay_type,
                )

            else:
                env = gym.make(self.env_id, **env_args)

            # env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            # env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.NormalizeReward(env, gamma=0.99) # TODO: replace with actual gamma
            if isinstance(env.observation_space, gym.spaces.Dict):
                print(f":: Original observation space: {env.observation_space}")
                env = gym.wrappers.FlattenObservation(env)
                print(f":: Flattened observation space: {env.observation_space}")

            if self.log_video_every > 0 and idx == 0:
                print(":: Recording Videos ::")
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder,
                    lambda t: t % self.log_video_every == 0,
                )
            return env

        return thunk

    @override
    @staticmethod
    def learn(config: ContConfig):
        ppo_agent = ContPPO(
            buffer_size=config.rollout_steps,
            **config.__dict__,
        )
        cbp_params = {} # Change cbp options here i.e. "maturity_threshold": jnp.inf

        np.random.seed(ppo_agent.seed)  # Seeding for np operations
        pprint(ppo_agent.__dict__)
        env_args = {}

        if ppo_agent.log_video_every > 0:
            base_video_dir = Path("videos")
            video_folder = base_video_dir / str(
                len(os.listdir(base_video_dir))
            )  # run_id for local videos
            os.makedirs(video_folder)
            env_args.update({"render_mode": "rgb_array"})
        else:
            video_folder = None

        if ppo_agent.log:
            tags = [
                "PPO",
                ppo_agent.env_id,
                ppo_agent.optim,
                ppo_agent.delay_type,
                ppo_agent.dormant_reset_method,
            ]
            # NOTE: If using layernorm, increase learning rate to 0.0005
            # fmt: off
            if ppo_agent.layer_norm: tags.append("LayerNorm")
            # fmt: on

            wandb.init(
                project="jax-ppo",
                name=f"{ppo_agent.run_name} {ppo_agent}",
                config=config.__dict__,  # Get from tyro etc
                tags=tags,
                # monitor_gym=True,
                save_code=True,
            )

        # Specific to this setup, should probably add a config file for env_args?
        if (
            ppo_agent.env_id == "ContinualAnt-v0" or ppo_agent.delay_type != "none"
        ):  # Add change every as param?
            env_args.update(
                {"change_every": ppo_agent.training_steps // ppo_agent.changes}
            )  # should be 10

        ckpt_path = "./checkpoints"
        assert not ppo_agent.rollout_steps % ppo_agent.batch_size, (  # TODO: Make adaptive
            "Must have rollout steps divisible into batches"
        )

        envs = gym.vector.SyncVectorEnv(
            [
                ppo_agent.make_env(i, video_folder=video_folder, env_args=env_args)
                for i in range(ppo_agent.n_envs)
            ]
        )
        dummy_obs, _ = envs.reset(seed=ppo_agent.seed)
        key = random.PRNGKey(ppo_agent.seed)
        current_global_step = 0

        actor_key, value_key, key = random.split(key, num=3)

        if ppo_agent.layer_norm:
            print(":: Using LayerNorm layers ::")
            actor_net_cls = ActorNetLayerNorm
            value_net_cls = ValueNetLayerNorm
        else:
            print(":: Using standard architecture ::")
            actor_net_cls = ActorNet
            value_net_cls = ValueNet

        # Select optimiser
        # fmt: off
        if ppo_agent.optim == "adam": tx = optax.adam
        if ppo_agent.optim == "adamw": tx = optax.adamw
        if ppo_agent.optim == "sgd": tx = optax.sgd
        if ppo_agent.optim == "muon": tx = optax.contrib.muon
        if ppo_agent.optim == "muonw": tx = partial(optax.contrib.muon, weight_decay=0.01)
        # For some reason loads of decay seems to work better...

        # Continual backpropergation
        if ppo_agent.dormant_reset_method != "none":
            cbp_value_key, cbp_actor_key, key = random.split(key, num=3)

            match ppo_agent.dormant_reset_method:
                case "cbp": reset_method = continual_backprop
                case "ccbp": reset_method = continuous_continual_backprop
                case "ccbp2": reset_method = continuous_continual_backprop2
                case "redo": reset_method = redo

            act_ts_kwargs = dict(rng=cbp_actor_key, reset_method=reset_method) | cbp_params
            val_ts_kwargs = dict(rng=cbp_value_key, reset_method=reset_method) | cbp_params

            trainstate_cls = ResettingTrainState
        else:
            trainstate_cls = TrainState
            act_ts_kwargs = {}
            val_ts_kwargs = {}

        # fmt: on
        last_obs, first_info = envs.reset()
        last_episode_starts = np.ones((ppo_agent.n_envs,), dtype=bool)

        # Create trainstates
        actor_ts, value_ts = ppo_agent.setup_network_trainstates(
            last_obs,
            envs.single_action_space.shape[0],
            actor_key,
            value_key,
            actor_net_cls=actor_net_cls,
            value_net_cls=value_net_cls,
            trainstate_cls=trainstate_cls,
            act_ts_kwargs=act_ts_kwargs,
            val_ts_kwargs=val_ts_kwargs,
        )

        while current_global_step < ppo_agent.training_steps:
            rollout, rollout_info, env_infos = ppo_agent.get_rollout(
                actor_ts,
                value_ts,
                envs,
                last_obs,
                last_episode_starts,
                key,
            )

            current_global_step += ppo_agent.rollout_steps * ppo_agent.n_envs

            actor_ts, value_ts, key, training_info = ppo_agent.outer_loop(
                key, actor_ts, value_ts, rollout
            )
            if ppo_agent.delay_type != "none":
                env_infos = {"mean_delay_mag": np.mean([x["delay_mag"] for x in env_infos])}
            else:
                env_infos = {}

            full_logs = training_info | rollout_info | env_infos
            pprint(full_logs)

            if ppo_agent.log:
                wandb.log(full_logs, step=current_global_step)

                if current_global_step // ppo_agent.rollout_steps * ppo_agent.n_envs % 10 == 0:  # fmt: skip
                    print(f":: Checkpointing to --> {ckpt_path} :: ")
                    try:
                        wandb.save(ckpt_path)
                    except:
                        print("Checkpoint failed")
                        breakpoint()

        # Upload videos and close
        if ppo_agent.log:
            if ppo_agent.log_video_every > 0:
                print("[ ] Uploading Videos ...", end="\r")
                for video_name in os.listdir(video_folder):
                    print("Check line bellow")
                    wandb.log({video_name: wandb.Video(str(video_folder / video_name))})
                print(r"[x] Uploading Videos ...")

            wandb.finish()
        envs.close()


if __name__ == "__main__":
    config = tyro.cli(ContConfig)
    ContPPO.learn(config)

# _reward = np.where(truncated, self.gamma * value_ts.apply_fn(value_ts.params, jnp.array(_obs)).item(), reward) # should be added to r anyway
#
# @partial(jax.jit, static_argnames="self")
# def compute_returns_and_advantage(  # TODO: Replace loop with scan keeping advs ass carry instead of at/set
#     self, rewards, values, episode_starts, last_value: Array, done: np.ndarray
# ) -> None:
#     buffer_size = values.shape[0]
#     advantages = jnp.ones(buffer_size)
#
#     last_gae_lam = 0
#     for step in reversed(range(buffer_size)):
#         if step == buffer_size - 1:
#             next_non_terminal = 1.0 - done.astype(np.float32)
#             next_values = last_value
#         else:
#             next_non_terminal = 1.0 - episode_starts[step + 1]
#             next_values = values[step + 1]
#         # next values shape (1024, 4, 4 1) check sbx and logic
#         delta = (
#             rewards[step]
#             + self.gamma * next_values * next_non_terminal
#             - values[step]
#         )
#         last_gae_lam = (
#             delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#         )
#         advantages = advantages.at[step].set(last_gae_lam[0])
#
#     returns = advantages + values
#     return returns, advantages
#
