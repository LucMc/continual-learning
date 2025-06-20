import enum
import brax.envs
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
from continual_learning.optim.continual_backprop import CBPTrainState
from continual_learning.optim.continuous_continual_backprop import CCBPTrainState
from continual_learning.optim.ccbp_2 import CCBP2TrainState
from continual_learning.utils.miscellaneous import compute_plasticity_metrics
from continual_learning.utils.wrappers_rd import (
    # ContinualRandomIntervalDelayWrapper,
    ContinualIntervalDelayWrapper,
)
from continual_learning.rl.cont_ppo import ContConfig
from continual_learning.rl.ppo_brax import BraxPPO, BraxConfig
import gymnasium_robotics

# import time
# from memory_profiler import profile
from jaxtyping import jaxtyped, TypeCheckError
from beartype import beartype as typechecker


@dataclass(frozen=True)
class BraxContConfig(ContConfig, BraxConfig):  # Inherit for defaults
    """Only need to specify new params, or overwrite existing defaults in ppo.py"""

    env_id: str = "ant"  # BRAX env name
    training_steps: int = 500_000 * 256  # total training time-steps
    n_envs: int = 32  # number of parralel training envs
    rollout_steps: int = 1024 * 16  # env steps per rollout
    batch_size: int = 64 * 4  # minibatch size


@dataclass(frozen=True)
class BraxContPPO(BraxPPO, BraxContConfig):
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

    def make_env(self, video_folder: str = None, env_args: dict = {}):
        # TODO: Add wrappers
        return brax.envs.wrappers.training.wrap(brax.envs.get_environment(self.env_id))

    @override
    @staticmethod
    def learn(config: BraxContConfig):
        ppo_agent = BraxContPPO(buffer_size=config.rollout_steps, **config.__dict__)
        cbp_params = {}  # Change cbp options here i.e. "maturity_threshold": jnp.inf
        env_args = {}

        np.random.seed(ppo_agent.seed)  # Seeding for np operations
        pprint(ppo_agent.__dict__)

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
        if ppo_agent.env_id == "ContinualAnt-v0" or ppo_agent.delay_type != "none":
            env_args.update({"change_every": ppo_agent.training_steps // ppo_agent.changes})

        ckpt_path = "./checkpoints"
        assert not ppo_agent.rollout_steps % ppo_agent.batch_size, (
            "rollout steps indivisible into batches"
        )

        key = random.PRNGKey(ppo_agent.seed)
        env_keys, actor_key, value_key, key = random.split(key, num=4)
        initial_reset_keys = random.split(env_keys, num=ppo_agent.n_envs)
        env = ppo_agent.make_env(video_folder=video_folder, env_args=env_args)

        states = env.reset(initial_reset_keys)
        current_global_step = 0

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
        elif ppo_agent.optim == "adamw": tx = optax.adamw
        elif ppo_agent.optim == "sgd": tx = optax.sgd
        elif ppo_agent.optim == "muon": tx = optax.contrib.muon
        elif ppo_agent.optim == "muonw": tx = partial(optax.contrib.muon, weight_decay=0.01)
        else: raise "Unsupported optimiser"

        # Continual backpropergation
        if ppo_agent.dormant_reset_method != "none":
            cbp_value_key, cbp_actor_key, key = random.split(key, num=3)

            match ppo_agent.dormant_reset_method:
                case "cbp": trainstate_cls = CBPTrainState
                case "ccbp": trainstate_cls = CCBPTrainState
                case "ccbp2": trainstate_cls = CCBP2TrainState

            act_ts_kwargs = dict(rng=cbp_actor_key) | cbp_params
            val_ts_kwargs = dict(rng=cbp_value_key) | cbp_params
        else:
            trainstate_cls = TrainState
            act_ts_kwargs = {}
            val_ts_kwargs = {}

        # fmt: on
        last_episode_starts = np.ones((ppo_agent.n_envs,), dtype=bool)

        # Create trainstates
        actor_ts, value_ts = ppo_agent.setup_network_trainstates(
            states.obs,
            env.action_size,
            actor_key,
            value_key,
            actor_net_cls=actor_net_cls,
            value_net_cls=value_net_cls,
            trainstate_cls=trainstate_cls,
            act_ts_kwargs=act_ts_kwargs,
            val_ts_kwargs=val_ts_kwargs,
        )

        while current_global_step < ppo_agent.training_steps:
            states, rollout, rollout_info, env_infos = ppo_agent.get_rollout(
                actor_ts, value_ts, env, states, key
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

        ppo_agent.cleanup()


if __name__ == "__main__":
    config = tyro.cli(BraxContConfig)
    BraxContPPO.learn(config)
