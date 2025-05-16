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
from typing import Tuple, Literal
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
from continual_learning.utils.miscellaneous import compute_plasticity_metrics
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
    cbp: bool = False  # Weather or not to use continual backpropergation
    optim: Literal["adam", "adamw", "sgd", "muon", "muonw"] = "muonw"
    run_name: str = ""  # Postfix name for training run
    delay_type: Literal[
        "none", "random", "random_incremental", "constant", "incremental"
    ] = "none"
    changes: int = 10  # How many env changes for continual learning (if using ContinualAnt-v0 or delay=True)


@dataclass(frozen=True)
class ContPPO(PPO, ContConfig):
    buffer_size: int = 2048

    # @jaxtyped(typechecker=typechecker)
    @partial(jax.jit, static_argnames=["self", "apply_fn"])
    def actor_loss(self, actor_params, apply_fn, obs_batch, action_batch, old_log_prob_batch, adv_batch):
        dist = apply_fn(actor_params, obs_batch)
        log_prob = dist.log_prob(action_batch)
        entropy = dist.entropy()
        ratio = jnp.exp(log_prob - old_log_prob_batch)

        approx_kl = ((old_log_prob_batch - log_prob) ** 2).mean() / 2  # Just for logging
        clip_fraction = (ratio < (1 - self.clip_range)) | (ratio > (1 + self.clip_range))
        return (
            -jnp.minimum(
                ratio * adv_batch,
                adv_batch * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range),
            ).mean()
            - jnp.mean(entropy) * self.ent_coef
        ), (log_prob.mean(), approx_kl.mean(), clip_fraction.mean())


    @partial(jax.jit, static_argnames=["self", "apply_fn"])
    def value_loss(self, value_params, apply_fn, obs_batch, ret_batch, old_val_batch):
        new_values = apply_fn(value_params, obs_batch)
        v_clipped = old_val_batch + jnp.clip(
            new_values - old_val_batch, -self.vf_clip_range, self.vf_clip_range
        )
        return self.vf_coef * jnp.mean(
            jnp.maximum((ret_batch - new_values) ** 2, (ret_batch - v_clipped) ** 2)
        ), new_values.mean()
        # Alternatively unclipped (default inf bounds anyway) -- return 0.5 * jnp.mean((ret_batch - new_values) ** 2)  # vf coef


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
        # TODO: Add pre/postfix to indicate debug/logging variables?
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
                (actor_loss_v, (lp_mean, approx_kl_mean, clip_fraction_mean)),
                actor_grads,
            ) = jax.value_and_grad(self.actor_loss, has_aux=True)(
                actor_ts.params,
                actor_ts.apply_fn,
                obss[i],
                actions[i],
                old_log_probs[i],
                adv_norm,
            )
            actor_ts = actor_ts.apply_gradients(grads=actor_grads)
            actor_loss_total += actor_loss_v

            (value_loss_v, value_mean), value_grads = jax.value_and_grad(
                self.value_loss, has_aux=True
            )(value_ts.params, value_ts.apply_fn, obss[i], returns[i], old_values[i])
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

    def get_rollout(
        self,
        actor_ts: TrainState,
        value_ts: TrainState,
        envs: gym.vector.VectorEnv,
        last_obs: np.ndarray,  # Array["#n_envs"]
        last_episode_start: np.ndarray,  # Array["#n_envs"]
        key: PRNGKeyArray,
    ):
        rollout_size = self.rollout_steps // self.n_envs
        episode_starts = np.zeros((rollout_size, self.n_envs))
        truncated = np.zeros((rollout_size, self.n_envs))
        # values = np.zeros((self.rollout_steps, n_envs))
        rewards = np.zeros((rollout_size, self.n_envs))
        log_probs = np.zeros((rollout_size, self.n_envs))
        stds = np.zeros((rollout_size,) + envs.action_space.shape)
        obss = np.zeros((rollout_size,) + envs.observation_space.shape)
        actions = np.zeros((rollout_size,) + envs.action_space.shape)
        env_infos = []

        for i in range(self.rollout_steps // self.n_envs):
            action_key, key = random.split(key)
            action_dist = actor_ts.apply_fn(actor_ts.params, jnp.array(last_obs))
            action = action_dist.sample(seed=action_key)
            log_prob = action_dist.log_prob(action)

            value = value_ts.apply_fn(value_ts.params, jnp.array(last_obs))

            _obs, reward, terminated, truncated, info = envs.step(np.array(action))

            # TODO: Check shapes are correct for multiple envs
            rewards[i] = reward
            actions[i] = action
            episode_starts[i] = last_episode_start
            log_probs[i] = log_prob
            obss[i] = last_obs
            stds[i] = action_dist.stddev()
            env_infos.append(info)

            episode_start = False
            last_obs = _obs
            last_episode_start = terminated

        values = value_ts.apply_fn(value_ts.params, jnp.array(obss))
        last_values = value_ts.apply_fn(value_ts.params, jnp.array(last_obs))

        returns, advantages = jax.vmap(
            self.compute_returns_and_advantage, in_axes=(1, 1, 1, 0, 0)
        )(
            rewards,
            values.squeeze(axis=-1),
            episode_starts,
            last_values,
            last_episode_start,
        )

        returns = returns.T
        advantages = advantages.T

        # Metrics
        ret_var = np.var(returns.flatten())

        explained_var = (
            np.nan
            if ret_var == 0
            else float(1 - (np.var(advantages.flatten()) / ret_var))
        )

        rollout_info = {
            "mean rollout reward": np.mean(rewards),  # TODO: ignore the last episode
            "advantage_mean": jnp.mean(advantages),
            "advantage_std": jnp.std(advantages),
            "explained variance": explained_var,
            "actor lr": actor_ts.opt_state[-1].hyperparams["learning_rate"],
            "action_dist_std": stds.mean(),
            "value lr": value_ts.opt_state[-1].hyperparams["learning_rate"],
        }

        return (
            (
                jnp.array(obss),
                jnp.array(actions),
                values,
                log_probs,
                advantages,
                returns,
            ),
            rollout_info,
            env_infos,
        )


def make_env(ppo_agent: PPO, idx: int, video_folder: str = None, env_args: dict = {}):
    def thunk():
        if ppo_agent.delay_type != "none":
            print(":: Added continual time delays ::")
            change_every = env_args.pop("change_every")
            env = gym.make(ppo_agent.env_id, **env_args)
            # env = ContinualRandomIntervalDelayWrapper(
            env = ContinualIntervalDelayWrapper(
                env,
                change_every=change_every,
                obs_delay_range=range(0, 4),
                act_delay_range=range(0, 4),
                delay_type=ppo_agent.delay_type,
            )

        else:
            env = gym.make(ppo_agent.env_id, **env_args)

        # env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.NormalizeReward(env, gamma=0.99) # TODO: replace with actual gamma
        if isinstance(env.observation_space, gym.spaces.Dict):
            print(f":: Original observation space: {env.observation_space}")
            env = gym.wrappers.FlattenObservation(env)
            print(f":: Flattened observation space: {env.observation_space}")
        # >>> END OF ADDITION <<<

        if ppo_agent.log_video_every > 0 and idx == 0:
            print(":: Recording Videos ::")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder,
                lambda t: t % ppo_agent.log_video_every == 0,
            )
        return env

    return thunk


# @profile
@partial(jax.jit, static_argnames=["ppo_agent"])
def outer_loop(
    key: PRNGKeyArray,
    actor_ts: TrainState,
    value_ts: TrainState,
    rollout: Tuple,
    ppo_agent: PPO,
):
    n_minibatches = ppo_agent.buffer_size // ppo_agent.batch_size
    swap_and_reshape = lambda x: jnp.swapaxes(x, 0, 1).reshape(
        (ppo_agent.buffer_size,) + x.shape[2:]
    )

    shape_minibatches = lambda x, idxs: x[idxs].reshape(
        (n_minibatches, ppo_agent.batch_size) + x.shape[1:]
    )

    actor_ts_pams = actor_ts.params
    flat_rollout = tuple(map(swap_and_reshape, rollout))

    def inner_loop(carry, _):
        actor_ts, value_ts, key = carry
        key, perm_key = random.split(key)
        idxs = random.permutation(perm_key, ppo_agent.buffer_size)
        mb_rollout = tuple(shape_minibatches(x, idxs) for x in flat_rollout)

        actor_ts, value_ts, info = PPO.update(
            ppo_agent,
            actor_ts,
            value_ts,
            *mb_rollout,
        )
        return (actor_ts, value_ts, key), info

    (_actor_ts, _value_ts, key), info = jax.lax.scan(
        inner_loop, (actor_ts, value_ts, key), jnp.arange(ppo_agent.epochs)
    )

    # remove to see over epochs, or change to min/max if curious
    # TODO: Is info over last epoch better?
    act_plasticity = compute_plasticity_metrics(
        actor_ts.params, _actor_ts.params, ppo_agent.learning_rate, label="actor"
    )
    val_plasticity = compute_plasticity_metrics(
        value_ts.params, _value_ts.params, ppo_agent.learning_rate, label="critic"
    )

    info = jax.tree.map(lambda x: x.mean(), info) | act_plasticity | val_plasticity
    return _actor_ts, _value_ts, key, info


def main(config: ContConfig):
    ppo_agent = ContPPO(
        buffer_size=config.rollout_steps,
        **config.__dict__,
    )
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
        tags = ["PPO", ppo_agent.env_id, ppo_agent.optim, ppo_agent.delay_type]
        # NOTE: If using layernorm, increase learning rate to 0.0005
        # fmt: off
        if ppo_agent.layer_norm: tags.append("LayerNorm")
        if ppo_agent.cbp: tags.append("ContinualBackprop")
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
            make_env(ppo_agent, i, video_folder=video_folder, env_args=env_args)
            for i in range(ppo_agent.n_envs)
        ]
    )

    dummy_obs, _ = envs.reset(seed=ppo_agent.seed)
    key = random.PRNGKey(ppo_agent.seed)
    current_global_step = 0

    actor_key, value_key, key = random.split(key, num=3)

    if ppo_agent.layer_norm:
        print(":: Using LayerNorm layers ::")
        actor_net = ActorNetLayerNorm(
            envs.action_space.shape[-1]
        )  # Have these as options
        value_net = ValueNetLayerNorm()
    else:
        print(":: Using standard architecture ::")
        actor_net = ActorNet(envs.action_space.shape[-1])  # Have these as options
        value_net = ValueNet()

    # Select optimiser
    # fmt: off
    if ppo_agent.optim == "adam": tx = optax.adam
    if ppo_agent.optim == "adamw": tx = optax.adamw
    if ppo_agent.optim == "sgd": tx = optax.sgd
    if ppo_agent.optim == "muon": tx = optax.contrib.muon
    if ppo_agent.optim == "muonw": tx = partial(optax.contrib.muon, weight_decay=0.0001)
    # fmt: on

    opt = optax.chain(
        optax.clip_by_global_norm(ppo_agent.max_grad_norm),
        optax.inject_hyperparams(tx)(
            learning_rate=optax.linear_schedule(  # Does this have an adverse effect of continual learning?
                init_value=ppo_agent.learning_rate,
                end_value=ppo_agent.learning_rate / 10,
                transition_steps=2_000_000,  # ppo_agent.training_steps
            ),
        ),
    )

    # Continual backpropergation
    if ppo_agent.cbp:
        cbp_value_key, cbp_actor_key, key = random.split(key, num=3)

        actor_ts = CBPTrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_key, dummy_obs),
            tx=opt,
            rng=cbp_actor_key
        )
        value_ts = CBPTrainState.create(
            apply_fn=value_net.apply,
            params=value_net.init(value_key, dummy_obs),
            tx=opt,
            rng=cbp_value_key
        )
    else:
        actor_ts = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_key, dummy_obs),
            tx=opt,
        )
        value_ts = TrainState.create(
            apply_fn=value_net.apply,
            params=value_net.init(value_key, dummy_obs),
            tx=opt,
        )

    last_obs, first_info = envs.reset()
    last_episode_starts = np.ones((ppo_agent.n_envs,), dtype=bool)

    while current_global_step < ppo_agent.training_steps:
        print("\ncurrent_global_step:", current_global_step)
        rollout, rollout_info, env_infos = ppo_agent.get_rollout(
            actor_ts,
            value_ts,
            envs,
            last_obs,
            last_episode_starts,
            key,
        )

        current_global_step += ppo_agent.rollout_steps * ppo_agent.n_envs

        actor_ts, value_ts, key, training_info = outer_loop(
            key, actor_ts, value_ts, rollout, ppo_agent
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
    main(config)

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
