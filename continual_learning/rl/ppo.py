from ast import Tuple
from chex import dataclass
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import numpy as np
import optax
import distrax
from pprint import pprint
from functools import partial
import wandb
import tyro
import os
from pathlib import Path
from continual_learning.nn import ActorNet, ValueNet
from continual_learning.utils.metrics import compute_plasticity_metrics

"""
Base PPO class with networks in ../networks/nn.py modified to pass features to optimiser
"""


@dataclass(frozen=True)
class Config:
    """All the options for the experiment, these are all loaded into 'self' in PPO class"""

    seed: int = 0  # Random seed
    training_steps: int = 500_000 * 2  # total training time-steps
    n_envs: int = 1 * 2  # number of parralel training envs
    rollout_steps: int = 64 * 20 * 2  # env steps per rollout
    env_id: str = "Ant-v5"
    batch_size: int = 64 * 2  # minibatch size
    clip_range: float = 0.2  # policy clip range
    epochs: int = 10  # number of epochs for fitting mini-batches
    max_grad_norm: float = 0.5  # maximum gradient norm
    gamma: float = 0.99  # discount factor
    vf_clip_range: float = np.inf  # vf clipping (typically higher than clip_range)
    ent_coef: float = 0.0  # how much exploration?
    gae_lambda: float = 0.95  # bias-variance tradeoff in gae
    learning_rate: float = 3e-4  # lr for both actor and critic
    vf_coef: float = 0.5  # balance vf loss magnitude
    log_video_every: int = 0  # save video locally/wandb every X time-steps
    log: bool = False  # log with wandb


@dataclass(frozen=True)
class PPO(Config):
    buffer_size: int = 2048

    @partial(jax.jit, static_argnames=["self", "apply_fn"])
    def actor_loss(
        self, actor_params, apply_fn, obs_batch, action_batch, old_log_prob_batch, adv_batch
    ):
        (mean, scale), actor_features = apply_fn(actor_params, obs_batch)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=scale)
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
        ), (log_prob.mean(), approx_kl.mean(), clip_fraction.mean(), actor_features)

    @partial(jax.jit, static_argnames=["self", "apply_fn"])
    def value_loss(self, value_params, apply_fn, obs_batch, ret_batch, old_val_batch):
        new_values, value_features = apply_fn(value_params, obs_batch)
        v_clipped = old_val_batch + jnp.clip(
            new_values - old_val_batch, -self.vf_clip_range, self.vf_clip_range
        )
        return self.vf_coef * jnp.mean(
            jnp.maximum((ret_batch - new_values) ** 2, (ret_batch - v_clipped) ** 2)
        ), (new_values.mean(), value_features)
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
            actor_ts = actor_ts.apply_gradients(grads=actor_grads)
            actor_loss_total += actor_loss_v

            (value_loss_v, (value_mean, value_features)), value_grads = jax.value_and_grad(
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

    # @jaxtyped(typechecker=typechecker)
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
        rewards = np.zeros((rollout_size, self.n_envs))
        log_probs = np.zeros((rollout_size, self.n_envs))

        stds = np.zeros((rollout_size,) + envs.action_space.shape)
        obss = np.zeros((rollout_size,) + envs.observation_space.shape)
        actions = np.zeros((rollout_size,) + envs.action_space.shape)
        infos = []

        for i in range(self.rollout_steps // self.n_envs):
            action_key, key = random.split(key)
            (mean, scale), _ = actor_ts.apply_fn(actor_ts.params, jnp.array(last_obs))
            action_dist = distrax.MultivariateNormalDiag(
                loc=mean, scale_diag=scale
            )  # Create here
            action = action_dist.sample(seed=action_key)
            log_prob = action_dist.log_prob(action)

            value = value_ts.apply_fn(value_ts.params, jnp.array(last_obs))

            _obs, reward, terminated, truncated, info = envs.step(np.array(action))

            rewards[i] = reward
            actions[i] = action
            episode_starts[i] = last_episode_start
            log_probs[i] = log_prob
            obss[i] = last_obs
            stds[i] = action_dist.stddev()
            infos.append(info)

            last_obs = _obs
            last_episode_start = terminated

        values, _ = value_ts.apply_fn(value_ts.params, jnp.array(obss))
        last_values, _ = value_ts.apply_fn(value_ts.params, jnp.array(last_obs))

        returns, advantages = jax.vmap(
            self.compute_returns_and_advantage, in_axes=(1, 1, 1, 0, 0)
        )(
            rewards,
            values.squeeze(axis=-1),  # remove squeeze
            episode_starts,
            last_values,
            last_episode_start,
        )

        returns = returns.T
        advantages = advantages.T

        rollout_info = {
            "mean_rollout_reward": np.mean(rewards),
            "advantage_mean": jnp.mean(advantages),
            "advantage_std": jnp.std(advantages),
            "explained_variance": float(
                1 - (np.var(advantages.flatten()) / np.var(returns.flatten()))
            ),
            # "actor_lr": actor_ts.opt_state[0][1].hyperparams["learning_rate"], # TODO
            "action_dist_std": stds.mean(),
            # "value_lr": value_ts.opt_state[0][1].hyperparams["learning_rate"],
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
            infos,
        )

    # @jaxtyped(typechecker=typechecker)
    @partial(jax.jit, static_argnames="self")
    def compute_returns_and_advantage(
        self,
        rewards: Float[Array, "#rollout_steps"],
        values: Float[Array, "#rollout_steps"],
        episode_starts: Float[Array, "#rollout_steps"],
        last_value: Float[Array, "1"],
        done: Array,
    ) -> tuple[Array, Array]:
        buffer_size = values.shape[0]

        # for step in reversed(range(buffer_size)):
        def gae_step(last_gae_lam, step):
            next_non_terminal, next_values = jax.lax.cond(
                step == buffer_size - 1,
                lambda: (1.0 - done, last_value[0]),
                lambda: (1.0 - episode_starts[step + 1], values[step + 1]),
            )

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

            return last_gae_lam, last_gae_lam[0]

        _, advantages = jax.lax.scan(
            gae_step, jnp.array([0.0]), jnp.arange(buffer_size), reverse=True
        )

        returns = advantages + values
        return returns, advantages

    def make_env(self, idx: int, video_folder: str = None, env_args: dict = {}):
        def thunk():
            env = gym.make(self.env_id, **env_args)
            # env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            # env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.NormalizeReward(env, gamma=self.gamma) # TODO: replace with actual gamma
            if self.log_video_every and idx == 0:
                print(":: Recording Videos ::")
                env = gym.wrappers.RecordVideo(
                    env,
                    episode_trigger=lambda t: t % 20 == 0,
                    video_folder=video_folder,
                )
            return env

        return thunk

    @partial(jax.jit, static_argnames=["self"])
    def outer_loop(
        self,
        key: PRNGKeyArray,
        actor_ts: TrainState,
        value_ts: TrainState,
        rollout: Tuple,
    ):
        n_minibatches = self.buffer_size // self.batch_size
        swap_and_reshape = lambda x: jnp.swapaxes(x, 1, 1).reshape(
            (self.buffer_size,) + x.shape[2:]
        )

        shape_minibatches = lambda x, idxs: x[idxs].reshape(
            (n_minibatches, self.batch_size) + x.shape[1:]
        )

        flat_rollout = tuple(map(swap_and_reshape, rollout))

        def inner_loop(carry, _):
            actor_ts, value_ts, key = carry
            key, perm_key = random.split(key)
            idxs = random.permutation(perm_key, self.buffer_size)
            mb_rollout = tuple(shape_minibatches(x, idxs) for x in flat_rollout)

            actor_ts, value_ts, info = self.update(
                actor_ts,
                value_ts,
                *mb_rollout,
            )
            
            return (actor_ts, value_ts, key), info

        (_actor_ts, _value_ts, key), info = jax.lax.scan(
            inner_loop, (actor_ts, value_ts, key), jnp.arange(self.epochs)
        )

        # Add other metrics here if needed
        act_plasticity = compute_plasticity_metrics(
            actor_ts.params, _actor_ts.params, self.learning_rate, label="actor"
        )
        val_plasticity = compute_plasticity_metrics(
            value_ts.params, _value_ts.params, self.learning_rate, label="critic"
        )

        info = jax.tree.map(lambda x: x.mean(), info) | act_plasticity | val_plasticity
        return _actor_ts, _value_ts, key, info

    def setup_network_trainstates(
        self,
        obs,
        act_shape,
        actor_key,
        value_key,
        opt,
        actor_net_cls=ActorNet,
        value_net_cls=ValueNet,
        trainstate_cls=TrainState,
        reset_method_kwargs={},
        reset_method="cbp"
    ):
        actor_net = actor_net_cls(act_shape)
        value_net = value_net_cls()

        actor_ts = trainstate_cls.create(
            apply_fn=actor_net.apply_w_features,
            params=actor_net.init(actor_key, obs),
            tx=opt,
            reset_method=reset_method,
            reset_method_kwargs=reset_method_kwargs
        )
        value_ts = trainstate_cls.create(
            apply_fn=value_net.apply_w_features,
            params=value_net.init(value_key, obs),
            tx=opt,
            reset_method=reset_method,
            reset_method_kwargs=reset_method_kwargs)
        return actor_ts, value_ts

    @staticmethod
    def learn(config: Config):
        ppo_agent = PPO(buffer_size=config.rollout_steps, **config.__dict__)
        np.random.seed(ppo_agent.seed)  # Seeding for np operations

        if ppo_agent.log_video_every:
            base_video_dir = Path("videos")
            video_folder = base_video_dir / str(
                len(os.listdir(base_video_dir))
            )  # run_id for local videos
            os.makedirs(video_folder)
            env_args = {"render_mode": "rgb_array"}
        else:
            video_folder = None
            env_args = {}

        if ppo_agent.log:
            wandb.init(
                project="jax-ppo",
                name="ppo",
                config=config.__dict__,  # Get from tyro etc
                tags=["PPO", ppo_agent.env_id],
                # monitor_gym=True,
                save_code=True,
            )

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

        last_obs, first_info = envs.reset()
        key = random.PRNGKey(ppo_agent.seed)
        current_global_step = 0

        actor_key, value_key, key = random.split(key, num=3)
        actor_ts, value_ts = ppo_agent.setup_network_trainstates(
            last_obs, envs.single_action_space.shape[0], actor_key, value_key
        )
        last_episode_starts = np.ones((ppo_agent.n_envs,), dtype=bool)

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

            env_infos = {}  # Change this if there is anything in your env info you want to plot, i.e {"episode_length": env_infos["episode_length"]}

            full_logs = training_info | rollout_info | env_infos
            pprint(full_logs)

            if ppo_agent.log:
                wandb.log(full_logs, step=current_global_step)

                if current_global_step % 100_000 * ppo_agent.n_envs == 0:
                    wandb.save(ckpt_path)

        ppo_agent.cleanup()


    def cleanup(self):
        # Close stuff
        if self.log:
            # if abs(current_global_step % ppo_agent.log_video_every) < ppo_agent.rollout_steps:
            if self.log_video_every > 0:
                print("[ ] Uploading Videos ...", end="\r")
                for video_name in os.listdir(video_folder):
                    wandb.log({video_name: wandb.Video(str(base_video_dir / video_name))})
                print(r"[x] Uploading Videos ...")

            wandb.finish()

if __name__ == "__main__":
    config = tyro.cli(Config)
    PPO.learn(config)

# Alternative minibatch gen for mem-constrained devices
# def get_minibatch(data, idxs):
#     for i in range(n_minibatches):
#         yield data[idxs][i*ppo_agent.batch_size:(i+2)*ppo_agent.batch_size]
