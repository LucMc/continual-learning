from typing import Tuple, Generator, Never
from chex import dataclass
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
import optax
import distrax
from pprint import pprint
from functools import partial
import wandb
import tyro
# from memory_profiler import profile


@dataclass(frozen=True)
class ExperimentConfig:
    """All the options for the experiment, all accessable within PPO class"""

    training_steps: int = 2_000_000
    n_envs: int = 1
    rollout_steps: int = 2048
    env_id: str = "LunarLander-v3"
    batch_size: int = 64
    clip_range: float = 0.2
    epochs: int = 10
    max_grad_norm: int = 0.5
    gamma: float = 0.99
    vf_clip_range: float = np.inf
    ent_coef: float = 0.0
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    vf_coef: float = 0.5
    render: bool = False


class ActorNet(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x) -> distrax.Distribution:
        x = nn.Dense(64)(x)  # be careful of x shape
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.n_actions, name="mu")(x)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (
                1,
                self.n_actions,
            ),
        )
        logstd_batch = jnp.broadcast_to(
            log_std, mean.shape
        )  # Make logstd the same shape as actions
        return distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(logstd_batch)
        )


class ValueNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        q_value = nn.Dense(1)(x)
        return q_value


@dataclass(frozen=True)
class PPO(ExperimentConfig):
    buffer_size: int = 2048

    @partial(jax.jit, static_argnames=["self"])
    def update(
        self,
        actor_ts,
        value_ts,
        key,
        obss,
        actions,
        old_values,
        old_log_probs,
        advantages,
        returns,
    ):
        def actor_loss(
            actor_params, obs_batch, action_batch, old_log_prob_batch, adv_batch
        ):
            dist = actor_ts.apply_fn(actor_params, obs_batch)
            log_prob = dist.log_prob(action_batch)
            entropy = dist.entropy()
            ratio = jnp.exp(log_prob - old_log_prob_batch)

            approx_kl = (
                (old_log_prob_batch - log_prob) ** 2
            ).mean() / 2  # Just for logging
            clip_fraction = (ratio < (1 - self.clip_range)) | (
                ratio > (1 + self.clip_range)
            )
            return (
                -jnp.minimum(
                    ratio * adv_batch,
                    adv_batch
                    * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range),
                ).mean()
                - jnp.mean(entropy) * self.ent_coef
            ), (log_prob.mean(), approx_kl.mean(), clip_fraction.mean())

        def value_loss(value_params, obs_batch, ret_batch, old_val_batch):
            new_values = value_ts.apply_fn(value_params, obs_batch)
            v_clipped = old_val_batch + jnp.clip(
                new_values - old_val_batch, -self.vf_clip_range, self.vf_clip_range
            )
            return self.vf_coef * jnp.mean(
                jnp.maximum((ret_batch - new_values) ** 2, (ret_batch - v_clipped) ** 2)
            ), new_values.mean()
            # Alternatively unclipped (default inf bounds anyway) -- return 0.5 * jnp.mean((ret_batch - new_values) ** 2)  # vf coef

        # Shuffle idxs
        n_minibatches = obss.shape[0]

        actor_loss_total = 0
        value_loss_total = 0
        value_total = 0
        lp_total = 0
        kl_total = 0
        clip_fraction_total = 0

        for i in range(n_minibatches):  # TODO: scan and carry losses?
            # advantage normalisation
            adv_norm = (advantages[i] - advantages[i].mean()) / (
                advantages[i].std() + 1e-8
            )

            (
                (actor_loss_v, (lp_mean, approx_kl_mean, clip_fraction_mean)),
                actor_grads,
            ) = jax.value_and_grad(actor_loss, has_aux=True)(
                actor_ts.params,
                obss[i],
                actions[i],
                old_log_probs[i],
                adv_norm,
            )
            actor_ts = actor_ts.apply_gradients(grads=actor_grads)
            actor_loss_total += actor_loss_v

            (value_loss_v, value_mean), value_grads = jax.value_and_grad(
                value_loss, has_aux=True
            )(value_ts.params, obss[i], returns[i], old_values[i])
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
                    initializer=0.0,
                ),
                "actor_g_mag": jax.tree.reduce(
                    lambda acc, g: acc + jnp.sum(jnp.abs(g)), actor_grads, initializer=0
                ),
            },
        )

    def get_rollout(
        self,
        actor_ts,
        value_ts,
        envs,
        last_obs,
        last_episode_start,
        rollout_steps,
        batch_size,
        key,
    ):
        n_envs = envs.num_envs
        episode_starts = np.zeros((rollout_steps, n_envs))
        truncated = np.zeros((rollout_steps, n_envs))
        # values = np.zeros((rollout_steps, n_envs))
        rewards = np.zeros((rollout_steps, n_envs))
        log_probs = np.zeros((rollout_steps, n_envs))
        stds = np.zeros((rollout_steps,) + envs.action_space.shape)
        obss = np.zeros((rollout_steps,) + envs.observation_space.shape)
        actions = np.zeros((rollout_steps,) + envs.action_space.shape)

        for i in range(rollout_steps):
            action_key, key = random.split(key)
            action_dist = actor_ts.apply_fn(actor_ts.params, jnp.array(last_obs))
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

            episode_start = False
            last_obs = _obs
            last_episode_start = terminated

        values = value_ts.apply_fn(value_ts.params, jnp.array(obss))
        # Fix truncated using value
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

        # Metrics
        ret_var = np.var(returns.flatten())

        explained_var = (
            np.nan
            if ret_var == 0
            else float(1 - np.var(returns.flatten() - values.flatten()) / ret_var)
        )

        print("mean reward:", np.mean(rewards))
        print("stds:", stds.mean())
        print("explained_var:", explained_var)
        # print("actor lr:", actor_ts.opt_state[-1].hyperparams["learning_rate"])
        # print("value lr:", value_ts.opt_state[-1].hyperparams["learning_rate"])

        return (
            jnp.array(obss),
            jnp.array(actions),
            values,
            log_probs,
            advantages,
            returns,
        ), {
            "mean rollout reward": np.mean(rewards),
            "advantage_mean": jnp.mean(advantages),
            "advantage_std": jnp.std(advantages),
            "explained variance": explained_var,
            "actor lr": actor_ts.opt_state[-1].hyperparams["learning_rate"],
            "value lr": value_ts.opt_state[-1].hyperparams["learning_rate"],
        }

    def compute_returns_and_advantage(  # TODO: Replace loop with scan keeping advs ass carry instead of at/set
        self, rewards, values, episode_starts, last_value: Array, done: np.ndarray
    ) -> None:
        buffer_size = values.shape[0]
        advantages = jnp.ones(buffer_size)

        last_gae_lam = 0
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - done.astype(np.float32)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_values = values[step + 1]
            # next values shape (1024, 4, 4 1) check sbx and logic
            delta = (
                rewards[step]
                + self.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            advantages = advantages.at[step].set(last_gae_lam[0])

        returns = advantages + values
        return returns, advantages


def make_env(env_id: str, idx: int, gamma: float, env_args: dict = {}):
    def thunk():
        env = gym.make(env_id, **env_args)
        # env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)

        if "render_mode" in env_args.keys():
            env = gym.wrappers.RecordVideo(
                env, episode_trigger=lambda t: t % 200 == 0, video_folder="./videos/"
            )
        return env

    return thunk


# Alternative minibatch gen for mem-constrained devices
# def get_minibatch(data, idxs):
#     for i in range(n_minibatches):
#         yield data[idxs][i*ppo_agent.batch_size:(i+1)*ppo_agent.batch_size]


# @profile
@partial(jax.jit, static_argnames=["ppo_agent"])
def outer_loop(key, actor_ts, value_ts, rollout, ppo_agent):
    # This stuff doesn't need to be defined every epoch, maybe bring out and parse into this or attach to ppo_agent
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
            perm_key,
            *mb_rollout,
        )
        return (actor_ts, value_ts, key), info

    (actor_ts, value_ts, key), info = jax.lax.scan(
        inner_loop, (actor_ts, value_ts, key), jnp.arange(ppo_agent.epochs)
    )

    # remove to see over epochs, or change to min/max if curious
    info = jax.tree.map(lambda x: x.mean(), info)
    return actor_ts, value_ts, key, info


def main(config):
    # TODO: Wandb watch models and log videos
    ppo_agent = PPO(buffer_size=config.n_envs * config.rollout_steps, **config.__dict__)

    wandb.init(
        project="jax-ppo",
        name="ppo-0.0",
        config=config.__dict__,  # Get from tyro etc
        tags=["PPO", ppo_agent.env_id],  # Maybe put env id here or something
    )

    env_args = (
        {"render_mode": "rgb_array", "continuous": True}
        if ppo_agent.render
        else {"continuous": True}
    )
    ckpt_path = "./checkpoints"
    assert not ppo_agent.rollout_steps % ppo_agent.batch_size, (
        "Must have rollout steps divisible into batches"
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(ppo_agent.env_id, i, ppo_agent.gamma, env_args=env_args)
            for i in range(ppo_agent.n_envs)
        ]
    )

    dummy_obs, _ = envs.reset()
    key = random.PRNGKey(0)
    current_global_step = 0

    actor_key, value_key, key = random.split(key, num=3)

    actor_net = ActorNet(envs.action_space.shape[-1])
    value_net = ValueNet()
    opt = optax.chain(
        optax.clip_by_global_norm(ppo_agent.max_grad_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=optax.linear_schedule(
                init_value=ppo_agent.learning_rate,
                end_value=ppo_agent.learning_rate / 10,
                transition_steps=ppo_agent.training_steps,
            ),
        ),
    )

    actor_ts = TrainState.create(
        apply_fn=actor_net.apply, params=actor_net.init(actor_key, dummy_obs), tx=opt
    )
    value_ts = TrainState.create(
        apply_fn=value_net.apply, params=value_net.init(value_key, dummy_obs), tx=opt
    )

    last_obs, first_info = envs.reset()
    last_episode_starts = np.ones((ppo_agent.n_envs,), dtype=bool)

    while current_global_step < ppo_agent.training_steps:
        # TODO: Add if statement to reduce rollout if current_global_step is near total_training_steps
        print("\ncurrent_global_step:", current_global_step)
        rollout, rollout_info = ppo_agent.get_rollout(
            actor_ts,
            value_ts,
            envs,
            last_obs,
            last_episode_starts,
            ppo_agent.rollout_steps,
            ppo_agent.batch_size,
            key,
        )

        current_global_step += ppo_agent.rollout_steps * ppo_agent.n_envs

        actor_ts, value_ts, key, training_info = outer_loop(
            key, actor_ts, value_ts, rollout, ppo_agent
        )
        full_logs = training_info | rollout_info
        pprint(full_logs)

        wandb.log(full_logs, step=current_global_step)

        if current_global_step % 100_000 == 0:
            wandb.save(ckpt_path)


if __name__ == "__main__":
    config = tyro.cli(ExperimentConfig)
    main(config)

# _reward = np.where(truncated, self.gamma * value_ts.apply_fn(value_ts.params, jnp.array(_obs)).item(), reward) # should be added to r anyway
