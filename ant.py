from typing import Callable, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from brax import envs as brax_envs
from brax.envs.base import Env
from flax.core.scope import DenyList
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray

import wandb
from continual_learning.configs.models import MLPConfig
from continual_learning.configs.optim import AdamConfig
from continual_learning.configs.rl import PolicyNetworkConfig, ValueFunctionConfig
from continual_learning.models import get_model, get_model_cls
from continual_learning.models.rl import Policy
from continual_learning.types import Activation, StdType


class Transition(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    values: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    next_observations: jax.Array
    log_probs: jax.Array
    extra: dict


def compute_gae(rewards, values, last_values, gamma, gae_lambda, terminations, truncations):
    truncation_mask = 1 - truncations
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate([values[1:], last_values[None, ...]], axis=0)
    deltas = rewards + gamma * (1 - terminations) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(last_values)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + gamma * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs,
        (gae_lambda, acc),
        (truncation_mask, deltas, terminations),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], last_values[None, ...]], axis=0)
    advantages = (
        rewards + gamma * (1 - terminations) * vs_t_plus_1 - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def update(
    policy: TrainState,
    vf: TrainState,
    data: Transition,
    key: PRNGKeyArray,
    entropy_coeff,
    vf_coeff,
    gamma,
    gae_lambda,
    reward_scale,
    clip_eps,
    norm_advantage,
    num_minibatches,
    num_epochs,
):
    final_obs = jax.tree.map(lambda x: x[-1], data.next_observations)
    last_values = vf.apply_fn(
        vf.params, final_obs, training=True, rngs={"dropout": key}
    ).squeeze(-1)
    rewards = data.rewards * reward_scale

    value_targets, advantages = compute_gae(
        rewards,
        data.values,
        last_values,
        gamma,
        gae_lambda,
        data.terminations,
        data.truncations,
    )

    def loss(
        policy_and_vf_params,
        key,
        data: Transition,
        advantages: jax.Array,
        value_targets: jax.Array,
    ):
        policy_params, vf_params = policy_and_vf_params

        values = vf.apply_fn(
            vf_params, data.observations, training=True, rngs={"dropout": key}
        ).squeeze(-1)

        if norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy
        dist = policy.apply_fn(
            policy_params, data.observations, training=True, rngs={"dropout": key}
        )
        log_probs = dist.log_prob(data.actions)
        ratio = jnp.exp(log_probs - data.log_probs)
        policy_loss = -jnp.minimum(
            ratio * advantages, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        ).mean()

        entropy_loss = entropy_coeff * dist.entropy().mean()

        # VF
        vf_loss = vf_coeff * 0.5 * jnp.power(value_targets - values, 2).mean()

        total_loss = policy_loss + vf_loss + entropy_loss

        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": vf_loss,
            "entropy_loss": entropy_loss,
        }

    def update_minibatch(carry, xs):
        policy, vf, key = carry
        data, advantages, value_targets = xs

        key, dropout_key = jax.random.split(key)

        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
            (policy.params, vf.params), dropout_key, data, advantages, value_targets
        )

        policy = policy.apply_gradients(grads=grads[0])
        vf = vf.apply_gradients(grads=grads[1])

        return (policy, vf, key), metrics

    def update_epoch(carry, _):
        policy, vf, key = carry
        key, perm_key = jax.random.split(key)

        def shuffle(x: jax.Array):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(perm_key, x, axis=0)
            x = jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))
            return x

        (policy, vf, key), metrics = jax.lax.scan(
            update_minibatch,
            (policy, vf, key),
            jax.tree.map(shuffle, (data, advantages, value_targets)),
            length=num_minibatches,
        )

        return (policy, vf, key), metrics

    (policy, vf, key), metrics = jax.lax.scan(
        update_epoch, (policy, vf, key), None, length=num_epochs
    )

    return (policy, vf, key), metrics


def rollout(
    policy: TrainState,
    vf: TrainState,
    envs: Env,
    env_states,
    key: PRNGKeyArray,
    rollout_size: int,
    num_envs: int,
):
    def step(carry, _):
        env_states, key = carry
        key, action_key, dropout_key = jax.random.split(key, 3)
        action, log_prob = policy.apply_fn(
            policy.params, env_states.obs, training=True, rngs={"dropout": dropout_key}
        ).sample_and_log_prob(seed=action_key)
        values = vf.apply_fn(
            vf.params, env_states.obs, training=True, rngs={"dropout": dropout_key}
        ).squeeze(-1)
        next_state = envs.step(env_states, action)

        truncations = next_state.info["truncation"]
        data = Transition(
            observations=env_states.obs,
            actions=action,
            rewards=next_state.reward,
            values=values,
            terminations=next_state.done * (1 - truncations),
            truncations=truncations,
            next_observations=next_state.obs,  # pyright: ignore[reportArgumentType]
            log_probs=log_prob,
            extra={
                "sum_reward": next_state.info["episode_metrics"]["sum_reward"],
                "episode_done": next_state.info["episode_done"],
            },
        )

        return (next_state, key), data

    (env_states, key), data = jax.lax.scan(
        step,
        (env_states, key),
        None,
        length=rollout_size // num_envs,
    )

    return env_states, key, data  # (T, Envs, D)


def main():
    wandb.init(
        project="continual_learning",
        name="brax_ppo_scratch_4_attempt_11",
        tags=["BRAX-PPO", "ant"],
        # mode="disabled",
    )

    SEED = 46

    TOTAL_TIMESTEPS = 50_000_000

    BATCH_SIZE = 2048
    NUM_ENVS = 4096
    NUM_MINIBATCHES = 32
    UNROLL_LENGTH = 5
    action_repeat = 1

    LEARNING_RATE = 3e-4

    # PPO
    ENTROPY_COEFF = 1e-2
    VF_COEFF = 0.5
    GAMMA = 0.97
    GAE_LAMBDA = 0.95
    REWARD_SCALE = 10.0
    CLIP_EPS = 0.3
    NORM_ADVANTAGE = True
    NUM_EPOCHS = 4

    envs = brax_envs.training.wrap(
        brax_envs.get_environment(env_name="ant"),
        episode_length=1000,
        action_repeat=action_repeat,
    )

    rollout_size = (
        BATCH_SIZE * NUM_MINIBATCHES * UNROLL_LENGTH * action_repeat
    )  # Keep rollout size same

    key = jax.random.PRNGKey(SEED)
    key, key_policy, key_value, key_env = jax.random.split(key, 4)

    reset_fn = jax.jit(envs.reset)
    key_envs = jax.random.split(key_env, NUM_ENVS)
    env_states = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, env_states.obs)
    obs_spec = jax.ShapeDtypeStruct(obs_shape, jnp.float32)

    policy_config = PolicyNetworkConfig(
        optimizer=AdamConfig(learning_rate=3e-4),
        network=MLPConfig(
            num_layers=4,
            hidden_size=32,
            output_size=envs.action_size,
            activation_fn=Activation.Swish,
            kernel_init=jax.nn.initializers.lecun_normal(),
            dtype=jnp.float32,
        ),
        std_type=StdType.MLP_HEAD,
    )
    vf_config = ValueFunctionConfig(
        optimizer=AdamConfig(learning_rate=3e-4),
        network=MLPConfig(
            num_layers=5,
            hidden_size=256,
            output_size=1,
            activation_fn=Activation.Swish,
            kernel_init=jax.nn.initializers.lecun_normal(),
            dtype=jnp.float32,
        ),
    )

    policy_network = get_model_cls(policy_config.network)
    policy_module = Policy(policy_network, policy_config)
    policy = TrainState.create(
        apply_fn=policy_module.apply,
        params=policy_module.lazy_init(
            key_policy,
            obs_spec,
            training=False,
            mutable=DenyList(["activations", "preactivations"]),
        ),
        tx=optax.adam(learning_rate=LEARNING_RATE),
    )

    vf_module = get_model(vf_config.network)
    vf = TrainState.create(
        apply_fn=vf_module.apply,
        params=vf_module.lazy_init(
            key_value,
            obs_spec,
            training=False,
            mutable=DenyList(["activations", "preactivations"]),
        ),
        tx=optax.adam(learning_rate=LEARNING_RATE),
    )

    @jax.jit
    def train_step(policy, vf, env_states, i, key):
        env_states, key, data = rollout(
            policy, vf, envs, env_states, key, rollout_size, NUM_ENVS
        )

        (policy, vf, key), metrics = update(
            policy,
            vf,
            data,
            key,
            ENTROPY_COEFF,
            VF_COEFF,
            GAMMA,
            GAE_LAMBDA,
            REWARD_SCALE,
            CLIP_EPS,
            NORM_ADVANTAGE,
            NUM_MINIBATCHES,
            NUM_EPOCHS,
        )

        logs = jax.tree.map(jnp.mean, metrics)

        return policy, vf, env_states, i + rollout_size, key, logs, data.extra

    i = 0
    for _ in range(0, TOTAL_TIMESTEPS, rollout_size):
        policy, vf, env_states, i, key, logs, extras = train_step(
            policy, vf, env_states, i, key
        )
        logs = logs | {
            "episode_return": extras["sum_reward"][extras["episode_done"].astype(bool)].mean()
        }
        wandb.log(logs, step=i)

    wandb.finish()


if __name__ == "__main__":
    main()
