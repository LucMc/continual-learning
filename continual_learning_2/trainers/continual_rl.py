import time
from functools import partial
from typing import NamedTuple

import jax
import jax.experimental
import jax.flatten_util
import jax.numpy as jnp
from flax.core.scope import DenyList
from jaxtyping import PRNGKeyArray

from continual_learning_2.configs.envs import EnvConfig
from continual_learning_2.configs.logging import LoggingConfig
from continual_learning_2.configs.models import MLPConfig
from continual_learning_2.configs.optim import AdamConfig
from continual_learning_2.configs.rl import PolicyNetworkConfig, PPOConfig, ValueFunctionConfig
from continual_learning_2.configs.training import RLTrainingConfig
from continual_learning_2.envs import JittableContinualLearningEnv, get_benchmark
from continual_learning_2.envs.base import JittableVectorEnv
from continual_learning_2.models import get_model, get_model_cls
from continual_learning_2.models.rl import Policy
from continual_learning_2.optim import get_optimizer
from continual_learning_2.types import (
    Activation,
    EnvState,
    LogDict,
    Observation,
    Rollout,
    StdType,
)
from continual_learning_2.utils.buffers import compute_gae_scan
from continual_learning_2.utils.monitoring import (
    Logger,
    accumulate_concatenated_metrics,
    explained_variance,
    get_logs,
    prefix_dict,
    pytree_histogram,
)
from continual_learning_2.utils.training import TrainState


class PPO:
    @staticmethod
    @partial(jax.jit, static_argnames=("cfg"), donate_argnames=("policy", "vf", "key"))
    def update(
        key: PRNGKeyArray,
        policy: TrainState,
        vf: TrainState,
        data: Rollout,
        next_obs: Observation,
        cfg: PPOConfig,
    ) -> tuple[PRNGKeyArray, TrainState, TrainState, LogDict]:
        assert data.values is not None and data.log_probs is not None

        key, last_values_dropout_key = jax.random.split(key)
        last_values = vf.apply_fn(
            vf.params, next_obs, training=True, rngs={"dropout": last_values_dropout_key}
        ).squeeze(-1)
        value_targets, advantages = compute_gae_scan(
            data, data.values, last_values, cfg.gamma, cfg.gae_lambda
        )

        def loss(
            policy_and_vf_params,
            key: PRNGKeyArray,
            data: Rollout,
            advantages: jax.Array,
            value_targets: jax.Array,
        ):
            policy_params, vf_params = policy_and_vf_params
            key, dropout_key = jax.random.split(key)

            values, value_intermediates = vf.apply_fn(
                vf_params,
                data.observations,
                training=True,
                rngs={"dropout": dropout_key},
                mutable=("activations", "preactivations")
            )
            values = values.squeeze(-1)

            if cfg.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy
            dist, actor_intermediates = policy.apply_fn(
                policy_params,
                data.observations,
                training=True,
                rngs={"dropout": dropout_key},
                mutable=("activations", "preactivations")
            )
            log_probs = dist.log_prob(data.actions)
            ratio = jnp.exp((log_ratio := log_probs - data.log_probs))
            policy_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages,
            ).mean()

            entropy_loss = cfg.entropy_coefficient * dist.entropy().mean()

            # VF
            vf_loss = cfg.vf_coefficient * 0.5 * jnp.power(value_targets - values, 2).mean()
            # vf_loss = jnp.minimum(jnp.power(value_targets - values, 2),
            #                       (value_targets - jnp.clip(values, data.values - 0.2, data.values + 0.2)**2)).mean()

            total_loss = policy_loss + vf_loss + entropy_loss

            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient((jnp.abs(ratio - 1.0) > cfg.clip_eps).mean())

            # Intermediates
            actor_feats = actor_intermediates["activations"]["main"]
            value_feats = value_intermediates["activations"]

            return total_loss, ({
                "metrics/total_loss": total_loss,
                "metrics/policy_loss": policy_loss,
                "metrics/vf_loss": vf_loss,
                "metrics/entropy_loss": entropy_loss,
                "metrics/approx_kl": approx_kl,
                "metrics/clip_fracs": clip_fracs,
                "metrics/values": values.mean(),
            }, actor_feats, value_feats)

        def update_minibatch(carry, xs):
            policy, vf, key = carry
            data, advantages, value_targets = xs

            key, loss_key = jax.random.split(key)
            (_, (metrics, actor_feats, value_feats)), grads = jax.value_and_grad(loss, has_aux=True)(
                (policy.params, vf.params), loss_key, data, advantages, value_targets
            )
            policy_grads, vf_grads = grads[0], grads[1]

            # Update policy
            policy_grads_flat, _ = jax.flatten_util.ravel_pytree(policy_grads)
            policy_grads_hist_dict = pytree_histogram(policy_grads["params"])
            policy = policy.apply_gradients(grads=policy_grads, features=actor_feats)

            policy_params_flat, _ = jax.flatten_util.ravel_pytree(policy.params["params"])
            policy_param_hist_dict = pytree_histogram(policy.params["params"])

            # Updave vf
            vf_grads_flat, _ = jax.flatten_util.ravel_pytree(vf_grads)
            vf_grads_hist_dict = pytree_histogram(vf_grads["params"])
            vf = vf.apply_gradients(grads=vf_grads, features=value_feats)

            vf_params_flat, _ = jax.flatten_util.ravel_pytree(vf.params)
            vf_param_hist_dict = pytree_histogram(vf.params["params"])

            metrics = metrics | {
                "nn/policy_gradient_norm": jnp.linalg.norm(policy_grads_flat),
                "nn/policy_parameter_norm": jnp.linalg.norm(policy_params_flat),
                **prefix_dict("nn/policy_gradients", policy_grads_hist_dict),
                **prefix_dict("nn/policy_parameters", policy_param_hist_dict),
                "nn/vf_gradient_norm": jnp.linalg.norm(vf_grads_flat),
                "nn/vf_parameter_norm": jnp.linalg.norm(vf_params_flat),
                **prefix_dict("nn/vf_gradients", vf_grads_hist_dict),
                **prefix_dict("nn/vf_parameters", vf_param_hist_dict),
            }

            return (policy, vf, key), metrics

        def update_epoch(carry, _):
            policy, vf, key = carry
            key, perm_key = jax.random.split(key)

            def shuffle(x: jax.Array):
                x = x.reshape(-1, *x.shape[2:])
                x = jax.random.permutation(perm_key, x, axis=0)
                x = jnp.reshape(x, (cfg.num_gradient_steps, -1, *x.shape[1:]))
                return x

            (policy, vf, key), metrics = jax.lax.scan(
                update_minibatch,
                (policy, vf, key),
                jax.tree.map(shuffle, (data, advantages, value_targets)),
                length=cfg.num_gradient_steps,
            )

            return (policy, vf, key), metrics

        (policy, vf, key), metrics = jax.lax.scan(
            update_epoch, (policy, vf, key), None, length=cfg.num_epochs
        )

        # Finalize logs
        logs = prefix_dict(
            "data",
            {
                **get_logs("advantages", advantages),
                **get_logs("returns", value_targets),
                **get_logs("values", data.values),
                **get_logs("rewards", data.rewards),
                **get_logs("actions", data.actions),
                "approx_entropy": -data.log_probs.mean(),
            },
        )
        logs["metrics/explained_variance"] = explained_variance(
            data.values.reshape(-1), value_targets.reshape(-1)
        )
        logs.update(accumulate_concatenated_metrics(metrics))

        return key, policy, vf, logs


class TrainerState(NamedTuple):
    policy: TrainState
    vf: TrainState
    key: PRNGKeyArray
    total_steps: int


State = tuple[TrainerState, Observation, EnvState]


class JittedContinualPPOTrainer(PPO):
    policy: TrainState
    vf: TrainState
    key: PRNGKeyArray
    cfg: PPOConfig

    benchmark: JittableContinualLearningEnv
    logger: Logger

    def __init__(
        self,
        seed: int,
        ppo_config: PPOConfig,
        env_cfg: EnvConfig,
        train_cfg: RLTrainingConfig,
        logs_cfg: LoggingConfig,
    ):
        self.key, policy_init_key, vf_init_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        self.cfg = ppo_config
        self.logger = Logger(
            logs_cfg,
            run_config={
                "algorithm": ppo_config,
                "benchmark": env_cfg,
                "training": train_cfg,
            },
        )
        benchmark = get_benchmark(seed, env_cfg)
        if not isinstance(benchmark, JittableContinualLearningEnv):
            raise ValueError(
                "Benchmark must be an end-to-end JAX environment. Use ContinualPPOTrainer otherwise."
            )

        self.benchmark = benchmark
        self.train_cfg = train_cfg

        policy_network_module = get_model_cls(ppo_config.policy_config.network)
        policy_module = Policy(policy_network_module, ppo_config.policy_config)

        self.policy = TrainState.create(
            apply_fn=policy_module.apply,
            params=policy_module.lazy_init(
                policy_init_key,
                self.benchmark.observation_spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=get_optimizer(ppo_config.policy_config.optimizer),
            kernel_init=ppo_config.policy_config.network.kernel_init,
            bias_init=ppo_config.policy_config.network.bias_init,
        )

        vf_module = get_model(ppo_config.vf_config.network)
        self.vf = TrainState.create(
            apply_fn=vf_module.apply,
            params=vf_module.lazy_init(
                vf_init_key,
                self.benchmark.observation_spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=get_optimizer(ppo_config.vf_config.optimizer),
            kernel_init=ppo_config.vf_config.network.kernel_init,
            bias_init=ppo_config.vf_config.network.bias_init,
        )

        self.start_step = self.total_steps = 0
        self.steps_per_task = train_cfg.steps_per_task

    def train(self):
        def rollout(envs: JittableVectorEnv, state: State) -> tuple[State, Rollout]:
            def step(state: State, _) -> tuple[State, Rollout]:
                (policy, vf, key, total_steps), observation, env_states = state

                key, action_key = jax.random.split(key)
                actions, log_probs = policy.apply_fn(
                    policy.params, observation
                ).sample_and_log_prob(seed=action_key)
                values = vf.apply_fn(vf.params, observation).squeeze(-1)
                env_states, data = envs.step(env_states, actions)

                def _print(x: int):
                    if x % 10_000 == 0:
                        print(f"{x}, SPS: {x / (time.time() - start_time)}")

                jax.experimental.io_callback(_print, None, total_steps)

                rollout = Rollout(
                    observations=observation,
                    actions=actions,
                    rewards=data.reward,
                    terminated=data.terminated,
                    truncated=data.truncated,
                    log_probs=log_probs,
                    values=values,
                    next_observations=data.next_observation,
                    infos=data.info,
                )

                return (
                    TrainerState(policy, vf, key, total_steps + self.benchmark.num_envs),
                    data.next_observation,
                    env_states,
                ), rollout

            return jax.lax.scan(
                step,
                state,
                None,
                length=self.cfg.num_rollout_steps // self.benchmark.num_envs,
            )

        @jax.jit
        def train_step(envs: JittableVectorEnv, state: State) -> tuple[State, LogDict, dict]:
            state, data = rollout(envs, state)
            agent_state, observation, env_states = state

            key, policy, vf, logs = self.update(
                agent_state.key,
                agent_state.policy,
                agent_state.vf,
                data,
                next_obs=observation,
                cfg=self.cfg,
            )

            state = (
                TrainerState(policy, vf, key, agent_state.total_steps),
                observation,
                env_states,
            )

            # TODO: eval?

            return state, logs, data.infos

        start_time = time.time()

        for _, envs in enumerate(self.benchmark.tasks):
            env_state, obs = envs.init()

            for _ in range(
                self.total_steps % self.steps_per_task,
                self.steps_per_task,
                self.cfg.num_rollout_steps,
            ):
                state, logs, infos = train_step(
                    envs,
                    (
                        TrainerState(self.policy, self.vf, self.key, self.total_steps),
                        obs,
                        env_state,
                    ),
                )
                (self.policy, self.vf, self.key, self.total_steps), obs, env_state = state

                episode_infos = infos["episode_metrics"]
                dones = infos["episode_done"].astype(bool)
                episode_logs = {
                    "charts/num_episodes": dones.sum(),
                    "charts/mean_episodic_length": infos["steps"][dones].mean(),
                    "charts/mean_episodic_return": episode_infos["sum_reward"][dones].mean(),
                }
                sps = {
                    "charts/SPS": (self.total_steps - self.start_step)
                    / (time.time() - start_time)
                }
                self.logger.log(logs | sps | episode_logs, step=self.total_steps)

        self.logger.close()


if __name__ == "__main__":
    SEED = 44

    start = time.time()
    trainer = JittedContinualPPOTrainer(
        seed=SEED,
        ppo_config=PPOConfig(
            policy_config=PolicyNetworkConfig(
                optimizer=AdamConfig(learning_rate=3e-4),
                network=MLPConfig(
                    num_layers=4,
                    hidden_size=32,
                    output_size=8,
                    activation_fn=Activation.Swish,
                    kernel_init=jax.nn.initializers.lecun_normal(),
                    dtype=jnp.float32,
                ),
                std_type=StdType.MLP_HEAD,
            ),
            vf_config=ValueFunctionConfig(
                optimizer=AdamConfig(learning_rate=3e-4),
                network=MLPConfig(
                    num_layers=5,
                    hidden_size=256,
                    output_size=1,
                    activation_fn=Activation.Swish,
                    kernel_init=jax.nn.initializers.lecun_normal(),
                    dtype=jnp.float32,
                ),
            ),
            num_rollout_steps=2048 * 32 * 5,
            num_epochs=4,
            num_gradient_steps=32,
            gamma=0.97,
            gae_lambda=0.95,
            entropy_coefficient=1e-2,
            clip_eps=0.3,
            vf_coefficient=0.5,
            normalize_advantages=True,
        ),
        env_cfg=EnvConfig("slippery_ant", num_envs=4096, num_tasks=1, episode_length=1000),
        train_cfg=RLTrainingConfig(
            resume=False,
            steps_per_task=150_000_000,
        ),
        logs_cfg=LoggingConfig(
            run_name="continual_ant_debug_12",
            wandb_entity="lucmc",
            wandb_project="crl_experiments",
            save=False,  # Disable checkpoints cause it's so fast anyway
            wandb_mode="online",
        ),
    )

    trainer.train()

