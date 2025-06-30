# pyright: reportCallIssue=false
import os
import time
from collections import deque
from functools import partial
from typing import Deque, NamedTuple

import chex
import distrax
import flax.traverse_util
import jax
import jax.experimental
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.checkpoint_managers as ocp_mgrs
from flax.core import DenyList, FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray

from continual_learning_2.configs import (
    EnvConfig,
    LoggingConfig,
)
from continual_learning_2.configs.rl import PPOConfig
from continual_learning_2.configs.training import RLTrainingConfig
from continual_learning_2.envs import (
    ContinualLearningEnv,
    VectorEnv,
    get_benchmark,
)
from continual_learning_2.envs.base import JittableContinualLearningEnv, Timestep
from continual_learning_2.models import get_model
from continual_learning_2.optim import get_optimizer
from continual_learning_2.types import (
    Action,
    EnvState,
    LogDict,
    LogProb,
    Observation,
    Rollout,
    Value,
)
from continual_learning_2.utils.buffers import RolloutBuffer, compute_gae_scan
from continual_learning_2.utils.monitoring import (
    Logger,
    accumulate_concatenated_metrics,
    compute_srank,
    explained_variance,
    get_dormant_neuron_logs,
    get_last_metrics,
    get_linearised_neuron_logs,
    get_logs,
    prefix_dict,
    pytree_histogram,
)
from continual_learning_2.utils.training import TrainState

os.environ["XLA_FLAGS"] = (
    " --xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)


class PPO:
    @staticmethod
    @jax.jit
    def _sample_action_value_and_log_prob(
        key: PRNGKeyArray, policy: TrainState, vf: TrainState, observation: Observation
    ) -> tuple[PRNGKeyArray, Action, LogProb, Value]:
        dist: distrax.Distribution
        key, action_key, p_dropout, vf_dropout = jax.random.split(key, 4)
        dist = policy.apply_fn(
            policy.params, observation, training=True, rngs={"dropout": p_dropout}
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        value = vf.apply_fn(
            vf.params, observation, training=True, rngs={"dropout": vf_dropout}
        )
        return key, action, log_prob, value  # pyright: ignore[reportReturnType]

    @staticmethod
    @jax.jit
    def _sample_action_deterministic(
        policy: TrainState, observation: Observation
    ) -> tuple[PRNGKeyArray, Action]:
        dist: distrax.Distribution
        dist = policy.apply_fn(policy.params, observation, training=False)
        return dist.mode()

    @staticmethod
    @jax.jit
    def _get_value(
        vf: TrainState, key: PRNGKeyArray, observation: Observation
    ) -> tuple[PRNGKeyArray, Value]:
        key, vf_dropout = jax.random.split(key)
        value = vf.apply_fn(
            vf.params, observation, training=True, rngs={"dropout": vf_dropout}
        )
        return key, value

    @staticmethod
    @partial(jax.jit, static_argnames=("cfg"), donate_argnames=("policy", "vf", "key"))
    def update(
        key: PRNGKeyArray,
        policy: TrainState,
        vf: TrainState,
        data: Rollout,
        next_obs: Float[Observation, " ..."],
        cfg: PPOConfig,
    ) -> tuple[PRNGKeyArray, TrainState, TrainState, LogDict]:
        key, last_values_key = jax.random.split(key)
        last_values = vf.apply_fn(
            vf.params, next_obs, training=True, rngs={"dropout": last_values_key}
        )
        data = compute_gae_scan(data, last_values, cfg.gamma, cfg.gae_lambda)

        assert data.advantages is not None and data.returns is not None
        assert data.values is not None and data.log_probs is not None
        diagnostic_logs = prefix_dict(
            "data",
            {
                **get_logs("advantages", data.advantages),
                **get_logs("returns", data.returns),
                **get_logs("values", data.values),
                **get_logs("rewards", data.rewards),
                **get_logs("num_episodes", data.dones.sum(axis=1), hist=False, std=False),
                "approx_entropy": -data.log_probs.mean(),
            },
        )

        def update_policy(
            policy: TrainState,
            data: Rollout,
            key: PRNGKeyArray,
            cfg: PPOConfig,
        ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
            assert data.advantages is not None
            key, dropout_key = jax.random.split(key)

            if cfg.normalize_advantages:
                advantages = (
                    data.advantages - data.advantages.mean(axis=0, keepdims=True)
                ) / (data.advantages.std(axis=0, keepdims=True) + 1e-8)
            else:
                advantages = data.advantages

            def policy_loss(params: FrozenDict):
                action_dist: distrax.Distribution
                new_log_probs: Float[Array, " *batch"]
                assert data.log_probs is not None

                action_dist, intermediates = policy.apply_fn(
                    params,
                    data.observations,
                    training=True,
                    rngs={"dropout": dropout_key},
                    mutable=("activations", "preactivations"),
                )
                new_log_probs = action_dist.log_prob(data.actions)  # pyright: ignore[reportAssignmentType]
                log_ratio = new_log_probs.reshape(data.log_probs.shape) - data.log_probs
                ratio = jnp.exp(log_ratio)

                # For logs
                approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
                clip_fracs = jax.lax.stop_gradient(
                    (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean()
                )

                pg_loss1 = -advantages * ratio  # pyright: ignore[reportOptionalOperand]
                pg_loss2 = -advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
                    ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                )
                pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                entropy_loss = action_dist.entropy().mean()

                return pg_loss - cfg.entropy_coefficient * entropy_loss, (
                    {
                        "metrics/entropy_loss": entropy_loss,
                        "metrics/policy_loss": pg_loss,
                        "metrics/approx_kl": approx_kl,
                        "metrics/clip_fracs": clip_fracs,
                    },
                    intermediates,
                )

            (_, (logs, intermediates)), policy_grads = jax.value_and_grad(
                policy_loss, has_aux=True
            )(policy.params)
            policy_grads_flat, _ = jax.flatten_util.ravel_pytree(policy_grads)
            grads_hist_dict = pytree_histogram(policy_grads["params"])

            activations = intermediates["activations"]
            activations_hist_dict = pytree_histogram(activations)
            activations_flat = {
                k: v[0]  # pyright: ignore[reportIndexIssue]
                for k, v in flax.traverse_util.flatten_dict(activations, sep="/").items()
            }
            dormant_neuron_logs = get_dormant_neuron_logs(activations_flat)  # pyright: ignore[reportArgumentType]
            srank_logs = jax.tree.map(compute_srank, activations_flat)

            preactivations = intermediates["preactivations"]
            preactivations_flat = {
                k: v[0]  # pyright: ignore[reportIndexIssue]
                for k, v in flax.traverse_util.flatten_dict(preactivations, sep="/").items()
            }
            linearised_neuron_logs = get_linearised_neuron_logs(preactivations_flat)  # pyright: ignore[reportArgumentType]

            policy = policy.apply_gradients(grads=policy_grads)
            policy_params_flat, _ = jax.flatten_util.ravel_pytree(policy.params["params"])
            param_hist_dict = pytree_histogram(policy.params["params"])

            return (
                policy,
                key,
                logs
                | {
                    "nn/policy_gradient_norm": jnp.linalg.norm(policy_grads_flat),
                    "nn/policy_parameter_norm": jnp.linalg.norm(policy_params_flat),
                    **prefix_dict("nn/policy_gradients", grads_hist_dict),
                    **prefix_dict("nn/policy_parameters", param_hist_dict),
                    **prefix_dict("nn/policy_activations", activations_hist_dict),
                    **prefix_dict("nn/policy_dormant_neurons", dormant_neuron_logs),
                    **prefix_dict("nn/policy_linearised_neurons", linearised_neuron_logs),
                    **prefix_dict("nn/policy_srank", srank_logs),
                    **prefix_dict("nn/policy_gradients", grads_hist_dict),
                },
            )

        def update_value_function(
            vf: TrainState,
            data: Rollout,
            key: PRNGKeyArray,
            cfg: PPOConfig,
        ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
            key, dropout_key = jax.random.split(key)

            def value_function_loss(params: FrozenDict):
                new_values: Float[Array, "*batch 1"]
                new_values, intermediates = vf.apply_fn(
                    params,
                    data.observations,
                    training=True,
                    rngs={"dropout": dropout_key},
                    mutable=["preactivations", "activations"],
                )

                chex.assert_equal_shape((new_values, data.returns))
                assert data.values is not None and data.returns is not None

                vf_loss = 0.5 * ((new_values - data.returns) ** 2).mean()

                return cfg.vf_coefficient * vf_loss, (
                    {
                        "metrics/vf_loss": vf_loss,
                        "metrics/values": new_values.mean(),
                    },
                    intermediates,
                )

            (_, (logs, intermediates)), vf_grads = jax.value_and_grad(
                value_function_loss, has_aux=True
            )(vf.params, data)
            vf_grads_flat, _ = jax.flatten_util.ravel_pytree(vf_grads)
            grads_hist_dict = pytree_histogram(vf_grads["params"])

            activations = intermediates["activations"]
            activations_hist_dict = pytree_histogram(activations)
            activations_flat = {
                k: v[0]  # pyright: ignore[reportIndexIssue]
                for k, v in flax.traverse_util.flatten_dict(activations, sep="/").items()
            }
            dormant_neuron_logs = get_dormant_neuron_logs(activations_flat)  # pyright: ignore[reportArgumentType]
            srank_logs = jax.tree.map(compute_srank, activations_flat)

            preactivations = intermediates["preactivations"]
            preactivations_flat = {
                k: v[0]  # pyright: ignore[reportIndexIssue]
                for k, v in flax.traverse_util.flatten_dict(preactivations, sep="/").items()
            }
            linearised_neuron_logs = get_linearised_neuron_logs(preactivations_flat)  # pyright: ignore[reportArgumentType]

            vf = vf.apply_gradients(grads=vf_grads)
            vf_params_flat, _ = jax.flatten_util.ravel_pytree(vf.params)
            param_hist_dict = pytree_histogram(vf.params["params"])

            return (
                vf,
                key,
                logs
                | {
                    "nn/vf_gradient_norm": jnp.linalg.norm(vf_grads_flat),
                    "nn/vf_parameter_norm": jnp.linalg.norm(vf_params_flat),
                    **prefix_dict("nn/vf_gradients", grads_hist_dict),
                    **prefix_dict("nn/vf_parameters", param_hist_dict),
                    **prefix_dict("nn/vf_activations", activations_hist_dict),
                    **prefix_dict("nn/vf_dormant_neurons", dormant_neuron_logs),
                    **prefix_dict("nn/vf_linearised_neurons", linearised_neuron_logs),
                    **prefix_dict("nn/vf_srank", srank_logs),
                },
            )

        def train_minibatch(carry, minibatch: Rollout):
            key, policy, vf = carry
            policy, key, policy_logs = update_policy(policy, minibatch, key, cfg)
            vf, key, vf_logs = update_value_function(vf, minibatch, key, cfg)
            return (policy, vf, key), (policy_logs | vf_logs)

        def train_epoch(carry, _):
            policy, vf, key, data = carry

            key, permutation_key = jax.random.split(key)
            rollout_size = data.observations.shape[0] * data.observations.shape[1]

            permutation = jax.random.permutation(permutation_key, rollout_size)
            data = jax.tree.map(lambda x: x.reshape((rollout_size, *x.shape[2:])), data)
            shuffled_data = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), data)

            minibatches = jax.tree.map(
                lambda x: x.reshape(cfg.num_gradient_steps, -1, *x.shape[1:]),
                shuffled_data,
            )

            (policy, vf, key), logs = jax.lax.scan(
                train_minibatch, (policy, vf, key), minibatches
            )
            return (policy, vf, key, data), logs

        (policy, vf, key, _), logs = jax.lax.scan(
            train_epoch,
            (policy, vf, key, data),
            None,
            length=cfg.num_epochs,
        )

        # Finalize logs
        final_logs = {}
        final_logs["metrics/explained_variance"] = explained_variance(
            data.values.reshape(-1), data.returns.reshape(-1)
        )
        final_logs.update(accumulate_concatenated_metrics(logs))

        return key, policy, vf, diagnostic_logs | final_logs


class RLCheckpoints:
    ckpt_mgr: ocp.CheckpointManager

    def __init__(self, seed: int, logs_cfg: LoggingConfig):
        self.ckpt_mgr = ocp.CheckpointManager(
            logs_cfg.checkpoint_dir / f"{logs_cfg.run_name}_{seed}",
            options=ocp.CheckpointManagerOptions(
                save_decision_policy=ocp_mgrs.AnySavePolicy(
                    [
                        ocp_mgrs.FixedIntervalPolicy(logs_cfg.save_interval),
                        ocp_mgrs.save_decision_policy.PreemptionCheckpointingPolicy(),
                    ]
                ),
                preservation_policy=ocp_mgrs.AnyPreservationPolicy(
                    [
                        ocp_mgrs.LatestN(n=1),
                        ocp_mgrs.EveryNSeconds(60 * 60),  # Hourly checkpoints
                        ocp_mgrs.BestN(  # Top 3
                            n=3,
                            get_metric_fn=lambda x: x[logs_cfg.best_metric],
                            reverse=True,
                        ),
                    ]
                ),
            ),
        )


class ContinualPPOTrainer(PPO, RLCheckpoints):
    policy: TrainState
    vf: TrainState
    key: PRNGKeyArray
    cfg: PPOConfig

    benchmark: ContinualLearningEnv
    logger: Logger

    def __init__(
        self,
        seed: int,
        ppo_config: PPOConfig,
        env_cfg: EnvConfig,
        train_cfg: RLTrainingConfig,
        logs_cfg: LoggingConfig,
    ):
        RLCheckpoints.__init__(self, seed, logs_cfg)

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
        benchmark = get_benchmark(env_cfg)
        if not isinstance(benchmark, ContinualLearningEnv):
            raise ValueError(
                "Benchmark must not be an end-to-end JAX environment. Use JittableContinualPPOTrainer for that."
            )

        self.benchmark = benchmark
        self.train_cfg = train_cfg

        policy_module = get_model(ppo_config.policy_config)
        self.policy = TrainState.create(
            apply_fn=jax.jit(policy_module.apply, static_argnames=("training", "mutable")),
            params=policy_module.lazy_init(
                policy_init_key,
                self.benchmark.observation_spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=get_optimizer(ppo_config.policy_config.optim_config),
            kernel_init=ppo_config.policy_config.kernel_init,
            bias_init=ppo_config.policy_config.bias_init,
        )

        vf_module = get_model(ppo_config.vf_config)
        self.value_function = TrainState.create(
            apply_fn=jax.jit(vf_module.apply, static_argnames=("training", "mutable")),
            params=vf_module.lazy_init(
                vf_init_key,
                self.benchmark.observation_spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=get_optimizer(ppo_config.vf_config.optim_config),
            kernel_init=ppo_config.vf_config.kernel_init,
            bias_init=ppo_config.vf_config.bias_init,
        )

        self.start_step = self.total_steps = 0
        self.steps_per_task = train_cfg.num_steps // env_cfg.num_tasks
        self.buffer = RolloutBuffer(ppo_config.num_rollout_steps, self.benchmark)
        self.episodic_returns: Deque[float] = deque([], maxlen=20 * self.benchmark.num_envs)
        self.episodic_lengths: Deque[int] = deque([], maxlen=20 * self.benchmark.num_envs)

    def save(self, envs: VectorEnv, metrics: dict[str, float] | None):
        self.ckpt_mgr.save(
            self.total_steps,
            args=ocp.args.Composite(
                policy=ocp.args.StandardSave(self.policy),
                vf=ocp.args.StandardSave(self.vf),
                key=ocp.args.JaxRandomKeySave(self.key),
                buffer=ocp.args.PyTreeSave(self.buffer.save()),
                envs=ocp.args.JsonSave(envs.save()),
                benchmark=ocp.args.JsonSave(self.benchmark.save()),
            ),
            metrics=metrics,
        )

    def load(self, step: int):
        if step == -1:
            latest_step = self.ckpt_mgr.latest_step()
            assert latest_step is not None, "No checkpoint found"
            step = latest_step

        ckpt = self.ckpt_mgr.restore(
            self.ckpt_mgr.latest_step() if step == -1 else step,
            args=ocp.args.Composite(
                policy=ocp.args.StandardRestore(self.policy),
                vf=ocp.args.StandardRestore(self.vf),
                key=ocp.args.JaxRandomKeyRestore(self.key),
                buffer=ocp.args.PyTreeRestore(self.buffer.save()),
                envs=ocp.args.JsonRestore(),
                benchmark=ocp.args.JsonRestore(),
            ),
        )
        self.policy = ckpt["policy"]
        self.vf = ckpt["vf"]
        self.key = ckpt["key"]
        self.benchmark.load(ckpt["benchmark"], ckpt["envs"])
        self.buffer.load(ckpt["buffer"])
        self.start_step = self.total_steps = step

    def train(self):
        if self.train_cfg.resume:
            self.load(step=self.train_cfg.resume_from_step)

        start_time = time.time()

        for _, envs in enumerate(self.benchmark.tasks):
            obs = envs.init()

            for _ in range(
                self.total_steps % self.steps_per_task,
                self.steps_per_task,
                step=self.benchmark.num_envs,
            ):
                actions, log_probs, values = self.sample(obs)
                timestep = envs.step(actions)

                done = np.logical_or(timestep.terminated, timestep.truncated)

                self.buffer.add(
                    obs=obs,
                    action=actions,
                    reward=timestep.reward,
                    done=done,
                    value=values,
                    log_prob=log_probs,
                )

                obs = timestep.next_observation

                episode_lengths = timestep.final_episode_lengths[done]
                episode_returns = timestep.final_episode_returns[done]
                for _l, _r in zip(episode_lengths, episode_returns):
                    self.episodic_lengths.append(_l)
                    self.episodic_returns.append(_r)

                if self.total_steps % self.logger.cfg.interval == 0 and self.episodic_returns:
                    logs = {
                        "charts/mean_episodic_return": np.mean(
                            list(self.episodic_returns)
                        ).item(),
                        "charts/mean_episodic_length": np.mean(
                            list(self.episodic_lengths)
                        ).item(),
                        "charts/SPS": (self.total_steps - self.start_step)
                        / (time.time() - start_time),
                    }
                    self.logger.log(logs, step=self.total_steps)

                if self.buffer.ready:
                    rollouts = self.buffer.get()
                    self.key, self.policy, self.vf, logs = self.update(
                        rollouts,
                        dones=timestep.terminated,
                        next_obs=np.where(
                            done[:, None],
                            timestep.final_observation,
                            timestep.next_observation,
                        ),
                    )
                    self.logger.log(logs, step=self.total_steps)
                    self.buffer.reset()

                metrics = None
                if self.total_steps % self.logger.cfg.eval_interval == 0:
                    metrics = self.benchmark.evaluate(
                        self, forgetting=self.logger.cfg.catastrophic_forgetting
                    )

                self.save(envs, metrics)

                self.total_steps += self.benchmark.num_envs

    def sample(self, observation: Observation) -> tuple[Action, LogProb, Value]:
        self.key, action, log_prob, value = self._sample_action_value_and_log_prob(
            self.key, self.policy, self.vf, observation
        )
        return jax.device_get((action, log_prob, value))

    def eval_action(self, observation: Observation) -> Action:
        return jax.device_get(self._sample_action_deterministic(self.policy, observation))


class TrainerState(NamedTuple):
    policy: TrainState
    vf: TrainState
    key: PRNGKeyArray
    total_steps: int


State = tuple[TrainerState, Observation, EnvState]


class JittedContinualPPOTrainer(PPO, RLCheckpoints):
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
        RLCheckpoints.__init__(self, seed, logs_cfg)

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
        benchmark = get_benchmark(env_cfg)
        if not isinstance(benchmark, JittableContinualLearningEnv):
            raise ValueError(
                "Benchmark must be an end-to-end JAX environment. Use ContinualPPOTrainer otherwise."
            )

        self.benchmark = benchmark
        self.train_cfg = train_cfg

        policy_module = get_model(ppo_config.policy_config)
        self.policy = TrainState.create(
            apply_fn=jax.jit(policy_module.apply, static_argnames=("training", "mutable")),
            params=policy_module.lazy_init(
                policy_init_key,
                self.benchmark.observation_spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=get_optimizer(ppo_config.policy_config.optim_config),
            kernel_init=ppo_config.policy_config.kernel_init,
            bias_init=ppo_config.policy_config.bias_init,
        )

        vf_module = get_model(ppo_config.vf_config)
        self.value_function = TrainState.create(
            apply_fn=jax.jit(vf_module.apply, static_argnames=("training", "mutable")),
            params=vf_module.lazy_init(
                vf_init_key,
                self.benchmark.observation_spec,
                training=False,
                mutable=DenyList(["activations", "preactivations"]),
            ),
            tx=get_optimizer(ppo_config.vf_config.optim_config),
            kernel_init=ppo_config.vf_config.kernel_init,
            bias_init=ppo_config.vf_config.bias_init,
        )

        self.start_step = self.total_steps = 0
        self.steps_per_task = train_cfg.num_steps // env_cfg.num_tasks

    def save(self, experiment_state: State, metrics: dict[str, float] | None):
        agent_state, _, env_state = experiment_state
        self.ckpt_mgr.save(
            self.total_steps,
            args=ocp.args.Composite(
                policy=ocp.args.StandardSave(agent_state.policy),
                vf=ocp.args.StandardSave(agent_state.vf),
                key=ocp.args.JaxRandomKeySave(agent_state.key),
                benchmark=ocp.args.PyTreeSave(self.benchmark.save(env_state)),
            ),
            metrics=metrics,
        )

    def load(self, step: int):
        if step == -1:
            latest_step = self.ckpt_mgr.latest_step()
            assert latest_step is not None, "No checkpoint found"
            step = latest_step

        # Get dummy env state so we can use PyTreeRestore
        task = next(self.benchmark.tasks)
        dummy_state = task.init()

        ckpt = self.ckpt_mgr.restore(
            self.ckpt_mgr.latest_step() if step == -1 else step,
            args=ocp.args.Composite(
                policy=ocp.args.StandardRestore(self.policy),
                vf=ocp.args.StandardRestore(self.vf),
                key=ocp.args.JaxRandomKeyRestore(self.key),
                benchmark=ocp.args.PyTreeRestore(self.benchmark.save(dummy_state)),
            ),
        )
        self.policy = ckpt["policy"]
        self.vf = ckpt["vf"]
        self.key = ckpt["key"]
        self.benchmark.load(ckpt["benchmark"])
        self.start_step = self.total_steps = step

    def train(self):
        if self.train_cfg.resume:
            self.load(step=self.train_cfg.resume_from_step)

        start_time = time.time()

        for _, envs in enumerate(self.benchmark.tasks):
            env_state, obs = envs.init()

            def rollout(state: State) -> tuple[State, Rollout]:
                def step(state: State, _) -> tuple[State, Rollout]:
                    (policy, vf, key, total_steps), observation, env_states = state
                    key, actions, log_probs, values = self._sample_action_value_and_log_prob(
                        key, policy, vf, observation
                    )
                    env_states, data = envs.step(env_states, actions)

                    rollout = Rollout(
                        observations=observation,
                        actions=actions,
                        rewards=data.reward,
                        dones=jnp.logical_or(data.terminated, data.truncated),
                        log_probs=log_probs,
                        values=values,
                        final_episode_lenghts=data.final_episode_lengths,
                        final_episode_returns=data.final_episode_returns,
                        final_observations=data.final_observation,
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
                    length=self.cfg.num_rollout_steps,
                )

            @jax.jit
            def train_step(state: State, _) -> tuple[State, LogDict]:
                state, data = rollout(state)
                agent_state, observation, env_states = state

                assert data.final_observations is not None
                next_obs = jnp.where(data.dones[-1], data.final_observations[-1], observation)
                key, policy, vf, logs = self.update(
                    agent_state.key,
                    agent_state.policy,
                    agent_state.vf,
                    data,
                    next_obs=next_obs,
                    cfg=self.cfg,
                )

                state = (
                    TrainerState(policy, vf, key, agent_state.total_steps),
                    observation,
                    env_states,
                )

                # Logging
                def logging_cb(state: State, logs: LogDict) -> None:
                    self.total_steps += self.benchmark.num_envs * self.cfg.num_rollout_steps
                    logs = logs | {
                        "charts/SPS": (self.total_steps - self.start_step)
                        / (time.time() - start_time)
                    }
                    self.logger.log(logs, step=self.total_steps)
                    self.save(state)

                assert data.final_episode_lenghts is not None
                assert data.final_episode_returns is not None
                running_logs = logs | {
                    "charts/mean_episodic_length": jnp.mean(data.final_episode_lenghts),
                    "charts/mean_episodic_returns": jnp.mean(data.final_episode_returns),
                }
                jax.experimental.io_callback(logging_cb, None, state, running_logs)

                # TODO: eval?

                return state, logs

            ((self.policy, self.vf, self.key, self.total_steps), _, _), logs = jax.lax.scan(
                train_step,
                (
                    TrainerState(self.policy, self.vf, self.key, self.total_steps),
                    obs,
                    env_state,
                ),
                None,
                length=(self.total_steps % self.steps_per_task)
                // (self.cfg.num_rollout_steps * self.benchmark.num_envs),
            )

            self.logger.log(get_last_metrics(logs))
