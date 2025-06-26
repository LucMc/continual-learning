# pyright: reportCallIssue=false
import abc
import os
import time
from collections import deque
from functools import partial
from typing import Deque

import chex
import distrax
import flax.traverse_util
import jax
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
    TrainingConfig,
)
from continual_learning_2.envs import (
    ContinualLearningEnv,
    VectorEnv,
    get_benchmark,
)
from continual_learning_2.models import get_model
from continual_learning_2.optim import get_optimizer
from continual_learning_2.types import Action, LogDict, LogProb, Observation, Rollout, Value
from continual_learning_2.utils.buffers import RolloutBuffer
from continual_learning_2.utils.monitoring import (
    Logger,
    compute_srank,
    get_dormant_neuron_logs,
    get_linearised_neuron_logs,
    prefix_dict,
    pytree_histogram,
)
from continual_learning_2.utils.training import TrainState

os.environ["XLA_FLAGS"] = (
    " --xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)


class ContinualPPOTrainer(abc.ABC):
    policy: TrainState
    vf: TrainState
    key: PRNGKeyArray

    benchmark: ContinualLearningEnv
    logger: Logger

    ckpt_mgr: ocp.CheckpointManager

    def __init__(
        self,
        seed: int,
        ppo_config,
        env_cfg: EnvConfig,
        train_cfg: TrainingConfig,
        logs_cfg: LoggingConfig,
    ):
        self.key, policy_init_key, vf_init_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        self.logger = Logger(
            logs_cfg,
            run_config={
                "algorithm": ppo_config,
                "benchmark": env_cfg,
                "training": train_cfg,
            },
        )
        self.benchmark = get_benchmark(env_cfg)
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

        self.start_step = 0
        self.total_steps = 0
        self.steps_per_task = 0  # TODO: Get from some config
        self.buffer = RolloutBuffer(ppo_config.num_rollout_steps, self.benchmark)
        self.episodic_returns: Deque[float] = deque([], maxlen=20 * self.benchmark.num_envs)
        self.episodic_lengths: Deque[int] = deque([], maxlen=20 * self.benchmark.num_envs)

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
                            get_metric_fn=lambda x: x["metrics/eval_score"],
                            reverse=True,
                        ),
                    ]
                ),
            ),
        )

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
        assert self.total_steps == 0, "Load was called before training started"

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
        self.start_step = self.total_steps = step + 1

    def train(self):
        if self.train_cfg.resume:
            self.load(step=self.train_cfg.resume_from_step)

        start_time = time.time()

        for _, envs in enumerate(self.benchmark.tasks):
            obs, episode_started = envs.init()

            for _ in range(
                self.total_steps % self.steps_per_task, self.steps_per_task, step=envs.num_envs
            ):
                actions, log_probs, values = self.sample(obs)
                timestep = envs.step(actions)

                self.buffer.add(
                    obs=obs,
                    action=actions,
                    reward=timestep.reward,
                    episode_start=episode_started,
                    value=values,
                    log_prob=log_probs,
                )

                episode_started = np.logical_or(timestep.terminated, timestep.truncated)
                obs = timestep.next_observation

                episode_lengths = timestep.final_episode_lengths[episode_started]
                episode_returns = timestep.final_episode_returns[episode_started]
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
                    logs = self.update(
                        rollouts,
                        dones=timestep.terminated,
                        next_obs=np.where(
                            episode_started[:, None],
                            timestep.final_observation,
                            timestep.next_observation,
                        ),
                    )
                    self.logger.log(logs, step=self.total_steps)
                    self.buffer.reset()

                metrics = None
                if self.total_steps % self.logger.cfg.eval_interval == 0:
                    # TODO: Agent interface
                    metrics = self.benchmark.evaluate(
                        self.policy, forgetting=self.logger.cfg.catastrophic_forgetting
                    )

                self.save(envs, metrics)

                self.total_steps += envs.num_envs

    @staticmethod
    @jax.jit
    def _sample_action_value_and_log_prob(
        key: PRNGKeyArray, policy: TrainState, vf: TrainState, observation: Observation
    ) -> tuple[PRNGKeyArray, Action, LogProb, Value]:
        dist: distrax.Distribution
        key, action_key, p_dropout, vf_dropout = jax.random.split(key)
        dist = policy.apply_fn(
            policy.params, observation, training=True, rngs={"dropout": p_dropout}
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        value = vf.apply_fn(
            vf.params, observation, training=True, rngs={"dropout": vf_dropout}
        )
        return key, action, log_prob, value  # pyright: ignore[reportReturnType]

    def sample(self, observation: Observation) -> tuple[Action, LogProb, Value]:
        self.key, action, log_prob, value = self._sample_action_value_and_log_prob(
            self.key, self.policy, self.vf, observation
        )
        return action, log_prob, value

    @staticmethod
    @partial(
        jax.jit,
        donate_argnames=("policy", "key"),
        static_argnames=("clip_eps", "normalize_advantages", "entropy_coefficient"),
    )
    def update_policy(
        policy: TrainState,
        data: Rollout,
        key: PRNGKeyArray,
        clip_eps: float,
        normalize_advantages: bool,
        entropy_coefficient: float,
    ) -> tuple[TrainState, PRNGKeyArray, LogDict]:
        assert data.advantages is not None
        key, dropout_key = jax.random.split(key)

        if normalize_advantages:
            advantages = (data.advantages - data.advantages.mean(axis=0, keepdims=True)) / (
                data.advantages.std(axis=0, keepdims=True) + 1e-8
            )
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
            clip_fracs = jax.lax.stop_gradient((jnp.abs(ratio - 1.0) > clip_eps).mean())

            pg_loss1 = -advantages * ratio  # pyright: ignore[reportOptionalOperand]
            pg_loss2 = -advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
                ratio, 1 - clip_eps, 1 + clip_eps
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            entropy_loss = action_dist.entropy().mean()

            return pg_loss - entropy_coefficient * entropy_loss, (
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

    @staticmethod
    @partial(
        jax.jit,
        donate_argnames=("vf", "key"),
        static_argnames=("clip_vf_loss", "clip_eps", "vf_coefficient"),
    )
    def update_value_function(
        vf: TrainState,
        data: Rollout,
        key: PRNGKeyArray,
        clip_vf_loss: bool,
        clip_eps: float,
        vf_coefficient: float,
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

            if clip_vf_loss:
                vf_loss_unclipped = (new_values - data.returns) ** 2
                v_clipped = data.values + jnp.clip(
                    new_values - data.values, -clip_eps, clip_eps
                )
                vf_loss_clipped = (v_clipped - data.returns) ** 2
                vf_loss = 0.5 * jnp.maximum(vf_loss_unclipped, vf_loss_clipped).mean()
            else:
                vf_loss = 0.5 * ((new_values - data.returns) ** 2).mean()

            return vf_coefficient * vf_loss, (
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
