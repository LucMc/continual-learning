import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from continual_learning_2.types import Rollout, Value


@jax.jit
def compute_gae_scan(
    rollout: Rollout,
    values: Value,
    last_values: Float[Array, "... 1"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Value, Value]:
    """Adapted from https://github.com/google/brax/blob/main/brax/training/agents/ppo/losses.py#L38

    NOTE: This GAE implementation essentially ignores truncated timesteps and takes them out of learning entirely.
    This is an efficient way of circumventing the "poisoning" of the learning process from bad value target estimates at truncated timesteps.
    However, this is sample inefficient, and ideally we would get the value of the final observation and add it to the target for truncated timesteps. But, that would require extra neural network forward passes, which is hardware inefficient. So that's the trade-off.
    """
    truncation_mask = 1 - rollout.truncated
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate([values[1:], last_values[None, ...]], axis=0)
    deltas = rollout.rewards + gamma * (1 - rollout.terminated) * values_t_plus_1 - values
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
        (truncation_mask, deltas, rollout.terminated),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], last_values[None, ...]], axis=0)
    advantages = (
        rollout.rewards + gamma * (1 - rollout.terminated) * vs_t_plus_1 - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)
