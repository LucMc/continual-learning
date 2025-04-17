import pytest
import sbx.ppo

"""
* Test rollout collection loop
* Test GAE calculation
* Test critic loss
* Test Actor loss
"""
@staticmethod
@partial(jax.jit, static_argnames=["normalize_advantage"])
def _one_update(
    actor_state: TrainState,
    vf_state: TrainState,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    old_log_prob: np.ndarray,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    normalize_advantage: bool = True,
):
    # Normalize advantage
    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    if normalize_advantage and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def actor_loss(params):
        dist = actor_state.apply_fn(params, observations)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # ratio between old and new policy, should be one at the first iteration
        ratio = jnp.exp(log_prob - old_log_prob)
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

        # Entropy loss favor exploration
        # Approximate entropy when no analytical form
        # entropy_loss = -jnp.mean(-log_prob)
        # analytical form
        entropy_loss = -jnp.mean(entropy)

        total_policy_loss = policy_loss + ent_coef * entropy_loss
        return total_policy_loss

    pg_loss_value, grads = jax.value_and_grad(actor_loss, has_aux=False)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)

    def critic_loss(params):
        # Value loss using the TD(gae_lambda) target
        vf_values = vf_state.apply_fn(params, observations).flatten()
        return ((returns - vf_values) ** 2).mean()

    vf_loss_value, grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
    vf_state = vf_state.apply_gradients(grads=grads)

    # loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
    return (actor_state, vf_state), (pg_loss_value, vf_loss_value)

  
def to_minibatch_iterator(
    data: Rollout, num: int, seed: int
) -> Generator[Rollout, None, Never]:
    # Flatten batch dims
    rollouts = Rollout(
        *map(
            lambda x: x.reshape(-1, x.shape[-1]) if x is not None else None,
            data,
        )  # pyright: ignore[reportArgumentType]
    )

    rollout_size = rollouts.observations.shape[0]
    minibatch_size = rollout_size // num

    rng = np.random.default_rng(seed)
    rng_state = rng.bit_generator.state

    while True:
        for field in rollouts:
            rng.bit_generator.state = rng_state
            if field is not None:
                rng.shuffle(field)
        rng_state = rng.bit_generator.state
        for start in range(0, rollout_size, minibatch_size):
            end = start + minibatch_size
            yield Rollout(
                *map(
                    lambda x: x[start:end] if x is not None else None,  # pyright: ignore[reportArgumentType]
                    rollouts,
                )
            )

def test_rollout():
    pass
