import jax
import jax.numpy as jnp
import brax.envs

# 1. Configuration
env_name = "ant"  # Example environment
num_envs = 128    # Number of parallel environments
total_steps = 1000
seed = 0

# 2. Initialize PRNG Key
key = jax.random.PRNGKey(seed)
key_reset, key_actions, key_step = jax.random.split(key, 3)

# 3. Get the environment
# Brax v1 used brax.envs.create(), Brax v2 (current) uses get_environment()
env = brax.envs.get_environment(env_name=env_name)

# 4. JIT compile and vmap core environment functions
# Note: jit before vmap is generally more efficient
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

vmap_env_reset = jax.vmap(jit_env_reset, in_axes=(0,)) # vmap over keys
vmap_env_step = jax.vmap(jit_env_step, in_axes=(0, 0)) # vmap over states and actions

# 5. Initial Reset
# Create a batch of keys for resetting each parallel environment
initial_reset_keys = jax.random.split(key_reset, num_envs)
states = vmap_env_reset(initial_reset_keys)

# To store rewards for analysis (optional)
all_rewards = []
episode_rewards = jnp.zeros(num_envs)
episode_lengths = jnp.zeros(num_envs, dtype=jnp.int32)

print(f"Running {num_envs} parallel '{env_name}' environments for {total_steps} steps.")

# 6. Rollout Loop
for step_idx in range(total_steps // num_envs): # Assuming each step is a parallel step
    # a. Sample actions (replace with your policy)
    key_actions, key_sample = jax.random.split(key_actions)
    # Ensure actions are within the valid range; env.action_size gives the dimension
    actions = jax.random.uniform(
        key_sample,
        shape=(num_envs, env.action_size),
        minval=-1.0,  # Assuming actions are typically in [-1, 1]
        maxval=1.0
    )

    # b. Step the environments
    key_step, key_reset_if_done = jax.random.split(key_step) # Key for potential resets
    next_states = vmap_env_step(states, actions)

    # c. Collect rewards and update episode stats
    current_rewards = next_states.reward
    all_rewards.append(current_rewards)
    episode_rewards += current_rewards
    episode_lengths += 1

    # d. Handle Autoreset for done environments
    dones = next_states.done.astype(jnp.bool_) # Ensure it's boolean for jnp.where

    if jnp.any(dones):
        print(f"Step {step_idx * num_envs}: {jnp.sum(dones)} envs finished.")
        print(f"  Episode rewards for finished envs: {episode_rewards[dones]}")
        print(f"  Episode lengths for finished envs: {episode_lengths[dones]}")

        # Generate reset keys only for environments that are done
        reset_keys_for_done_envs = jax.random.split(key_reset_if_done, jnp.sum(dones))

        # Reset only the 'done' environments
        # Create new states for those that are done
        # Need to map the smaller set of reset_keys_for_done_envs to the full batch dimension
        # This is a bit tricky. A simpler way is to reset all and then pick.
        # More efficient: reset only those that need it.

        # Create a full batch of reset keys, but we'll only use some
        per_env_reset_keys = jax.random.split(key_reset_if_done, num_envs)
        new_states_on_reset = vmap_env_reset(per_env_reset_keys)

        # Use jax.tree_map and jnp.where to selectively reset
        # For each leaf in the PyTree (State), if done, take from new_states_on_reset, else from next_states
        states = jax.tree_map(
            lambda n_state, o_state: jnp.where(dones.reshape(-1, *([1]*(n_state.ndim-1))), n_state, o_state),
            new_states_on_reset,
            next_states
        )
        # Reset episode statistics for done environments
        episode_rewards = jnp.where(dones, 0.0, episode_rewards)
        episode_lengths = jnp.where(dones, 0, episode_lengths)
    else:
        states = next_states

    # Optional: Print progress
    if (step_idx * num_envs) % (total_steps // 10) == 0 and step_idx > 0:
        print(f"Progress: {(step_idx * num_envs)} / {total_steps} steps completed.")

# For analysis
all_rewards_np = jnp.array(all_rewards).block_until_ready() # (num_loops, num_envs)
print(f"\nRollout finished. Shape of all_rewards: {all_rewards_np.shape}")
print(f"Average reward per step per env: {jnp.mean(all_rewards_np)}")
