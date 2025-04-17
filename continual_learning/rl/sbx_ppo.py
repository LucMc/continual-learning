import gymnasium as gym
import os
import jax
import jax.random as random
import numpy as np # For converting JAX key to int seed if needed

# SBX is the JAX implementation
from sbx import PPO

# Using SB3 VecEnv components as in the original script
# Note: gymnasium.vector API (SyncVectorEnv/AsyncVectorEnv) might be more idiomatic with JAX
from stable_baselines3.common.env_util import make_vec_env # Can be useful, but we'll manually create for clarity
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Note: Removed unused/redundant wrappers: TimeLimit, OrderEnforcing, PassiveEnvChecker
# LunarLander-v3 already has a TimeLimit.


# Function to create a single environment instance, correctly handling seeding.
# This is the function that DummyVecEnv expects in its list.
def make_single_env(env_id, seed, continuous=True, **kwargs):
    """
    Creates a function that returns an instance of the environment.
    Needed for DummyVecEnv.
    """
    def _init():
        # Create the environment
        env = gym.make(env_id, continuous=continuous, **kwargs)
        # Seed the environment using the reset method, which is standard
        # Note: env.seed() is deprecated, use reset(seed=...)
        # We call reset here to ensure the seed is set initially.
        # The VecEnv might also call reset later.
        env.reset(seed=seed)
        return env
    return _init


def train():
    n_envs = 6
    env_id = "LunarLander-v3"
    continuous_actions = True # Explicitly state if using continuous version

    total_timesteps = 600_000
    learning_rate = 3e-4 # Common default for PPO
    # Keep your custom architecture
    policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
    logdir = "logs/sbx_lunarlander" # Use a more descriptive logdir

    # --- Ensure log directory exists ---
    os.makedirs(logdir, exist_ok=True)

    # --- Naming the run (optional, for organization) ---
    # SBX Tensorboard logs go into `logdir/PPO_X` automatically
    # You can create a specific subfolder if desired
    # run_name = f"PPO_{len(os.listdir(logdir))}"
    # log_path = os.path.join(logdir, run_name)
    # print(f"Tensorboard log path: {log_path}")
    # tensorboard_log=log_path # Pass this path to PPO

    # For simplicity, we'll let sbx create the PPO_X folder inside logdir
    print(f"Tensorboard logs will be saved in subfolders of: {logdir}")


    print("Total timesteps:", total_timesteps)
    print(f"Training {env_id} (Continuous: {continuous_actions}) with {n_envs} environments.")

    # --- Seeding ---
    base_seed = 42
    key = random.PRNGKey(base_seed) # Master JAX key

    # --- Create Environment Functions ---
    env_fns = []
    for i in range(n_envs):
        # Split the key for each environment
        key, subkey = random.split(key)
        # Convert JAX key to a suitable integer seed for Gymnasium env
        # Option 1: Use random integer derived from key
        env_seed = int(random.randint(subkey, (), 0, 2**31 - 1))
        # Option 2: Use sequential seeds (simpler, often sufficient)
        # env_seed = base_seed + i
        print(f"Creating env {i} with seed {env_seed}")
        env_fns.append(
            make_single_env(
                env_id,
                seed=env_seed,
                continuous=continuous_actions
                # Add any other gym.make kwargs here if needed
            )
        )

    # --- Create Vectorized Environment ---
    # Using DummyVecEnv and VecMonitor from SB3 as in the original script
    env = VecMonitor(DummyVecEnv(env_fns))

    # --- Seed the PPO model ---
    key, model_key = random.split(key)
    model_seed = int(random.randint(model_key, (), 0, 2**31 - 1))
    print(f"Seeding SBX PPO model with: {model_seed}")

    # --- Initialize PPO Model ---
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=logdir, # SBX will create PPO_1, PPO_2, etc. inside this
        seed=model_seed,        # Use the derived integer seed for the model
        verbose=1,              # Print training progress
        # Common PPO hyperparameters you might want to tune for sbx:
        # n_steps=1024,         # Number of steps per env per update
        # batch_size=64*n_envs, # Size of minibatches for optimization (often n_steps * n_envs / n_epochs)
        # n_epochs=10,          # Number of optimization epochs per update
        # gamma=0.99,           # Discount factor
        # gae_lambda=0.95,      # Factor for Generalized Advantage Estimation
    )

    # --- Train the Model ---
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10 # Log training stats every 10 updates
        )
    print("Training finished.")

    # --- Save the final model (optional) ---
    # Construct a potential save path based on tensorboard logs
    # This part is heuristic as sbx determines the exact log folder name (PPO_1, PPO_2 etc)
    try:
        # Find the latest PPO_X folder created by sbx in logdir
        run_folders = sorted([d for d in os.listdir(logdir) if d.startswith("PPO_")])
        if run_folders:
            latest_run_folder = os.path.join(logdir, run_folders[-1])
            save_path = os.path.join(latest_run_folder, "final_model.zip")
            model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            print("Could not automatically determine run folder to save model.")
            # Fallback save path
            fallback_path = os.path.join(logdir, "lunarlander_ppo_final.zip")
            model.save(fallback_path)
            print(f"Model saved to {fallback_path}")

    except Exception as e:
        print(f"Error saving model: {e}")

    # --- Close the environment ---
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    # JAX GPU memory preallocation prevention (optional but recommended)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85" # Optional: Limit JAX memory usage
    train()
