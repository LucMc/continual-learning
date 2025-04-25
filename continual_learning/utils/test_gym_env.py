import gymnasium as gym
from functools import partial

# from cont_rlrd.envs import RandomDelayEnv, ContinualIntervalRandomDelayEnv
from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env

from continual_learning.utils.wrappers_rd import ContinualRandomIntervalDelayWrapper
# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

env = ContinualRandomIntervalDelayWrapper(gym.make("HalfCheetah-v5"),
    obs_delay_range=range(0, 4),
    act_delay_range=range(0, 4),
    # mode="continual",
    interval_emb_type="two_hot",
    delay_emb_type="one_hot", # NOT YET SUPPORTED
    output="standard" # dcac/standard
)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
# model.save("ppo_cont")

# del model # remove to demonstrate saving and loading
#
# model = PPO.load("ppo_cont")

obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, terms, truns, info = env.step(action)
    done = terms | truns
    print(rewards)

