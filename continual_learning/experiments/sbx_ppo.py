import gymnasium as gym

from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from continual_learning.envs.slippery_ant_v5 import ContinualAnt, SlipperyAntEnv
from functools import partial

def compare_friction():
    ## Training
    configured_env = partial(SlipperyAntEnv, friction=1)
    env = VecMonitor(DummyVecEnv([configured_env for _ in range(4)])) # gym.make("Ant-v5", render_mode="human")

    model = PPO("MlpPolicy", env, tensorboard_log="sbx_logs/f_1")
    model.learn(total_timesteps=1_000_000, progress_bar=True)

## Testing in same friction
    configured_env = partial(SlipperyAntEnv, friction=5, render_mode="human")
    vec_env = DummyVecEnv([configured_env for _ in range(1)]) # gym.make("Ant-v5", render_mode="human")
    obs = vec_env.reset()

    print("Testing in high friction")
    for _ in range(1000):
        vec_env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

    vec_env.close()

## Testing in lower friction
    print("Testing in low friction")
    configured_env = partial(SlipperyAntEnv, friction=0.1, render_mode="human")
    vec_env = DummyVecEnv([configured_env for _ in range(1)]) # gym.make("Ant-v5", render_mode="human")
    obs = vec_env.reset()

    for _ in range(1000):
        vec_env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

    vec_env.close()

def train_continual_ant():
    total_time_steps = 5_000_000
    change_friction_every = 500_000
    n_envs = 4
    max_friction = 1.5
    min_friction = 0.75

    configured_env = partial(ContinualAnt,
                             change_friction_every=change_friction_every//n_envs,
                             max_friction=max_friction,
                             min_friction=min_friction)

    env = VecMonitor(DummyVecEnv([configured_env for _ in range(n_envs)])) 

    model = PPO("MlpPolicy", env, tensorboard_log="./sbx_logs")
    model.learn(total_timesteps=total_time_steps, progress_bar=True)


if __name__ == "__main__":
    compare_friction()
