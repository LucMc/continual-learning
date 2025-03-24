import gymnasium as gym
import os
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from continual_learning.envs.slippery_ant_v5 import ContinualAntEnv, SlipperyAntEnv
from functools import partial
from gymnasium.envs.mujoco.ant_v5 import AntEnv
import jax.random as random

def test_friction(friction: float, model=None):
    assert model, "please provide model to test"
    configured_env = partial(SlipperyAntEnv, friction=friction, render_mode="human")
    vec_env = DummyVecEnv([configured_env for _ in range(1)])
    obs = vec_env.reset()
    rewards = 0

    print("Testing with friction", friction)
    for _ in range(1000):
        vec_env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        rewards += reward

    vec_env.close()
    print(f"Friction: {friction} reward: {rewards}")

def make_env(base: gym.Env, env_spec, **kwargs):
    return lambda: TimeLimit(OrderEnforcing(PassiveEnvChecker(base(**kwargs))), env_spec.max_episode_steps)

def train_slippery_ant():
    friction = 1
    n_envs = 4
    total_timesteps = 500_000 * n_envs
    print("total_timesteps:\n", total_timesteps)

    env_spec = gym.spec("Ant-v5") # Could be simpler to just pass in max steps rather than require the whole spec
    env = VecMonitor(DummyVecEnv([make_env(SlipperyAntEnv, env_spec, friction=friction) for _ in range(4)])) # gym.make("Ant-v5", render_mode="human")

    model = PPO("MlpPolicy", env, learning_rate=1e-4, tensorboard_log=f"sbx_logs/f_{friction}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True) # Remember n_envs impacts total and change every

    ## Testing in same friction
    print("Testing with same friction")
    test_friction = partial(test_friction, model=model)
    test_friction(1)

    print("Testing with high friction")
    test_friction(4)

    ## Testing in lower friction
    print("Testing in low friction")
    test_friction(0.1)

def train_continual_ant():
    min_f = 0.1
    max_f = 2
    n_envs = 6
    changes = 6
    change_every = 320_000 # In unvecced timesteps

    total_timesteps = change_every * changes * n_envs
    policy_kwargs = {"net_arch" : {"pi": [256, 256], "vf": [256, 256]}}
    logdir = f"sbx_logs/n{n_envs}ce{(change_every*n_envs)//100_000}"
    run_name = f"PPO_{len(os.listdir(logdir)) if os.path.isdir(logdir) else 1}"

    print("total_timesteps:\n", total_timesteps)
    print(f"logdir: {logdir} run {run_name}")

    env_spec = gym.spec("Ant-v5") # Could be simpler to just pass in max steps rather than require the whole spec
    env = VecMonitor(DummyVecEnv([make_env(ContinualAntEnv,
                                           env_spec,
                                           seed=random.PRNGKey(11),
                                           change_friction_every=change_every,
                                           max_friction=max_f,
                                           min_friction=min_f) for _ in range(n_envs)]))


    model = PPO("MlpPolicy", env, learning_rate=1e-4, policy_kwargs=policy_kwargs, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps, progress_bar=True) # Remember n_envs impacts total and change every

    ## Testing in same friction
    print("Testing with same friction")
    test = partial(test_friction, model=model)
    test(1)

    print("Testing with high friction")
    test(max_f)

    ## Testing in lower friction
    print("Testing in low friction")
    test(min_f)

if __name__ == "__main__":
    train_continual_ant()
