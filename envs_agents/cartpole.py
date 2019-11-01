import math

import gym
import numpy
import torch
import torch.nn as nn
from gym.envs.classic_control import CartPoleEnv

from abstract_agents import QModel
from envs_agents.parallel_environment import SingleEnvWrapper, ParallelEnv
from visualize import visualize_it


class CartPolePreprocessWrapper(gym.Env):
    def __init__(self, env: gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, act):
        actions = act.get("actions").cpu().numpy()
        env_step = self.env.step(actions)
        return self.process_env_step(env_step)

    def process_env_step(self, env_step):
        return {
            "reward": torch.tensor(env_step.get("reward"), dtype=torch.float),
            "done": torch.tensor(env_step.get("done")),
            "observation": torch.tensor(env_step.get("observation"), dtype=torch.float),
        }

    def reset(self):
        env_step = self.env.reset()
        return self.process_env_step(env_step)

    def render(self, mode="human"):
        return self.env.render(mode)


class CartPoleDictEnvWrapper(gym.Env):
    def __init__(self, max_angle=12, max_num_steps=1000):
        self.env = CartPoleEnv()
        # self.env.theta_threshold_radians = max_angle * 2 * math.pi / 360
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.step_counter = 0
        self.max_num_steps = max_num_steps

    def step(self, action):
        if isinstance(action, numpy.ndarray):
            action = action[0]
        assert isinstance(action, numpy.int64)
        obs, _, done, _ = self.env.step(action)
        self.step_counter += 1
        if self.step_counter % self.max_num_steps == 0:
            done = True
        if done:
            reward = -10.0
            obs = self.env.reset()
        else:
            reward = 0.0
        return {"observation": obs, "reward": reward, "done": int(done)}

    def reset(self):
        obs = self.env.reset()
        return {"observation": obs, "reward": 0.0, "done": int(False)}

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class CartPoleAgent(QModel):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.num_actions = action_space.n
        self.nn = nn.Sequential(
            *[
                nn.Linear(obs_space.shape[0], 24),
                nn.ReLU(),
                nn.Linear(24, 24),
                nn.ReLU(),
                nn.Linear(24, self.num_actions),
            ]
        )

    def forward(self, observation):
        x = observation.get("observation")
        return self.nn(x)


def build_CartPoleEnv(num_envs, use_multiprocessing):
    if not use_multiprocessing:
        assert num_envs == 1
        env = CartPolePreprocessWrapper(SingleEnvWrapper(CartPoleDictEnvWrapper()))
    else:

        def build_env_supplier(i):
            def env_supplier():
                env = CartPoleDictEnvWrapper()
                env.seed(1000 + i)
                return env

            return env_supplier

        env = CartPolePreprocessWrapper(ParallelEnv.build(build_env_supplier, num_envs))
    return env


if __name__ == "__main__":
    # env = build_CartPoleEnv(num_envs=3,use_multiprocessing=True)
    env = build_CartPoleEnv(num_envs=1, use_multiprocessing=False)
    x = env.reset()
    agent = CartPoleAgent(env.observation_space, env.action_space)
    agent_step = agent.step(x)
    env.step(agent_step)
    visualize_it(env, agent)
