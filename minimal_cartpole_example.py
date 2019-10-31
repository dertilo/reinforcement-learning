import time

import gym
import torch
import torch.nn as nn
import numpy as np


def epsgreedy_action(num_actions, policy_actions, epsilon):
    random_actions = torch.randint_like(policy_actions, num_actions)
    selector = torch.rand_like(random_actions, dtype=torch.float32)
    return torch.where(selector > epsilon, policy_actions, random_actions)

class CartPoleAgent(nn.Module):
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

    def step(self, obs, eps=0.001):
        observation_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        q_values = self.nn(observation_tensor)
        policy_actions = q_values.argmax(dim=1)
        if eps > 0.0:
            actions = epsgreedy_action(self.num_actions, policy_actions, eps)
        else:
            actions = policy_actions
        return int(actions.numpy()[0])


def visualize_it(env: gym.Env, agent: CartPoleAgent, pause_dur=0.1, max_steps=1000):

    for c in range(max_steps):
        obs = env.reset()
        while True:
            time.sleep(pause_dur)
            is_open = env.render()

            action = agent.step(obs)
            obs, reward, done, info = env.step(action)
            if not is_open or done:
                break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    x = env.reset()
    obs, reward, done, info = env.step(1)
    agent = CartPoleAgent(env.observation_space, env.action_space)
    visualize_it(env, agent)
