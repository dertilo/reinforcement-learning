from typing import NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

"""
based on: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
"""


class RLparams(NamedTuple):
    gamma: float = 0.99
    seed: int = 543
    render: bool = False
    log_interval: int = 10


class PolicyAgent(nn.Module):
    def __init__(self, obs_dim, num_actions) -> None:
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, 24)
        self.affine2 = nn.Linear(24, num_actions)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def step(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def collect_experience(env: gym.Env, agent: PolicyAgent):
    state, ep_reward = env.reset(), 0
    exp = []
    for t in range(1, 10000):  # Don't infinite loop while learning
        action, log_probs = agent.step(state)
        state, reward, done, _ = env.step(action)
        exp.append((action, log_probs, reward))
        ep_reward += reward
        if done:
            break

    return ep_reward, exp


if __name__ == "__main__":
    args = RLparams()
    env = gym.make("CartPole-v1")
    agent: PolicyAgent = PolicyAgent(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    running_reward = 0
    episode_counter = 0
    while running_reward < env.spec.reward_threshold:
        ep_reward, exp = collect_experience(env, agent)

        R = 0
        policy_loss = []
        returns = []
        for action, log_prob, r in reversed(exp):
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (_, log_prob, _), R in zip(exp, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        episode_counter += 1
        if episode_counter % args.log_interval == 0:
            print(
                "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    episode_counter, ep_reward, running_reward
                )
            )

    while True:
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action, _ = agent.step(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
            env.render()


"""
Episode 200	Last reward: 500.00	Average reward: 472.84
"""
