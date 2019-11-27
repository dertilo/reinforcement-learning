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


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 24)
        self.affine2 = nn.Linear(24, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def collect_experience(env: gym.Env):
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    return ep_reward


if __name__ == "__main__":
    args = RLparams()

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    env = gym.make("CartPole-v1")
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    running_reward = 0
    episode_counter = 0
    while running_reward < env.spec.reward_threshold:
        ep_reward = collect_experience(env)

        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.rewards[:]
        del policy.saved_log_probs[:]

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
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
            env.render()


"""
Episode 200	Last reward: 500.00	Average reward: 472.84
"""
