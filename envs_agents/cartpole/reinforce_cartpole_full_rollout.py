import os
from typing import NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import Monitor
from torch.distributions import Categorical
from tqdm import tqdm

"""
based on: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
"""

eps = np.finfo(np.float32).eps.item()


class REfullrolloutArgs(NamedTuple):
    num_games: int = 1000
    gamma: float = 0.99
    seed: int = 543


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


def collect_experience(
    env: gym.Env, agent: PolicyAgent
) -> list[tuple[int, float, float]]:
    observation, ep_reward = env.reset(), 0
    experience = []
    for t in range(1, 1000):  # Don't infinite loop while learning
        action, log_probs = agent.step(observation)
        observation, reward, done, _ = env.step(action)
        experience.append((action, log_probs, reward))
        if done:
            break

    return experience


def calc_returns(exp: list[tuple[int, float, float]], gamma):
    returns = []
    Returrn = 0  # discounted sum
    for action, log_prob, reward in reversed(exp):
        Returrn = reward + gamma * Returrn
        returns.insert(0, Returrn)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def calc_loss(exp: list[tuple[int, float, float]], gamma: float):
    returns = calc_returns(exp, gamma)
    policy_loss = [-log_prob * R for (_, log_prob, _), R in zip(exp, returns)]
    policy_loss = torch.cat(policy_loss).sum()
    return policy_loss


def train(env: gym.Env, agent: PolicyAgent, args: REfullrolloutArgs):
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)

    for k in tqdm(range(args.num_games)):
        exp = collect_experience(env, agent)

        policy_loss = calc_loss(exp, args.gamma)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


def run_cartpole_reinforce_full_rollout(args, log_dir="./logs/reinforce_full_rollout"):
    os.makedirs(log_dir, exist_ok=True)
    env = CartPoleEnv()
    agent = PolicyAgent(env.observation_space.shape[0], env.action_space.n)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    # env = BenchMonitor(env, log_dir, allow_early_resets=True)
    train(env, agent, args)
    return agent, env


if __name__ == "__main__":
    args = REfullrolloutArgs(num_games=100)
    agent, env = run_cartpole_reinforce_full_rollout(args)

    env = Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)

    while True:
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action, _ = agent.step(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
            env.render()
