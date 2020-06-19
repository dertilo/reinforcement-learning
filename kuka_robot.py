# based on: examples/pybullet/gym/pybullet_envs/baselines/train_kuka_grasping.py
from typing import NamedTuple

import gym
import torch
from baselines.common.models import mlp
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from baselines import deepq
import pybullet as pb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

eps = np.finfo(np.float32).eps.item()


class KukaRobotEnv(KukaGymEnv):

    distance = 0

    def _reward(self):
        # rewards is height of target object
        blockPos, blockOrn = pb.getBasePositionAndOrientation(self.blockUid)
        closestPoints = pb.getClosestPoints(
            self.blockUid, self._kuka.kukaUid, 1000, -1, self._kuka.kukaEndEffectorIndex
        )
        distance = closestPoints[0][8]
        reward = 1.0 / distance
        # if distance < self.distance:
        #     reward = -1.0
        # else:
        #     reward = 1.0
        # self.distance = distance
        return reward


class RLparams(NamedTuple):
    gamma: float = 0.99
    seed: int = 543
    render: bool = False
    log_interval: int = 10


class PolicyAgent(nn.Module):
    def __init__(self, obs_dim, num_actions) -> None:
        super().__init__()
        n_hid = 64
        self.affine1 = nn.Linear(obs_dim, n_hid)
        self.affine2 = nn.Linear(n_hid, num_actions)

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


def calc_returns(exp, gamma):
    returns = []
    R = 0
    for action, log_prob, r in reversed(exp):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def calc_loss(exp, gamma):
    returns = calc_returns(exp, gamma)
    policy_loss = [-log_prob * R for (_, log_prob, _), R in zip(exp, returns)]
    policy_loss = torch.cat(policy_loss).sum()
    return policy_loss


def train(env: KukaRobotEnv, agent: PolicyAgent, args: RLparams):

    num_games = 10
    for k in tqdm(range(num_games)):
        ep_reward, exp = collect_experience(env, agent)

        policy_loss = calc_loss(exp, args.gamma)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


if __name__ == "__main__":

    args = RLparams()
    env = KukaRobotEnv(renders=False, maxSteps=1000, isDiscrete=True)
    agent: PolicyAgent = PolicyAgent(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    train(env, agent, args)
    del env
    env = KukaRobotEnv(renders=True, maxSteps=1000, isDiscrete=True)
    while True:
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action, _ = agent.step(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
