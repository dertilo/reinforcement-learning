import os
from typing import NamedTuple, Any

from baselines.bench import Monitor as BenchMonitor
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

from algos.train_methods import flatten_parallel_rollout, flatten_array
from envs_agents.cartpole.common import (
    train_batch,
    gather_exp_via_rollout,
    World,
    CartPoleEnvSelfReset,
    build_experience_memory,
    EnvStep,
)
from rlutil.dictlist import DictList

eps = np.finfo(np.float32).eps.item()


class RLparams(NamedTuple):
    num_batches: int = 1000
    discount: float = 0.99
    seed: int = 543
    lr: float = 0.01
    num_rollout_steps: int = 32


class AgentStep(NamedTuple):
    actions: torch.LongTensor


class Rollout(NamedTuple):
    env_steps: EnvStep
    agent_steps: AgentStep
    returnn: torch.FloatTensor


class CartPoleReinforceAgent(nn.Module):
    def __init__(self, obs_dim, num_actions) -> None:
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, 24)
        self.affine2 = nn.Linear(24, num_actions)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def calc_distr(self, state):
        probs = self.forward(state)
        distr = Categorical(probs)
        return distr

    def step(self, env_step: EnvStep):
        distr = self.calc_distr(env_step.observation)
        actions = distr.sample()
        return AgentStep(actions)

    def loss(self, batch: Rollout):
        distr = self.calc_distr(batch.env_steps.observation)
        losses = -distr.log_prob(batch.agent_steps.actions) * batch.returnn
        return losses.mean()


def calc_returns(rewards, dones, num_rollout_steps, discount):
    returns = torch.zeros(rewards.shape[0] - 1, rewards.shape[1])
    next_reward = 0
    for i in reversed(range(num_rollout_steps)):
        mask = torch.tensor((1 - dones[i + 1]), dtype=torch.float32)
        returns[i] = rewards[i + 1] + discount * next_reward * mask
        next_reward = returns[i]

    # returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def do_rollout(w: World, params: RLparams) -> Rollout:
    assert w.exp_mem.current_idx == 0
    w.exp_mem.last_becomes_first()

    gather_exp_via_rollout(w.env, w.agent, w.exp_mem, params.num_rollout_steps)
    assert w.exp_mem.last_written_idx == params.num_rollout_steps

    env_steps = w.exp_mem.buffer.env
    agent_steps = w.exp_mem.buffer.agent
    returnn = calc_returns(
        rewards=env_steps.reward,
        dones=env_steps.done,
        num_rollout_steps=params.num_rollout_steps,
        discount=params.discount,
    )
    return Rollout(
        **{
            "env_steps": DictList(**flatten_parallel_rollout(env_steps[:-1])),
            "agent_steps": DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            "returnn": flatten_array(returnn),
        }
    )


def run_cartpole_reinforce(args, log_dir="./logs/reinforce"):
    os.makedirs(log_dir, exist_ok=True)
    env = CartPoleEnv()
    agent = CartPoleReinforceAgent(env.observation_space.shape[0], env.action_space.n)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    env = BenchMonitor(env, log_dir, allow_early_resets=True)
    env = CartPoleEnvSelfReset(env)

    exp_mem = build_experience_memory(agent, env, args.num_rollout_steps)
    w = World(env, agent, exp_mem)

    with torch.no_grad():
        w.agent.eval()
        gather_exp_via_rollout(w.env, w.agent, w.exp_mem, args.num_rollout_steps)

    optimizer = torch.optim.Adam(agent.parameters(), args.lr)

    for k in tqdm(range(args.num_batches)):
        with torch.no_grad():
            agent.eval()
            batch = do_rollout(w, args)
        train_batch(agent, batch, optimizer)

    return agent, env


if __name__ == "__main__":
    args = RLparams(num_batches=100)
    agent, env = run_cartpole_reinforce(args)

    env = Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)

    while True:
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = agent.step(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
            env.render()
