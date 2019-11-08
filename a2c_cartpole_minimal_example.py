from typing import Dict, Any, NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.envs.classic_control import CartPoleEnv
from torch.distributions import Categorical


class DictList(dict):
    # """A dictionnary of lists of same size. Dictionnary items can be
    # accessed using `.` notation and list items using `[]` notation.
    #
    # Example:
    #     >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
    #     >>> d.a
    #     [[1, 2], [3, 4]]
    #     >>> d[0]
    #     DictList({"a": [1, 2], "b": [5]})
    # """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    @classmethod
    def build(cls, d: dict):
        return DictList(
            **{k: cls.build(v) if isinstance(v, dict) else v for k, v in d.items()}
        )


def flatten_parallel_rollout(d):
    return {
        k: flatten_parallel_rollout(v) if isinstance(v, dict) else flatten_array(v)
        for k, v in d.items()
    }


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])


def fill_with_zeros(dim, d):
    return DictList(**{k: create_zeros(dim, v) for k, v in d.items()})


def create_zeros(dim, v):
    return (
        torch.zeros(*(dim,) + v.shape, dtype=v.dtype)
        if not isinstance(v, dict)
        else fill_with_zeros(dim, v)
    )


class ExperienceMemory(object):
    def __init__(self, buffer_capacity: int, datum: DictList):

        self.buffer_capacity = buffer_capacity
        self.current_idx = 0
        self.last_written_idx = 0
        self.buffer = fill_with_zeros(buffer_capacity, datum)
        self.log = {}

        self.store_single(datum)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def inc_idx(self):
        self.last_written_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

    def store_single(self, datum: DictList):
        self.buffer[self.current_idx] = datum
        self.inc_idx()
        return self.current_idx

    def last_becomes_first(self):
        assert self.current_idx == 0
        self.buffer[self.current_idx] = self.buffer[-1]
        self.inc_idx()
        return self.current_idx


class CartPoleA2CAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.num_actions = action_space.n
        self.embedding_size = 8
        self.nn = nn.Sequential(
            *[
                nn.Linear(obs_space.shape[0], 24),
                nn.ReLU(),
                nn.Linear(24, 24),
                nn.ReLU(),
                nn.Linear(24, self.embedding_size),
            ]
        )

        self.actor = nn.Sequential(nn.Linear(self.embedding_size, action_space.n))

        self.critic = nn.Sequential(nn.Linear(self.embedding_size, 1))

    def forward(self, obs_batch):

        observation_tensor = torch.tensor(obs_batch, dtype=torch.float)

        embedding = self.nn(observation_tensor)

        dist = Categorical(logits=F.log_softmax(self.actor(embedding), dim=1))

        value = self.critic(embedding).squeeze(1)

        return dist, value

    def step(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        actions = self.step_batch(obs_batch)
        return int(actions["actions"].numpy()[0])

    def step_batch(self, obs_batch, argmax=False) -> Dict[str, Any]:
        dist, values = self.forward(obs_batch)

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        logprob = dist.log_prob(actions)

        return {"actions": actions, "v_values": values, "logprobs": logprob}


def generalized_advantage_estimation(
    rewards, values, dones, num_rollout_steps, discount, gae_lambda
):
    assert values.shape[0] == 1 + num_rollout_steps
    advantage_buffer = torch.zeros(rewards.shape[0] - 1, rewards.shape[1])
    next_advantage = 0
    for i in reversed(range(num_rollout_steps)):
        mask = torch.tensor((1 - dones[i + 1]), dtype=torch.float32)
        bellman_delta = rewards[i + 1] + discount * values[i + 1] * mask - values[i]
        advantage_buffer[i] = (
            bellman_delta + discount * gae_lambda * next_advantage * mask
        )
        next_advantage = advantage_buffer[i]
    return advantage_buffer


class A2CParams(NamedTuple):
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_rollout_steps: int = 5
    discount: float = 0.99
    lr: float = 7e-4
    gae_lambda: float = 0.95


class World(NamedTuple):
    env: CartPoleEnv
    agent: CartPoleA2CAgent
    exp_mem: ExperienceMemory


def train_batch(w: World, p: A2CParams, optimizer):
    with torch.no_grad():
        exps = collect_experiences(w, p)

    w.agent.train()

    inds = np.arange(0, p.num_rollout_steps)

    sb = exps[inds]
    dist, value, _ = agent(sb.env_steps)

    entropy = dist.entropy().mean()

    policy_loss = -(dist.log_prob(sb.agent_steps.actions) * sb.advantages).mean()

    value_loss = (value - sb.returnn).pow(2).mean()

    loss = policy_loss - p.entropy_coef * entropy + p.value_loss_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
    optimizer.step()


def gather_exp_via_rollout(
    env_step_fun, agent_step_fun, exp_mem: ExperienceMemory, num_rollout_steps
):
    for _ in range(num_rollout_steps):
        env_step = env_step_fun(exp_mem[exp_mem.last_written_idx].agent)
        agent_step = agent_step_fun(env_step)
        exp_mem.store_single(DictList.build({"env": env_step, "agent": agent_step}))


def collect_experiences(w: World, params: A2CParams):
    agent.set_hidden_state(w.exp_mem[-1])
    assert w.exp_mem.current_idx == 0
    w.exp_mem.last_becomes_first()
    gather_exp_via_rollout(
        w.env.step, w.agent.step, w.exp_mem, params.num_rollout_steps
    )
    assert w.exp_mem.last_written_idx == params.num_rollout_steps

    env_steps = w.exp_mem.buffer.env
    agent_steps = w.exp_mem.buffer.agent
    advantages = generalized_advantage_estimation(
        rewards=env_steps.reward,
        values=agent_steps.v_values,
        dones=env_steps.done,
        num_rollout_steps=params.num_rollout_steps,
        discount=params.discount,
        gae_lambda=params.gae_lambda,
    )
    return DictList(
        **{
            "env_steps": DictList(**flatten_parallel_rollout(env_steps[:-1])),
            "agent_steps": DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            "advantages": flatten_array(advantages),
            "returnn": flatten_array(agent_steps[:-1].v_values + advantages),
        }
    )


if __name__ == "__main__":
    params = A2CParams()
    env = CartPoleEnv()
    agent: CartPoleA2CAgent = CartPoleA2CAgent(env.observation_space, env.action_space)
    # x = env.reset()
    # agent.step(x)

    initial_env_step = env.reset()

    with torch.no_grad():
        initial_agent_step = agent.step(initial_env_step)

    initial_exp = DictList.build({"env": initial_env_step, "agent": initial_agent_step})

    exp_mem = ExperienceMemory(params.num_rollout_steps + 1, initial_exp)

    w = World(env, agent, exp_mem)
    gather_exp_via_rollout(
        w.env.step, w.agent.step, w.exp_mem, params.num_rollout_steps
    )

    optimizer = torch.optim.Adam(agent.parameters(), params.lr)  #

    for k in range(2):
        train_batch(w, params, optimizer)
