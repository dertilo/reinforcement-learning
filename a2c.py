from typing import NamedTuple

import gym
import numpy
import torch
from tqdm import tqdm

from dictlist import DictList
from envs_agents.abstract_agents import ACModel
from train_methods import (
    gather_exp_via_rollout,
    flatten_array,
    flatten_parallel_rollout,
)
from experience_memory import ExperienceMemory

class RLParams(NamedTuple):
    gae_lambda: float = 0.95
    discount: float = 0.99
    num_rollout_steps: int = 5


def generalized_advantage_estimation(
    rewards, values, dones, p:RLParams
):
    assert values.shape[0] == 1 + p.num_rollout_steps
    advantage_buffer = torch.zeros(rewards.shape[0] - 1, rewards.shape[1])
    next_advantage = 0
    for i in reversed(range(p.num_rollout_steps)):
        mask = torch.tensor((1 - dones[i + 1].float()), dtype=torch.float32)
        bellman_delta = rewards[i + 1] + p.discount * values[i + 1] * mask - values[i]
        advantage_buffer[i] = (
            bellman_delta + p.discount * p.gae_lambda * next_advantage * mask
        )
        next_advantage = advantage_buffer[i]
    return advantage_buffer


def update_progess_bar(pbar, params: dict, f=0.95):
    for param_name, value in params.items():
        if "running" in param_name:
            value = f * pbar.postfix[0][param_name] + (1 - f) * value
        pbar.postfix[0][param_name] = round(value, 5)
    pbar.update()


class A2CParams(NamedTuple):
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    rlparams:RLParams=RLParams()
    lr: float = 5e-4
    num_recurr_steps: int = 1  # not yet implemented
    seed: int = 3
    model_name: str = "snake-a2c"
    num_envs: int = 8
    num_batches: int = 10
    num_processes: int = 0


class World(NamedTuple):
    env: gym.Env
    agent: ACModel
    exp_mem: ExperienceMemory


def build_experience_memory(env, agent: ACModel, p: A2CParams) -> ExperienceMemory:
    initial_env_step = env.reset()

    with torch.no_grad():
        initial_agent_step = agent.step(initial_env_step)
    initial_exp = DictList.build({"env": initial_env_step, "agent": initial_agent_step})

    exp_mem = ExperienceMemory(p.rlparams.num_rollout_steps + 1, initial_exp)

    gather_exp_via_rollout(env.step, agent.step, exp_mem, p.rlparams.num_rollout_steps)

    return exp_mem


def collect_experiences_calc_advantage(w: World, p: A2CParams):
    w.agent.set_hidden_state(w.exp_mem[-1])#TODO(tilo): not yet implemented recurrence
    assert w.exp_mem.current_idx == 0
    w.exp_mem.last_becomes_first()
    gather_exp_via_rollout(w.env.step, w.agent.step, w.exp_mem, p.rlparams.num_rollout_steps)
    assert w.exp_mem.last_written_idx == p.rlparams.num_rollout_steps

    env_steps = w.exp_mem.buffer.env
    agent_steps = w.exp_mem.buffer.agent
    advantages = generalized_advantage_estimation(
        rewards=env_steps.reward,
        values=agent_steps.v_values,
        dones=env_steps.done,
        p=p.rlparams
    )
    return DictList(
        **{
            "env_steps": DictList(**flatten_parallel_rollout(env_steps[:-1])),
            "agent_steps": DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            "advantages": flatten_array(advantages),
            "returnn": flatten_array(agent_steps[:-1].v_values + advantages),
        }
    )


class A2CAlgo(object):
    def __init__(self, env: gym.Env, agent: ACModel, p: A2CParams):

        assert p.rlparams.num_rollout_steps % p.num_recurr_steps == 0

        self.p: A2CParams = p

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        exp_mem = build_experience_memory(env, agent, p)

        self.w = World(env, agent, exp_mem)

        self.optimizer = torch.optim.Adam(agent.parameters(), p.lr)

    def train_batch(self):
        with torch.no_grad():
            exps = collect_experiences_calc_advantage(self.w,self.p)

        self.w.agent.train()
        loss = self.calc_loss(exps)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.w.agent.parameters(), self.p.max_grad_norm)
        self.optimizer.step()
        return float(torch.mean(exps.env_steps.reward).numpy())

    def calc_loss(self, sb):
        dist, value, _ = self.w.agent(sb.env_steps)
        entropy = dist.entropy().mean()
        policy_loss = -(dist.log_prob(sb.agent_steps.actions) * sb.advantages).mean()
        value_loss = (value - sb.returnn).pow(2).mean()
        loss = (
            policy_loss
            - self.p.entropy_coef * entropy
            + self.p.value_loss_coef * value_loss
        )
        return loss

    def train_model(self, num_batches):
        with tqdm(postfix=[{"running_per_step_reward": 0.0}]) as pbar:
            for k in range(num_batches):
                r = self.train_batch()
                update_progess_bar(pbar, {"running_per_step_reward": r})
