from typing import NamedTuple

import gym
import torch
from tqdm import tqdm

from rlutil.dictlist import DictList
from envs_agents.abstract_agents import ACModel
from algos.train_methods import (
    gather_exp_via_rollout,
    flatten_array,
    flatten_parallel_rollout,
)
from rlutil.experience_memory import ExperienceMemory
from rlutil.rl_utils import update_progess_bar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLParams(NamedTuple):
    gae_lambda: float = 0.95
    discount: float = 0.99
    num_rollout_steps: int = 5


def generalized_advantage_estimation(rewards, values, dones, p: RLParams):
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


class A2CParams(NamedTuple):
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    rlparams: RLParams = RLParams()
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
    assert w.exp_mem.current_idx == 0
    w.exp_mem.last_becomes_first()
    gather_exp_via_rollout(
        w.env.step, w.agent.step, w.exp_mem, p.rlparams.num_rollout_steps
    )
    assert w.exp_mem.last_written_idx == p.rlparams.num_rollout_steps

    env_steps = w.exp_mem.buffer.env
    agent_steps = w.exp_mem.buffer.agent
    advantages = generalized_advantage_estimation(
        rewards=env_steps.reward,
        values=agent_steps.v_values,
        dones=env_steps.done,
        p=p.rlparams,
    )
    return DictList(
        **{
            "env_steps": DictList(**flatten_parallel_rollout(env_steps[:-1])),
            "agent_steps": DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            "advantages": flatten_array(advantages),
            "returnn": flatten_array(agent_steps[:-1].v_values + advantages),
        }
    )


def calc_loss(w: World, p: A2CParams, sb: DictList):
    dist, value = w.agent.calc_dist_value(sb.env_steps)
    entropy = dist.entropy().mean()
    policy_loss = -(dist.log_prob(sb.agent_steps.actions) * sb.advantages).mean()
    value_loss = (value - sb.returnn).pow(2).mean()
    loss = policy_loss - p.entropy_coef * entropy + p.value_loss_coef * value_loss
    return loss


def train_batch(w: World, p: A2CParams, optimizer):
    with torch.no_grad():
        exps = collect_experiences_calc_advantage(w, p)

    w.agent.train()
    loss = calc_loss(w, p, exps)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(w.agent.parameters(), p.max_grad_norm)
    optimizer.step()
    return float(torch.mean(exps.env_steps.reward).numpy())


def train_a2c_model(agent, env, p: A2CParams, num_batches: int):
    if torch.cuda.is_available():
        agent.cuda()

    assert p.rlparams.num_rollout_steps % p.num_recurr_steps == 0

    exp_mem = build_experience_memory(env, agent, p)

    w = World(env, agent, exp_mem)

    optimizer = torch.optim.Adam(agent.parameters(), p.lr)

    with tqdm(postfix=[{"running_per_step_reward": 0.0}]) as pbar:
        for k in range(num_batches):
            r = train_batch(w, p, optimizer)
            update_progess_bar(pbar, {"running_per_step_reward": r})
