from typing import NamedTuple

import torch

from envs_agents.cartpole.a2c_cartpole_minimal_example import AgentStep
from rlutil.dictlist import DictList
from rlutil.experience_memory import ExperienceMemory


def train_batch(agent, batch, optimizer):

    agent.train()
    loss = agent.loss(batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(),max_norm=0.5)
    optimizer.step()
    return batch.env_steps.done.numpy()


def gather_exp_via_rollout(
    env, agent, exp_mem: ExperienceMemory, num_rollout_steps
):
    for _ in range(num_rollout_steps):
        env_step:NamedTuple = env.step(AgentStep(**exp_mem[exp_mem.last_written_idx].agent))
        agent_step:NamedTuple = agent.step(env_step)
        exp_mem.store_single(
            DictList.build({"env": env_step._asdict(), "agent": agent_step._asdict()})
        )


def build_experience_memory(agent, env, num_rollout_steps):
    initial_env_step = env.reset()
    with torch.no_grad():
        initial_agent_step = agent.step(initial_env_step)
    initial_exp = DictList.build(
        {"env": initial_env_step._asdict(), "agent": initial_agent_step._asdict()}
    )
    exp_mem = ExperienceMemory(num_rollout_steps + 1, initial_exp)
    return exp_mem