from typing import NamedTuple, Any

import numpy as np
import torch
from gym import Wrapper

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


class CartPoleEnvSelfReset(Wrapper):
    def step(self, action: DictList):
        agent_action = int(action.actions.numpy()[0])
        obs, _, done, info = super().step(agent_action)

        if done:
            obs = self.reset().observation
        reward = -10.0 if done else 1.0
        return self._form_output(obs, reward, done)

    def reset(self):
        obs = super().reset()
        return self._form_output(obs, 0, False)

    def _form_output(self, obs, reward, done):
        d = self._torchify(
            {
                "observation": np.expand_dims(obs, 0).astype("float32"),
                "reward": np.array([reward], dtype=np.float32),
                "done": np.array([int(done)]),
                "info": np.array([int(done)]),
            }
        )
        return EnvStep(**d)

    def _torchify(self, d):
        return {k: torch.from_numpy(v) for k, v in d.items()}


class EnvStep(NamedTuple):
    observation: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.LongTensor
    info: torch.LongTensor

class World(NamedTuple):
    env: CartPoleEnvSelfReset
    agent: Any # TODO(tilo)
    exp_mem: ExperienceMemory


class Rollout(NamedTuple):
    env_steps: EnvStep
    agent_steps: AgentStep
    advantages: torch.FloatTensor
    returnn: torch.FloatTensor