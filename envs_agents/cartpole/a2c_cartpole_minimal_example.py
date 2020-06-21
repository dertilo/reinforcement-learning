import os
from typing import NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Wrapper
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import Monitor
from torch.distributions import Categorical
from tqdm import tqdm

from envs_agents.cartpole.common import train_batch
from rlutil.dictlist import DictList
from rlutil.experience_memory import ExperienceMemory


def flatten_parallel_rollout(d):
    return {
        k: flatten_parallel_rollout(v) if isinstance(v, dict) else flatten_array(v)
        for k, v in d.items()
    }


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])


class A2CParams(NamedTuple):
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_rollout_steps: int = 4
    discount: float = 0.99
    lr: float = 1e-2
    gae_lambda: float = 0.95
    seed: int = 0
    num_batches: int = 2000


class EnvStep(NamedTuple):
    observation: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.LongTensor
    info: torch.LongTensor


class AgentStep(NamedTuple):
    actions: torch.LongTensor
    v_values: torch.FloatTensor


class Rollout(NamedTuple):
    env_steps: EnvStep
    agent_steps: AgentStep
    advantages: torch.FloatTensor
    returnn: torch.FloatTensor


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


class CartPoleA2CAgent(nn.Module):
    def __init__(self, obs_space, action_space, params: A2CParams):
        super().__init__()
        self.num_actions = action_space.n
        self.embedding_size = 8
        self.params = params
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

    def forward(self, observation_tensor):

        embedding = self.nn(observation_tensor)

        scores = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(scores, dim=1))

        value = self.critic(embedding).squeeze(1)

        return dist, value

    def calc_dist_value(self, observation: torch.FloatTensor):
        return self.forward(observation)

    def step(self, env_step: EnvStep, argmax=False):
        obs_batch = env_step.observation
        # if len(obs_batch.shape)<2:
        #     obs_batch = np.expand_dims(obs_batch,0)
        dist, values = self.forward(obs_batch)

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        assert not values.requires_grad
        return AgentStep(actions, values.data)

    def loss(self, batch: Rollout):
        dist, value = self.calc_dist_value(batch.env_steps.observation)
        entropy = dist.entropy().mean()
        policy_loss = -(
            dist.log_prob(batch.agent_steps.actions) * batch.advantages
        ).mean()
        value_loss = (value - batch.returnn).pow(2).mean()
        loss = (
            policy_loss
            - self.params.entropy_coef * entropy
            + self.params.value_loss_coef * value_loss
        )
        return loss


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


class World(NamedTuple):
    env: CartPoleEnvSelfReset
    agent: CartPoleA2CAgent
    exp_mem: ExperienceMemory


def gather_exp_via_rollout(
    env, agent, exp_mem: ExperienceMemory, num_rollout_steps
):
    for _ in range(num_rollout_steps):
        env_step:NamedTuple = env.step(AgentStep(**exp_mem[exp_mem.last_written_idx].agent))
        agent_step:NamedTuple = agent.step(env_step)
        exp_mem.store_single(
            DictList.build({"env": env_step._asdict(), "agent": agent_step._asdict()})
        )


def collect_experiences_calc_advantage(w: World, params: A2CParams) -> Rollout:
    assert w.exp_mem.current_idx == 0
    w.exp_mem.last_becomes_first()

    gather_exp_via_rollout(w.env, w.agent, w.exp_mem, params.num_rollout_steps)
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
    return Rollout(
        **{
            "env_steps": DictList(**flatten_parallel_rollout(env_steps[:-1])),
            "agent_steps": DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            "advantages": flatten_array(advantages),
            "returnn": flatten_array(agent_steps[:-1].v_values + advantages),
        }
    )


def visualize_it(env: gym.Env, agent: CartPoleA2CAgent, max_steps=1000):
    agent.eval()
    with torch.no_grad():
        while True:
            env_step = env.reset()
            for steps in range(max_steps):
                is_open = env.render()
                if not is_open:
                    return

                action = agent.step(env_step)
                env_step = env.step(DictList.build(action._asdict()))
                if not isinstance(env_step, EnvStep):
                    env_step = EnvStep(*env_step)
                if env_step.done:
                    break
            if steps < max_steps - 1:
                print("only %d steps" % steps)


from baselines.bench import Monitor as BenchMonitor


def run_cartpole_a2c(args: A2CParams, log_dir="./logs/a2c"):
    os.makedirs(log_dir, exist_ok=True)
    env = CartPoleEnv()
    env = BenchMonitor(env, log_dir, allow_early_resets=True)
    env = CartPoleEnvSelfReset(env)
    # env.seed(params.seed)
    # torch.manual_seed(params.seed)
    agent: CartPoleA2CAgent = CartPoleA2CAgent(
        env.observation_space, env.action_space, args
    )
    initial_env_step = env.reset()
    with torch.no_grad():
        initial_agent_step = agent.step(initial_env_step)
    initial_exp = DictList.build(
        {"env": initial_env_step._asdict(), "agent": initial_agent_step._asdict()}
    )
    exp_mem = ExperienceMemory(args.num_rollout_steps + 1, initial_exp)

    w = World(env, agent, exp_mem)
    with torch.no_grad():
        w.agent.eval()
        gather_exp_via_rollout(w.env, w.agent, w.exp_mem, args.num_rollout_steps)
    optimizer = torch.optim.Adam(agent.parameters(), args.lr)

    for k in tqdm(range(args.num_batches)):
        with torch.no_grad():
            w.agent.eval()
            rollout = collect_experiences_calc_advantage(w, args)

        train_batch(w.agent, rollout, optimizer)

    return agent, env


if __name__ == "__main__":
    #TODO(tilo): is still not learning properly!
    params = A2CParams(lr=0.01, num_rollout_steps=32,num_batches=2000,seed=1)
    agent, env = run_cartpole_a2c(params)

    env = Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)

    visualize_it(env, agent)
