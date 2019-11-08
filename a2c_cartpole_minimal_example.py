from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.envs.classic_control import CartPoleEnv
from torch.distributions import Categorical


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
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def step_single(self, obs):
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


if __name__ == "__main__":
    env = CartPoleEnv()
    agent: CartPoleA2CAgent = CartPoleA2CAgent(env.observation_space, env.action_space)
    x = env.reset()
    agent.step_single(x)
