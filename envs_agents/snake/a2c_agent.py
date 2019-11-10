from typing import Dict

import gym
import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from envs_agents.abstract_agents import ACModel, initialize_parameters


class SnakeA2CAgent(ACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.image_embedding_size = 64
        image_shape = obs_space.spaces["image"].shape
        self.visual_nn = nn.Sequential(
            *[
                nn.Linear(image_shape[0] * image_shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, self.image_embedding_size),
                nn.ReLU(),
            ]
        )


        self.embedding_size = self.semi_memory_size
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 1)
        )

        self.apply(initialize_parameters)

    @property
    def hiddenstate_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def calc_dist_value(self,observation:Dict[str,torch.Tensor]):
        return self.forward(observation.get('image'))

    def forward(self, image):

        x = image[:, :, :, 0].view(image.size(0), -1)
        # x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.visual_nn(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value