import gym
import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from envs_agents.abstract_agents import ACModel, initialize_parameters


class SnakeA2CAgent(ACModel):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()
        self.hidden_state = None
        self.has_hiddenstate = use_memory

        # Define image embedding
        # self.visual_nn = nn.Sequential(
        #     nn.Conv2d(3, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU()
        # )
        # n = obs_space["image"][0]
        # m = obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.image_embedding_size = 64  # ((n-1)//2-2)*((m-1)//2-2)*64
        image_shape = obs_space.spaces["image"].shape
        self.visual_nn = nn.Sequential(
            *[
                nn.Linear(image_shape[0] * image_shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, self.image_embedding_size),
                nn.ReLU(),
            ]
        )

        if self.has_hiddenstate:
            self.memory_rnn = nn.LSTMCell(
                self.image_embedding_size, self.semi_memory_size
            )

        self.embedding_size = self.semi_memory_size
        # if self.use_text:
        #     self.embedding_size += self.text_embedding_size

        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                # nn.Linear(self.embedding_size, 64),
                # nn.Tanh(),
                nn.Linear(self.image_embedding_size, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        self.critic = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            # nn.Tanh(),
            nn.Linear(self.image_embedding_size, 1)
        )

        self.apply(initialize_parameters)

    @property
    def hiddenstate_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, observation, memory=None):
        if "done" in observation:
            self._reset_hidden_state((1 - observation.get("done").float()).unsqueeze(1))

        image = observation.get("image")
        if memory is None:
            memory = self.hidden_state
        x = image[:, :, :, 0].view(image.size(0), -1)
        # x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.visual_nn(x)
        x = x.reshape(x.shape[0], -1)

        if self.has_hiddenstate:
            assert False
            hidden = (
                memory[:, : self.semi_memory_size],
                memory[:, self.semi_memory_size :],
            )
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
            self.hidden_state = memory
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def set_hidden_state(self, agent_step):
        self.hidden_state = agent_step.get("hidden_states")

    def _reset_hidden_state(self, mask):
        if self.hidden_state is None or self.hidden_state.shape != mask.shape:
            self.hidden_state = torch.zeros(mask.shape[0], self.hiddenstate_size)
        else:
            self.hidden_state = self.hidden_state * mask

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]