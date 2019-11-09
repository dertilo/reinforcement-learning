from torch import nn as nn

from envs_agents.abstract_agents import QModel, initialize_parameters


class SnakeDQNAgent(QModel):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.num_actions = action_space.n
        image_shape = obs_space.spaces["image"].shape
        self.q_nn = nn.Sequential(
            *[
                nn.Linear(image_shape[0] * image_shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_actions),
            ]
        )

        self.apply(initialize_parameters)

    def forward(self, observation):
        image = observation.get("image")
        x = image[:, :, :, 0].view(image.size(0), -1)
        return self.q_nn(x)