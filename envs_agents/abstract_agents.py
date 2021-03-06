from abc import abstractmethod
from typing import Dict, Any, NamedTuple

import torch
import torch.nn as nn


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_dist_value(self, observation):
        raise NotImplementedError

    def step(self, observation: Dict[str, torch.Tensor], argmax=False):
        dist, values = self.calc_dist_value(observation)

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        logprob = dist.log_prob(actions)

        return {"actions": actions, "v_values": values, "logprobs": logprob}


def mix_in_some_random_actions(policy_actions, random_action_proba, num_actions):
    if random_action_proba > 0.0:
        random_actions = torch.randint_like(policy_actions, num_actions)
        selector = torch.rand_like(random_actions, dtype=torch.float32)
        actions = torch.where(
            selector > random_action_proba, policy_actions, random_actions
        )
    else:
        actions = policy_actions
    return actions


class QModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, observation: Dict) -> torch.Tensor:
        raise NotImplementedError

    def calc_q_values(self, observation: Dict) -> torch.Tensor:
        return self.forward(observation)

    def step(
        self, observation: Dict[str, torch.Tensor], random_action_proba=0.001
    ) -> Dict[str, torch.Tensor]:
        q_values = self(observation)
        policy_actions = q_values.argmax(dim=1)
        actions = mix_in_some_random_actions(
            policy_actions, random_action_proba, self.num_actions
        )
        return {"actions": actions, "q_values": q_values}
