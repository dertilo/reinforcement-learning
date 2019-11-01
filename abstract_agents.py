from abc import abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn


class ACModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, observation, hidden_state=None):
        raise NotImplementedError

    @abstractmethod
    def set_hidden_state(self, hidden_state) -> None:
        raise NotImplementedError

    def step(
        self, observation: Dict[str, torch.Tensor], hidden_state=None, argmax=False
    ) -> Dict[str, Any]:
        dist, values, self.hidden_state = self(
            observation, hidden_state if hidden_state is not None else self.hidden_state
        )

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        logprob = dist.log_prob(actions)

        return {
            "actions": actions,
            "v_values": values,
            "logprobs": logprob,
            "hidden_states": self.hidden_state,
        }


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
