from abc import abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, observation, hidden_state=None):
        raise NotImplementedError

    @abstractmethod
    def set_hidden_state(self, hidden_state)->None:
        raise NotImplementedError

    def step(self,observation:Dict[str,torch.Tensor],hidden_state=None,argmax=False)->Dict[str,Any]:
        dist, values, self.hidden_state = self(observation, hidden_state if hidden_state is not None else self.hidden_state)

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        logprob = dist.log_prob(actions)

        return {'actions':actions,'v_values':values,'logprobs':logprob,'hidden_states':self.hidden_state}

def epsgreedy_action(num_actions, policy_actions, epsilon):
    random_actions = torch.randint_like(policy_actions, num_actions)
    selector = torch.rand_like(random_actions, dtype=torch.float32)
    return torch.where(selector > epsilon, policy_actions, random_actions)

class QModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, observation:Dict)->torch.Tensor:
        raise NotImplementedError

    def step(self,observation:Dict[str,torch.Tensor],eps=0.001)->Dict[str,torch.Tensor]:
        q_values = self(observation)
        policy_actions = q_values.argmax(dim=1)
        if eps>0.0:
            actions = epsgreedy_action(self.num_actions, policy_actions, eps)
        else:
            actions = policy_actions
        return {'actions':actions,'q_values':q_values}
