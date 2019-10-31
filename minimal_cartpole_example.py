import time
from typing import Iterator
import gym
import torch
import torch.nn as nn
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from torch.optim.rmsprop import RMSprop
import torch.nn.functional as F
from tqdm import tqdm


def epsgreedy_action(num_actions, policy_actions, epsilon):
    random_actions = torch.randint_like(policy_actions, num_actions)
    selector = torch.rand_like(random_actions, dtype=torch.float32)
    return torch.where(selector > epsilon, policy_actions, random_actions)


class CartPoleAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.num_actions = action_space.n
        self.nn = nn.Sequential(
            *[
                nn.Linear(obs_space.shape[0], 24),
                nn.ReLU(),
                nn.Linear(24, 24),
                nn.ReLU(),
                nn.Linear(24, self.num_actions),
            ]
        )

    def calc_q_values(self, obs_batch):
        observation_tensor = torch.tensor(obs_batch, dtype=torch.float)
        q_values = self.nn(observation_tensor)
        return q_values

    def step_batch(self, obs_batch, eps=0.001):
        q_values = self.calc_q_values(obs_batch)
        policy_actions = q_values.argmax(dim=1)
        if eps > 0.0:
            actions = epsgreedy_action(self.num_actions, policy_actions, eps)
        else:
            actions = policy_actions
        return actions

    def step_single(self, obs, eps=0.001):
        obs_batch = np.expand_dims(obs, 0)
        actions = self.step_batch(obs_batch, eps)
        return int(actions.numpy()[0])


def visualize_it(env: gym.Env, agent: CartPoleAgent):
    max_steps = 1000
    while True:
        obs = env.reset()
        for steps in range(max_steps):
            is_open = env.render()

            action = agent.step_single(obs)
            obs, reward, done, info = env.step(action)
            if not is_open or done:
                break
        if steps < max_steps - 1:
            print("only %d steps" % steps)


def experience_generator(agent, env: gym.Env):

    while True:
        obs = env.reset()
        action = agent.step_single(obs)
        for it in range(1000):
            next_obs, _, next_done, info = env.step(action)
            if next_done:
                next_reward = -10.0
            else:
                next_reward = 1.0
            yield {
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "next_reward": next_reward,
                "next_done": next_done,
            }

            obs = next_obs
            action = agent.step_single(obs)

            if next_done:
                break


def gather_experience(experience_iter: Iterator, batch_size: int = 32):
    experience = [next(experience_iter) for _ in range(batch_size)]
    exp_arrays = {
        key: np.array([exp[key] for exp in experience]) for key in experience[0].keys()
    }
    return exp_arrays


def train_agent(agent: CartPoleAgent, env: gym.Env):
    optimizer = RMSprop(agent.parameters())
    train_steps = 3_000
    discount = 0.99

    experience_iterator = iter(experience_generator(agent, env))
    with tqdm(postfix=[{"avg-reward": 0}]) as pbar:

        for ep in range(train_steps):
            with torch.no_grad():
                agent.eval()
                experience = gather_experience(experience_iterator, batch_size=32)
                next_q_values = agent.calc_q_values(experience["next_obs"])
                max_next_value, _ = next_q_values.max(dim=1)

                mask = torch.tensor((1 - experience["next_done"]), dtype=torch.float)
                next_reward = torch.tensor(experience["next_reward"], dtype=torch.float)
                estimated_return = next_reward + discount * max_next_value * mask

            agent.train()
            q_values = agent.calc_q_values(experience["obs"])
            actions = torch.tensor(experience["action"]).unsqueeze(1)
            q_selected = q_values.gather(1, actions).squeeze(1)

            loss_value = F.mse_loss(q_selected, estimated_return)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            pbar.postfix[0]["avg-reward"] = np.mean(experience["next_reward"])
            pbar.update()


if __name__ == "__main__":
    env = CartPoleEnv()
    x = env.reset()
    obs, reward, done, info = env.step(1)
    agent = CartPoleAgent(env.observation_space, env.action_space)
    train_agent(agent, env)
    visualize_it(env, agent)
