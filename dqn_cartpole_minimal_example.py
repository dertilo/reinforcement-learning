import time
from typing import Iterator, Dict
import gym
import torch
import torch.nn as nn
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from torch.optim.rmsprop import RMSprop
import torch.nn.functional as F
from tqdm import tqdm
from util.util_methods import iterable_to_batches


def mix_in_some_random_actions(policy_actions, eps, num_actions):
    if eps > 0.0:
        random_actions = torch.randint_like(policy_actions, num_actions)
        selector = torch.rand_like(random_actions, dtype=torch.float32)
        actions = torch.where(selector > eps, policy_actions, random_actions)
    else:
        actions = policy_actions
    return actions


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
        actions = mix_in_some_random_actions(policy_actions, eps, self.num_actions)
        return actions

    def step_single(self, obs, eps=0.001):
        obs_batch = np.expand_dims(obs, 0)
        actions = self.step_batch(obs_batch, eps)
        return int(actions.numpy()[0])


def visualize_it(env: gym.Env, agent: CartPoleAgent, max_steps=1000):

    while True:
        obs = env.reset()
        for steps in range(max_steps):
            is_open = env.render()
            if not is_open:
                return

            action = agent.step_single(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        if steps < max_steps - 1:
            print("only %d steps" % steps)


def experience_generator(agent, env: gym.Env):

    while True:
        obs = env.reset()
        for it in range(1000):
            action = agent.step_single(obs)
            next_obs, _, next_done, info = env.step(action)

            yield {
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "next_reward": -10.0 if next_done else 1.0,
                "next_done": next_done,
            }

            obs = next_obs

            if next_done:
                break


def gather_experience(experience_iter: Iterator, batch_size: int = 32):
    experience_batch = [next(experience_iter) for _ in range(batch_size)]
    exp_arrays = {
        key: np.array([exp[key] for exp in experience_batch])
        for key in experience_batch[0].keys()
    }
    return exp_arrays


def calc_estimated_return(
    agent: CartPoleAgent, experience: Dict[str, np.ndarray], discount=0.99
):
    next_q_values = agent.calc_q_values(experience["next_obs"])
    max_next_value, _ = next_q_values.max(dim=1)
    mask = torch.tensor((1 - experience["next_done"]), dtype=torch.float)
    next_reward = torch.tensor(experience["next_reward"], dtype=torch.float)
    estimated_return = next_reward + discount * max_next_value * mask
    return estimated_return


def calc_loss(agent, estimated_return, observation, action):
    q_values = agent.calc_q_values(observation)
    actions_tensor = torch.tensor(action).unsqueeze(1)
    q_selected = q_values.gather(1, actions_tensor).squeeze(1)
    loss_value = F.mse_loss(q_selected, estimated_return)
    return loss_value


def train_agent(agent: CartPoleAgent, env: gym.Env, num_batches=3_000, batch_size=32):
    optimizer = RMSprop(agent.parameters())
    experience_iterator = iter(experience_generator(agent, env))

    for it in tqdm(range(num_batches)):
        with torch.no_grad():
            agent.eval()
            exp = gather_experience(experience_iterator, batch_size=batch_size)
            for d in exp["next_done"]:
                yield d
            estimated_return = calc_estimated_return(agent, exp)

        agent.train()
        loss_value = calc_loss(agent, estimated_return, exp["obs"], exp["action"])
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()


def plot_average_game_length(dones, avg_size):
    batches = list(iterable_to_batches(dones, avg_size))
    avg_game_length = [avg_size / (1 + sum(b)) for b in batches]
    from matplotlib import pyplot as plt

    x = [k * avg_size for k in range(len(avg_game_length))]
    plt.plot(x, avg_game_length)
    plt.xscale("linear")
    plt.yscale("log")
    plt.ylabel("avg game-length")
    plt.xlabel("window index (averaged over %d steps)" % avg_size)
    plt.savefig("images/learn_curve_dqn_cartpole.png")
    visualize_it(env, agent)


if __name__ == "__main__":
    env = CartPoleEnv()
    x = env.reset()
    obs, reward, done, info = env.step(1)
    agent = CartPoleAgent(env.observation_space, env.action_space)
    batch_size = 32
    dones = list(train_agent(agent, env, num_batches=3000, batch_size=batch_size))

    plot_average_game_length(dones,avg_size=300 * batch_size)
