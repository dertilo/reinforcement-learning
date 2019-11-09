import numpy
import random
from collections import deque
from enum import IntEnum
from typing import List, Tuple

import gym
import torch
from gym import spaces
from gym_minigrid.minigrid import MiniGridEnv, Goal, Grid, Lava

from envs_agents.parallel_environment import SingleEnvWrapper, ParallelEnv
from visualize import visualize_it


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


class Snake(object):
    def __init__(self, parts: List[Tuple[int, int]]):
        self.body = deque(parts)

    def rm_tail(self):
        return self.body.pop()

    def grow_head(self, x, y):
        assert all((x, y) != pos for pos in self.body)
        self.body.appendleft((x, y))


class SnakeEnv(MiniGridEnv):
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2

    def __init__(self, size=9):

        super().__init__(grid_size=size, max_steps=None, see_through_walls=True)
        self.actions = SnakeEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def spawn_new_food(self):
        empties = [
            (i, j)
            for i in range(self.grid.height)
            for j in range(self.grid.width)
            if self.grid.get(i, j) is None
            and self.grid.get(i, j) != tuple(self.agent_pos)
        ]
        self.grid.set(*random.choice(empties), Goal())

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        # self.start_pos = (2, 2)
        yl, xl, _ = self.observation_space.spaces["image"].shape
        self.start_pos = (random.randint(2, yl - 2), random.randint(2, xl - 2))
        self.agent_pos = self.start_pos  # TODO: the env holding agent traits is shit!
        self.start_dir = random.randint(0, 3)
        self.agent_dir = self.start_dir
        self.snake = Snake([self.start_pos, tuple(self.start_pos - self.dir_vec)])
        [self.grid.set(*pos, Lava()) for pos in self.snake.body]

        self.spawn_new_food()

        self.mission = None

    def reset(self):
        return super().reset()

    def step(self, action):
        self.step_count += 1

        done = False

        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.forward:
            pass
        else:
            assert False, "unknown action: %d" % action

        fwd_pos = self.agent_pos + self.dir_vec
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None:
            self.grid.set(*self.agent_pos, Lava())
            self.snake.grow_head(*fwd_pos)
            self.grid.set(*self.snake.rm_tail(), None)
            self.agent_pos = fwd_pos

            reward = -0.001

        elif fwd_cell.type == "goal":
            self.grid.set(*self.agent_pos, Lava())
            self.snake.grow_head(*fwd_pos)
            self.agent_pos = fwd_pos

            self.spawn_new_food()
            reward = 1.0

        elif fwd_cell.type == "lava" or fwd_cell.type == "wall":
            reward = -1.0
            done = True

        else:
            assert False

        if self.step_count == 1 and done:
            assert False

        obs = self.gen_obs()
        assert any(
            [
                isinstance(self.grid.get(i, j), Goal)
                for i in range(self.grid.height)
                for j in range(self.grid.width)
            ]
        )
        return obs, reward, done, {}


class SnakeWrapper(gym.Env):
    def __init__(self):
        self.env = SnakeEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.step_counter = 0

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.step_counter += 1
        if self.step_counter % 1000 == 0:
            done = True
        if done:
            obs = self.env.reset()
        return {"observation": obs["image"], "reward": reward, "done": done}

    def reset(self):
        obs = self.env.reset()
        return {"observation": obs["image"], "reward": 0, "done": False}

    def render(self, mode="human"):
        return self.env.render(mode)

    def seed(self, seed=None):
        return self.env.seed(seed)


class PreprocessWrapper(gym.Env):
    def __init__(self, env: gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, act):
        actions = act.get("actions").cpu().numpy()
        env_step = self.env.step(actions)
        return self.process_env_step(env_step)

    def process_env_step(self, env_step):
        return {
            "reward": torch.tensor(env_step.get("reward"), dtype=torch.float),
            "done": torch.tensor(env_step.get("done")),
            "image": preprocess_images(env_step.get("observation")),
        }

    def reset(self):
        env_step = self.env.reset()
        return self.process_env_step(env_step)

    def render(self, mode="human"):
        return self.env.render(mode)


def build_SnakeEnv(num_envs, num_processes):
    if num_processes == 0:
        assert num_envs == 1
        env = PreprocessWrapper(SingleEnvWrapper(SnakeWrapper()))
    else:

        def build_env_supplier(i):
            def env_supplier():
                env = SnakeWrapper()
                env.seed(1000 + i)
                return env

            return env_supplier

        env = PreprocessWrapper(
            ParallelEnv.build(build_env_supplier, num_envs, num_processes)
        )
    return env


def minimal_test():

    env = build_SnakeEnv(num_envs=1, num_processes=0)
    x = env.reset()
    from envs_agents.snake.a2c_agent import SnakeA2CAgent

    agent = SnakeA2CAgent(env.observation_space, env.action_space)
    # agent = SnakeDQNAgent(env.observation_space, env.action_space)
    visualize_it(env, agent, pause_dur=0.1, num_steps=1000)


if __name__ == "__main__":
    minimal_test()
