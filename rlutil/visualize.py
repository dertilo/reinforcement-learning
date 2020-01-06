#!/usr/bin/env python3

import gym
import time

from rlutil.rl_utils import set_seeds

try:
    import gym_minigrid
except ImportError:
    pass


def visualize_it(env: gym.Env, agent, pause_dur=0.001, seed=0, num_steps=1000):

    set_seeds(seed)
    env.seed(seed)
    c = 0
    obs = env.reset()
    while True:
        c += 1
        if pause_dur > 0.0:
            time.sleep(pause_dur)
        is_open = env.render()

        action = agent.step(obs)
        obs = env.step(action)

        if not is_open:
            break
        if c > num_steps:
            break


# if __name__ == '__main__':
#     model_path = '/home/tilo/hpc/torch-rl/storage/snake-a2c-1e-100kb-4p/'
#     # model_path = '/home/tilo/hpc/torch-rl/storage/snake-ddqn-1e-0p-1000kb-1kmem/'
#     # model_path = '/home/tilo/torch-rl/storage/snake-ddqn-1e-0p-1000kb-1kmem/'
#     agent = load_model(model_path)
#     visualize_it(build_SnakeEnv(num_envs=1, num_processes=0),agent)
