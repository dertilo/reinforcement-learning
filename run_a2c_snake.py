import os
import sys

from a2c import A2CParams, train_a2c_model
from envs_agents.snake import build_SnakeEnv, SnakeA2CAgent
from rl_utils import set_seeds, save_model, load_model
from visualize import visualize_it

import torch


def run_a2c_experiments(storage_path):
    num_batches = 200

    experiments = [
        ('1p',A2CParams(num_processes=1,num_batches=num_batches)),
    ]
    for exp_specific_name, params in experiments:
        model_dir = storage_path + "/" + params.model_name+'_'+exp_specific_name

        set_seeds(params.seed)

        envs = build_SnakeEnv(num_envs=params.num_envs, num_processes=params.num_processes)
        agent = SnakeA2CAgent(envs.observation_space, envs.action_space)

        train_a2c_model(agent,envs,params,num_batches)

        save_model(agent, model_dir)


if __name__ == "__main__":
    storage_path = os.getcwd() + "/storage"
    run_a2c_experiments(storage_path)
    model_dir = storage_path + "/" + os.listdir(storage_path)[0]
    agent = load_model(model_dir)
    visualize_it(build_SnakeEnv(num_envs=1, num_processes=0), agent, pause_dur=0.1)
