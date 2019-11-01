import argparse
import os
import pprint
import sys

import gym
import torch

from dqn_algo import DQNAlgo
from envs_agents.cartpole import build_CartPoleEnv, CartPoleAgent
from train_methods import CsvLogger
from utils import set_seeds, get_logger, get_csv_writer
from visualize import visualize_it

if __name__ == "__main__":
    storage_path = os.getcwd() + "/storage"

    num_batches = 3000

    params = {
        "model_name": "cartpole-ddqn-1e-0p-%dkb" % (int(num_batches / 1000)),
        "seed": 1,
        "num_batches": num_batches,
        "num_rollout_steps": 1,
        "memory_size": 1000,
        "initial_eps_value": 0.001,
        "final_eps_value": 0.0001,
        "end_of_interpolation": num_batches,
        "num_processes": 0,
        "num_envs": 1,
    }

    args = argparse.Namespace(**params)
    model_dir = storage_path + "/" + args.model_name

    logger = get_logger(model_dir)
    csv_file, csv_writer = get_csv_writer(model_dir)

    set_seeds(args.seed)

    envs = build_CartPoleEnv(num_envs=args.num_envs, use_multiprocessing=False)
    q_model = CartPoleAgent(envs.observation_space, envs.action_space)
    target_model = CartPoleAgent(envs.observation_space, envs.action_space)

    if torch.cuda.is_available():
        q_model.cuda()

    algo = DQNAlgo(
        envs,
        q_model,
        target_model,
        num_rollout_steps=args.num_rollout_steps,
        lr=0.01,
        double_dpn=False,
        memory_size=args.memory_size,
        target_model_update_interval=20,
    )

    algo.train_model(
        args.num_batches,
        CsvLogger(csv_file, csv_writer, logger),
        initial_eps_value=args.initial_eps_value,
        final_eps_value=args.final_eps_value,
        end_of_interpolation=args.end_of_interpolation,
    )
    # visualize_it(envs, q_model)
