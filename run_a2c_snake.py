import os
import sys

from a2c import A2CAlgo
from envs_agents.snake import build_SnakeEnv, SnakeA2CAgent
from rl_utils import get_logger, set_seeds, save_model, load_model
from visualize import visualize_it

sys.path.append(os.getcwd())

import argparse
import pprint

import torch


def run_a2c_experiments(storage_path):
    num_batches = 200

    def update_default_params(exp_specific_name, p):

        params = {
            "model_name": "snake-a2c-1e-%dkb" % (int(num_batches / 1000)),
            "seed": 3,
            "num_envs": 8,
            "num_batches": num_batches,
            "num_rollout_steps": 5,
            "num_processes": 0,
        }
        params["model_name"] += "-" + exp_specific_name
        params.update(p)
        return params

    experiments = [
        ("1p", {"num_processes": 1}),
        # ('2p', {'num_processes': 2}),
        # ('4p', {'num_processes': 4})
    ]
    for exp_specific_name, exp_params in experiments:
        params = update_default_params(exp_specific_name, exp_params)
        args = argparse.Namespace(**params)
        model_dir = storage_path + "/" + args.model_name

        set_seeds(args.seed)

        envs = build_SnakeEnv(num_envs=args.num_envs, num_processes=args.num_processes)
        agent = SnakeA2CAgent(envs.observation_space, envs.action_space)

        if torch.cuda.is_available():
            agent.cuda()

        algo = A2CAlgo(
            envs,
            agent,
            num_rollout_steps=args.num_rollout_steps,
            discount=0.99,
            lr=5e-4,
            gae_lambda=0.95,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            num_recurr_steps=1,
        )

        algo.train_model(args.num_batches)

        save_model(agent, model_dir)


if __name__ == "__main__":
    storage_path = os.getcwd() + "/storage"
    run_a2c_experiments(storage_path)
    model_dir = storage_path + "/" + os.listdir(storage_path)[0]
    agent = load_model(model_dir)
    visualize_it(build_SnakeEnv(num_envs=1, num_processes=0), agent, pause_dur=0.1)
