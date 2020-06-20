import os

from gym.envs.classic_control import CartPoleEnv

from envs_agents.cartpole.dqn_cartpole_minimal_example import (
    CartPoleAgent,
    train_agent,
    run_cartpole_dqn,
)

from baselines.common import plot_util as pu
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dqn_dir = "logs/dqn"
    os.makedirs(dqn_dir, exist_ok=True)
    agent, _ = run_cartpole_dqn(log_dir=dqn_dir)

    results = pu.load_results(dqn_dir)
    f, ax = pu.plot_results(results)
    f.savefig("logs/logs.png")
