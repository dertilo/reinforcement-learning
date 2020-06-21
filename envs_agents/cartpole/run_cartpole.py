from baselines.common import plot_util as pu
from baselines.common.plot_util import smooth
from matplotlib import pyplot as plt

from envs_agents.cartpole.dqn_cartpole_minimal_example import run_cartpole_dqn
from envs_agents.cartpole.reinforce_cartpole_minimal_example import (
    RLparams,
    run_cartpole_reinforce,
)
import numpy as np

def lr_fn(r):
    x = np.cumsum(r.monitor.l)
    y = smooth(r.monitor.r, radius=10)
    return x, y


def tr_fn(r):
    x = r.monitor.t
    y = smooth(r.monitor.r, radius=10)
    return x, y


if __name__ == "__main__":
    run_cartpole_dqn()

    args = RLparams(num_games=500)
    run_cartpole_reinforce(args)

    results = pu.load_results("logs")
    f, ax = pu.plot_results(results, xy_fn=lr_fn, split_fn=lambda _: "")
    f.savefig("logs/steps_rewards.png")
    f, ax = pu.plot_results(results, xy_fn=tr_fn, split_fn=lambda _: "")
    f.savefig("logs/time_rewards.png")
