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


def plot_save_results(xy_fn, file="logs/time_rewards.png"):
    f, ax = pu.plot_results(
        results,
        xy_fn=xy_fn,
        split_fn=lambda _: "",
        average_group=True,
        shaded_err=False,
    )
    f.savefig(file)


if __name__ == "__main__":
    [run_cartpole_dqn(log_dir="logs/dqn-%d" % k, seed=k) for k in range(3)]

    [
        run_cartpole_reinforce(
            RLparams(num_games=200, seed=k), log_dir="logs/reinforce-%d" % k
        )
        for k in range(3)
    ]

    results = pu.load_results("logs")
    plot_save_results(lr_fn, "logs/steps_rewards.png")
    plot_save_results(tr_fn, "logs/time_rewards.png")
