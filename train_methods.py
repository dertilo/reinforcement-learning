import time

import torch
from typing import Dict, Callable

from dictlist import DictList
from experience_memory import ExperienceMemory


def flatten_parallel_rollout(d):
    return {
        k: flatten_parallel_rollout(v) if isinstance(v, dict) else flatten_array(v)
        for k, v in d.items()
    }


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])


class CsvLogger(object):
    def __init__(self, csv_file, csv_writer, logger, print_interval=100):
        self.update = 0
        self.print_interval = print_interval
        self.csv_file = csv_file
        self.csv_writer = csv_writer
        self.logger = logger

    def on_train_start(self):
        self.total_start_time = time.time()

    def log_it(self, logs):

        self.update += 1  # TODO

        duration = round(time.time() - self.total_start_time, 3)

        header = ["num updates", "duration in seconds"]
        logs = {
            k: v
            for k, v in zip(
                header + list(logs.keys()),
                [self.update, duration] + list(logs.values()),
            )
        }
        s = " ".join([k + "=%0.2f" % v for k, v in logs.items()])
        if self.update % self.print_interval == 0:
            self.logger.info(s)

        if self.update == 1:
            self.csv_writer.writerow(list(logs.keys()))
        self.csv_writer.writerow(list(logs.values()))
        self.csv_file.flush()


def step_logging_fun(log: Dict, exp: DictList):
    if log == {}:
        num_envs = len(exp)
        log_initial = {
            "rewards_sum": torch.zeros((num_envs,)),
            "num_steps_sum": torch.zeros((num_envs,)),
            "log_done_counter": 0,
            "log_episode_rewards": [],
            "log_num_steps": [],
        }
        for k, v in log_initial.items():
            log[k] = v

    reward = exp.env.reward
    done = torch.tensor(exp.env.done, dtype=torch.float)

    log["rewards_sum"] += torch.tensor(reward, dtype=torch.float)
    log["num_steps_sum"] += torch.ones(reward.shape[0])
    for i, done_ in enumerate(done):
        if done_:
            log["log_done_counter"] += 1
            log["log_episode_rewards"].append(log["rewards_sum"][i].item())
            log["log_num_steps"].append(log["num_steps_sum"][i].item())
    mask = torch.tensor(1 - done, dtype=torch.float)
    log["rewards_sum"] *= mask
    log["num_steps_sum"] *= mask


def gather_exp_via_rollout(
    env_step_fun, agent_step_fun, exp_memory: ExperienceMemory, num_rollout_steps
):
    for _ in range(num_rollout_steps):
        last_agent_step = exp_memory[exp_memory.last_written_idx].agent
        env_step = env_step_fun(last_agent_step)
        agent_step = agent_step_fun(env_step)
        exp_memory.store_single(DictList.build({"env": env_step, "agent": agent_step}))


def train_model(
    update_model: Callable[[], Dict],
    logger: CsvLogger,
    num_batches=None,
    duration_in_sec=None,
):
    logger.on_train_start()
    start = time.time()
    update_counter = 0

    def condition():
        if num_batches is not None:
            return update_counter < num_batches
        elif duration_in_sec is not None:
            return time.time() - start < duration_in_sec

    while condition():
        metrics_to_log = update_model()
        logger.log_it(metrics_to_log)
        update_counter += 1
