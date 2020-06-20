import random
import os
import numpy
import torch
import collections
import csv
import logging
import sys


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def set_seeds(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_stats(array):
    if len(array) > 0:
        stats = {
            "mean": numpy.mean(array),
            "median": numpy.median(array),
            "std": numpy.std(array),
            "min": numpy.amin(array),
            "max": numpy.amax(array),
        }
    else:
        stats = {
            "mean": numpy.NaN,
            "median": numpy.NaN,
            "std": numpy.NaN,
            "min": numpy.NaN,
            "max": numpy.NaN,
        }

    return stats


def get_model_path(model_dir):
    return os.path.join(model_dir, "model.pt")


def load_model(model_dir):
    path = get_model_path(model_dir)
    model = torch.load(path)
    model.eval()
    return model


def save_model(model, model_dir:str):
    path = get_model_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(model, path)


def get_log_path(model_dir):
    return os.path.join(model_dir, "train-log.txt")


def get_logger(model_dir):
    path = get_log_path(model_dir)
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger()


def get_vocab_path(model_dir):
    return os.path.join(model_dir, "vocab.json")


def get_csv_path(model_dir):
    return os.path.join(model_dir, "train-log.csv")


def get_csv_writer(model_dir):
    csv_path = get_csv_path(model_dir)
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

import numpy as np

def update_progess_bar(pbar, params: dict, k=100,r=0.05):
    f = np.exp(np.log(r)/k)
    for param_name, value in params.items():
        if "running" in param_name:
            value = f * pbar.postfix[0][param_name] + (1 - f) * value
        pbar.postfix[0][param_name] = round(value, 5)
    pbar.update()