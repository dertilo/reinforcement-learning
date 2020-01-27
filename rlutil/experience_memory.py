import torch

from rlutil.dictlist import DictList


class ExperienceMemory(object):
    def __init__(self, buffer_capacity: int, datum: DictList):

        self.buffer_capacity = buffer_capacity
        self.current_idx = 0
        self.last_written_idx = 0
        self.buffer = fill_with_zeros(buffer_capacity, datum)

        self.store_single(datum)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def inc_idx(self):
        self.last_written_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

    def store_single(self, datum: DictList):
        self.buffer[self.current_idx] = datum
        self.inc_idx()
        return self.current_idx

    def last_becomes_first(self):
        assert self.current_idx == 0
        self.buffer[self.current_idx] = self.buffer[-1]
        self.inc_idx()
        return self.current_idx

def fill_with_zeros(dim, d):
    return DictList(**{k: create_zeros(dim, v) for k, v in d.items()})


def create_zeros(dim: int, v):
    if torch.is_tensor(v):
        z = torch.zeros(*(dim,) + v.shape, dtype=v.dtype)
    elif isinstance(v, dict):
        z = fill_with_zeros(dim, v)
    else:
        assert False
    return z
