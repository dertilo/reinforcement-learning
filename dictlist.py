from abc import abstractmethod


class DictList(dict):
    # """A dictionnary of lists of same size. Dictionnary items can be
    # accessed using `.` notation and list items using `[]` notation.
    #
    # Example:
    #     >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
    #     >>> d.a
    #     [[1, 2], [3, 4]]
    #     >>> d[0]
    #     DictList({"a": [1, 2], "b": [5]})
    # """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    @classmethod
    def build(cls, d: dict):
        return DictList(
            **{k: cls.build(v) if isinstance(v, dict) else v for k, v in d.items()}
        )


if __name__ == "__main__":
    d = DictList(
        {"d1": DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]}), "c": [1, 2]}
    )
    print(d.keys())
    print(d[0])
    d[1] = d[0]
    print(d)
