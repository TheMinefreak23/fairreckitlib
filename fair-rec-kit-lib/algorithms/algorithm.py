"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class Algorithm(metaclass=ABCMeta):

    def __init__(self, algo, params):
        self._algo = algo
        self._params = params

    @abstractmethod
    def train(self, train_set):
        raise NotImplementedError

    def get_params(self):
        return self._params
