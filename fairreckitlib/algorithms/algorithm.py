"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class Algorithm(metaclass=ABCMeta):

    def __init__(self, algo, params, **kwargs):
        self.algo = algo
        self.params = params

        self.num_threads = kwargs['num_threads']
        self.rating_scale = kwargs['rating_scale']

    @abstractmethod
    def train(self, train_set):
        raise NotImplementedError
