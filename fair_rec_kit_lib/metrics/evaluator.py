""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class Evaluator(metaclass=ABCMeta):

    def __init__(self, metric, params):
        self._metric = metric
        self._params = params

    @abstractmethod
    def evaluate(self, test_set, recs):
        raise NotImplementedError
