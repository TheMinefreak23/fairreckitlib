"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod

from .algorithm import Algorithm


class PredictorAlgorithm(Algorithm, metaclass=ABCMeta):

    @abstractmethod
    def predict(self, user, item):
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, user_item_pairs):
        raise NotImplementedError()
