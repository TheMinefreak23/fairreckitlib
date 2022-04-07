"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod

from .algorithm import Algorithm


class RecommenderAlgorithm(Algorithm, metaclass=ABCMeta):

    @abstractmethod
    def recommend(self, user, num_items=10):
        raise NotImplementedError()

    @abstractmethod
    def recommend_batch(self, users, num_items=10):
        raise NotImplementedError()
