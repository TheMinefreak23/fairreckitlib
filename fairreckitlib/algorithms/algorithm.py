"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod


class Algorithm(metaclass=ABCMeta):
    """Base class for FairRecKit algorithms.

    An algorithm is used for carrying out recommender system experiments.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
    """
    def __init__(self, **kwargs):
        self.num_threads = kwargs['num_threads']

    @abstractmethod
    def get_params(self):
        """Get the parameters of the algorithm.

        Returns:
            dict with the algorithm parameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, train_set):
        """Trains the model on the specified train set.

        Args:
            train_set(pandas.DataFrame): with at least three columns:
                'user', 'item', 'rating'.
        """
        raise NotImplementedError()
