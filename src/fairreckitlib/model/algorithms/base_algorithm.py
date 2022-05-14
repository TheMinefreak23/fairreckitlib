"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseAlgorithm(metaclass=ABCMeta):
    """Base class for FairRecKit algorithms.

    An algorithm is used for carrying out recommender system experiments.

    Public methods:

    get_name
    get_num_threads
    get_params
    train
    """

    def __init__(self):
        """Construct the algorithm."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the algorithm.

        Returns:
            the algorithm name.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_threads(self) -> int:
        """Get the max number of threads the algorithm can use.

        Returns:
            the number of threads.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the algorithm.

        Returns:
            the algorithm parameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, train_set: pd.DataFrame) -> None:
        """Train the algorithm on the specified train set.

        Args:
            train_set: with at least three columns: 'user', 'item', 'rating'.
        """
        raise NotImplementedError()
