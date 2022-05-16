"""This module contains the base class for all algorithms.

Classes:

    BaseAlgorithm: the base class for algorithms.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseAlgorithm(metaclass=ABCMeta):
    """Base class for FairRecKit algorithms.

    An algorithm is used for carrying out recommender system experiments.
    Derived algorithms are expected to implement the abstract interface.

    Abstract methods:

    on_train

    Public methods:

    get_items
    get_name
    get_num_threads
    get_params
    get_users
    train
    """

    def __init__(self):
        """Construct the algorithm."""
        self.train_set = None
        self.users = None
        self.items = None

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

    def get_items(self) -> Optional[List[int]]:
        """Get the (unique) items the algorithm was trained on.

        Returns:
            a list of unique item IDs or None if the algorithm is not trained yet.
        """
        return self.items

    def get_users(self) -> Optional[List[int]]:
        """Get the (unique) users the algorithm was trained on.

        Returns:
            a list of unique user IDs or None if the algorithm is not trained yet.
        """
        return self.users

    def train(self, train_set: pd.DataFrame) -> None:
        """Train the algorithm on the specified train set.

        Args:
            train_set: with at least three columns: 'user', 'item', 'rating'.
        """
        self.train_set = train_set
        self.users = train_set['user'].unique()
        self.items = train_set['item'].unique()

        self.on_train()

    @abstractmethod
    def on_train(self) -> None:
        """Train the algorithm on the train set.

        Derived classes should implement the training logic
        of the algorithm.
        """
        raise NotImplementedError()
