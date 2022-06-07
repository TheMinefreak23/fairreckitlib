"""This module contains the base class for all algorithms.

Classes:

    BaseAlgorithm: the base class for algorithms.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

from .matrix import Matrix


class BaseAlgorithm(metaclass=ABCMeta):
    """Base class for FairRecKit algorithms.

    An algorithm is used for carrying out recommender system experiments.
    Derived algorithms are expected to implement the abstract interface.

    Abstract methods:

    on_train

    Public methods:

    get_name
    get_num_threads
    get_params
    get_train_set
    train
    """

    def __init__(self):
        """Construct the algorithm."""
        self.train_set = None

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

    def get_train_set(self) -> Optional[Matrix]:
        """Get the train set that the algorithm was trained on.

        Returns:
            the train set matrix or None if the algorithm is not trained yet.
        """
        return self.train_set

    def train(self, train_set: Matrix) -> None:
        """Train the algorithm on the specified train set.

        Args:
            train_set: the matrix train set.

        Raises:
            ArithmeticError: possibly raised by an algorithm on training.
            MemoryError: possibly raised by an algorithm on training.
            RuntimeError: possibly raised by an algorithm on training.
            TypeError: when the specified train set does not have the correct matrix format.
        """
        self.train_set = train_set
        self.on_train(self.train_set.get_matrix())

    @abstractmethod
    def on_train(self, train_set: Any) -> None:
        """Train the algorithm on the train set.

        Derived classes should implement the training logic
        of the algorithm. The train set can be anything depending
        on the matrix that is used.

        Args:
            train_set: the set to train the algorithm with.
        """
        raise NotImplementedError()
