"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base_algorithm import BaseAlgorithm


class BasePredictor(BaseAlgorithm, metaclass=ABCMeta):
    """Base class for FairRecKit predictors.

    A predictor is used for prediction experiments. It computes predictions
    for any user and item that it was trained on.

    Public methods:

    predict
    predict_batch
    """

    def __init__(self):
        """Construct the predictor."""
        BaseAlgorithm.__init__(self)

    @abstractmethod
    def predict(self, user: int, item: int) -> float:
        """Compute a prediction for the specified user and item.

        Args:
            user: the user ID.
            item: the item ID.

        Returns:
            the prediction rating.
        """
        raise NotImplementedError

    def predict_batch(self, user_item_pairs: pd.DataFrame) -> pd.DataFrame:
        """Compute the predictions for each of the specified user and item pairs.

        Args:
            user_item_pairs: with at least two columns: 'user', 'item'.

        Returns:
            dataFrame with the columns: 'user', 'item', 'prediction'.
        """
        pairs = user_item_pairs[['user', 'item']]
        pairs['prediction'] = np.zeros(len(pairs))
        for i, row in pairs.iterrows():
            pairs.at[i, 'prediction'] = self.predict(
                row['user'],
                row['item']
            )

        return pairs


class Predictor(BasePredictor, metaclass=ABCMeta):
    """Predictor that implements basic shared functionality."""

    def __init__(self, name: str, params: Dict[str, Any], num_threads: int):
        """Construct the predictor.

        Args:
            name: the name of the predictor.
            params: the parameters of the predictor.
            num_threads: the max number of threads the predictor can use.
        """
        BasePredictor.__init__(self)
        self.num_threads = num_threads
        self.predictor_name = name
        self.params = params

    def get_name(self) -> str:
        """Get the name of the predictor.

        Returns:
            the predictor name.
        """
        return self.predictor_name

    def get_num_threads(self) -> int:
        """Get the max number of threads the predictor can use.

        Returns:
            the number of threads.
        """
        return self.num_threads

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the predictor.

        Returns:
            the predictor parameters.
        """
        return dict(self.params)
