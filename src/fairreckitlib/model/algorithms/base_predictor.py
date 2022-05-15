"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base_algorithm import BaseAlgorithm


class BasePredictor(BaseAlgorithm, metaclass=ABCMeta):
    """Base class for FairRecKit predictors.

    A predictor is used for prediction experiments. It computes predictions
    for any user and item that it was trained on.
    Derived predictors are expected to implement the abstract interface.

    Abstract methods:

    on_predict
    on_predict_batch (optional)

    Public methods:

    predict
    predict_batch
    """

    def __init__(self):
        """Construct the predictor."""
        BaseAlgorithm.__init__(self)

    def predict(self, user: int, item: int) -> float:
        """Compute a prediction for the specified user and item.

        A prediction is impossible when the user and/or item is not
        present in the unique users and/or items it was trained on.
        Moreover, the prediction could also fail in the derived
        implementation of the predictor.

        Args:
            user: the user ID.
            item: the item ID.

        Returns:
            the predicted rating or NaN when impossible.
        """
        if user not in self.users:
            return math.nan
        if item not in self.items:
            return math.nan

        return self.on_predict(user, item)

    @abstractmethod
    def on_predict(self, user: int, item: int) -> float:
        """Compute a prediction for the specified user and item.

        The user and item are assumed to be present in the train
        set that the predictor was trained on.
        Derived implementations are allowed to return NaN when the
        prediction is impossible to compute.

        Args:
            user: the user ID.
            item: the item ID.

        Returns:
            the predicted rating or NaN when impossible.
        """
        raise NotImplementedError()

    def predict_batch(self, user_item_pairs: pd.DataFrame) -> pd.DataFrame:
        """Compute the predictions for each of the specified user and item pairs.

        All the users and items in the pairs that are not present in the train set that
        the predictor was trained on are filtered before predictions are made.

        Args:
            user_item_pairs: with at least two columns: 'user' and 'item'.

        Returns:
            a dataFrame with the columns: 'user', 'item', 'prediction'.
        """
        user_item_pairs = user_item_pairs[['user', 'item']]
        user_item_pairs = user_item_pairs[user_item_pairs['user'].isin(self.users)]
        user_item_pairs = user_item_pairs[user_item_pairs['item'].isin(self.items)]
        if len(user_item_pairs) == 0:
            return user_item_pairs

        return self.on_predict_batch(user_item_pairs)

    def on_predict_batch(self, user_item_pairs: pd.DataFrame) -> pd.DataFrame:
        """Compute the predictions for each of the specified user and item pairs.

        All the users and items in the pairs are assumed to be present in the train
        set that the predictor was trained on.
        A standard batch implementation is provided, but derived classes are
        allowed to override batching with their own logic.

        Args:
            user_item_pairs: with two columns: 'user' and 'item'.

        Returns:
            a dataFrame with the columns: 'user', 'item', 'prediction'.
        """
        user_item_pairs['prediction'] = np.zeros(len(user_item_pairs))
        for i, row in user_item_pairs.iterrows():
            user_item_pairs.at[i, 'prediction'] = self.predict(
                row['user'],
                row['item']
            )

        return user_item_pairs


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
