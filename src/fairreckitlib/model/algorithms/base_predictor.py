"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from .base_algorithm import Algorithm


class Predictor(Algorithm, metaclass=ABCMeta):
    """Base class for FairRecKit predictors.

    A predictor is used for prediction experiments. It can compute predictions
    for any user and item that it was trained on.
    """

    @abstractmethod
    def predict(self, user, item):
        """Compute a prediction for the specified user and item.

        Args:
            user(int): the user ID.
            item(int): the item ID.

        Returns:
            float: the prediction rating.
        """
        raise NotImplementedError

    def predict_batch(self, user_item_pairs):
        """Computes the predictions for each of the specified user and item pairs.

        Args:
            user_item_pairs(pandas.DataFrame): with at least two columns:
                'user', 'item'.

        Returns:
            pandas.DataFrame with the columns 'user', 'item', 'prediction'.
        """
        pairs = user_item_pairs[['user', 'item']]
        pairs['prediction'] = np.zeros(len(pairs))
        for i, row in pairs.iterrows():
            pairs.at[i, 'prediction'] = self.predict(
                row['user'],
                row['item']
            )

        return pairs
