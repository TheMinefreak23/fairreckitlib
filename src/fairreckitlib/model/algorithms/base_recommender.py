"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from .base_algorithm import Algorithm


class Recommender(Algorithm, metaclass=ABCMeta):
    """Base class for FairRecKit recommenders.

    A recommender is used for recommender experiments. It can compute a number of
    item recommendations for any user that it was trained on.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.
    """
    def __init__(self, **kwargs):
        Algorithm.__init__(self, **kwargs)
        self.rated_items_filter = kwargs['rated_items_filter']

    @abstractmethod
    def recommend(self, user, num_items=10):
        """Computes item recommendations for the specified user.

        Args:
            user(int): the user ID.
            num_items(int): the number of item recommendations to produce.

        Returns:
            pandas.DataFrame: with the columns 'item' and 'score'.
        """
        raise NotImplementedError()

    def recommend_batch(self, users, num_items=10):
        """Computes the items recommendations for each of the specified users.

        Args:
            users(array-like): the user ID's.
            num_items(int): the number of item recommendations to produce.

        Returns:
            pandas.DataFrame: with the columns 'rank', 'user', 'item', 'score'.
        """
        result = pd.DataFrame()

        for _, user in enumerate(users):
            item_scores = self.recommend(user, num_items)

            item_scores['rank'] = np.arange(1, 1 + num_items)
            item_scores['user'] = np.full(num_items, user)

            result = result.append(
                item_scores[['rank', 'user', 'item', 'score']],
                ignore_index=True
            )

        return result
