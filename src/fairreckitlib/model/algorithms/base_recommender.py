"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base_algorithm import BaseAlgorithm


class BaseRecommender(BaseAlgorithm, metaclass=ABCMeta):
    """Base class for FairRecKit recommenders.

    A recommender is used for recommender experiments. It computes a number of
    item recommendations for any user that it was trained on.
    Derived recommenders are expected to implement the abstract interface.

    Abstract methods:

    on_recommend
    on_recommend_batch (optional)

    Public methods:

    has_rated_items_filter
    recommend
    recommend_batch
    """

    def __init__(self, rated_items_filter: bool):
        """Construct the recommender.

        Args:
            rated_items_filter: whether to filter already rated items when
                producing item recommendations.
        """
        BaseAlgorithm.__init__(self)
        self.rated_items_filter = rated_items_filter

    def has_rated_items_filter(self) -> bool:
        """Get if the recommender filters already rated items when producing recommendations.

        Returns:
            whether the recommender filters already rated items.
        """
        return self.rated_items_filter

    def recommend(self, user: int, num_items: int=10) -> pd.DataFrame:
        """Compute item recommendations for the specified user.

        A recommendation is impossible when the user is not present in
        the unique users it was trained on and will return an empty dataframe.

        Args:
            user: the user ID to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            a dataframe with the columns: 'item' and 'score'.
        """
        if user not in self.users:
            return pd.DataFrame(columns=['item', 'score'])

        return self.on_recommend(user, num_items)

    @abstractmethod
    def on_recommend(self, user: int, num_items: int) -> pd.DataFrame:
        """Compute item recommendations for the specified user.

        The user is assumed to be present in the train set that the
        recommender was trained on.
        Derived implementations are expected to return a dataframe
        with the 'score' column in descending order.

        Args:
            user: the user ID to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            a dataframe with the columns: 'item' and 'score'.
        """
        raise NotImplementedError()

    def recommend_batch(self, users: List[int], num_items: int=10) -> pd.DataFrame:
        """Compute the items recommendations for each of the specified users.

        All the users that are not present in the train set that the recommender
        was trained on are filtered before recommendations are made.

        Args:
            users: the user ID's to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            a dataframe with the columns: 'rank', 'user', 'item', 'score'.
        """
        users = [u for u in users if u in self.users]
        if len(users) == 0:
            return pd.DataFrame(columns=['rank', 'user', 'item', 'score'])

        return self.on_recommend_batch(users, num_items)

    def on_recommend_batch(self, users: List[int], num_items: int) -> pd.DataFrame:
        """Compute the items recommendations for each of the specified users.

        All the users are assumed to be present in the train set that
        the recommender was trained on.
        A standard batch implementation is provided, but derived classes are
        allowed to override batching with their own logic.

        Args:
            users: the user ID's to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            a dataframe with the columns: 'rank', 'user', 'item', 'score'.
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


class Recommender(BaseRecommender, metaclass=ABCMeta):
    """Recommender that implements basic shared functionality."""

    def __init__(self, name: str, params: Dict[str, Any], num_threads: int,
                 rated_items_filter: bool):
        """Construct the recommender.

        Args:
            name: the name of the recommender.
            params: the parameters of the recommender.
            num_threads: the max number of threads the recommender can use.
            rated_items_filter: whether to filter already rated items when
                producing item recommendations.
        """
        BaseRecommender.__init__(self, rated_items_filter)
        self.num_threads = num_threads
        self.recommender_name = name
        self.params = params

    def get_name(self) -> str:
        """Get the name of the recommender.

        Returns:
            the recommender name.
        """
        return self.recommender_name

    def get_num_threads(self) -> int:
        """Get the max number of threads the recommender can use.

        Returns:
            the number of threads.
        """
        return self.num_threads

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the recommender.

        Returns:
            the recommender parameters.
        """
        return dict(self.params)
