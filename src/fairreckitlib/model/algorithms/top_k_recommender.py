"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .base_predictor import BasePredictor
from .base_recommender import BaseRecommender


class TopK(BaseRecommender):
    """Recommender that implements top K recommendations using a predictor."""

    def __init__(self, predictor: BasePredictor, rated_items_filter: bool):
        """Construct the TopK recommender.

        Args:
            predictor: the underlying predictor to use for recommendations.
            rated_items_filter: whether to filter already rated items when
                producing item recommendations.
        """
        BaseRecommender.__init__(self, rated_items_filter)
        self.predictor = predictor

    def get_name(self) -> str:
        """Get the name of the underlying predictor.

        Returns:
            the name of the underlying predictor.
        """
        return self.predictor.get_name()

    def get_num_threads(self) -> int:
        """Get the max number of threads the underlying predictor can use.

        Returns:
            the number of threads.
        """
        return self.predictor.get_num_threads()

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the underlying predictor.

        Returns:
            the parameters of the underlying predictor.
        """
        return self.predictor.get_params()

    def get_items(self) -> Optional[List[int]]:
        """Get the (unique) items the underlying predictor was trained on.

        Returns:
            a list of unique item IDs or None if the algorithm is not trained yet.
        """
        return self.predictor.get_items()

    def get_users(self) -> Optional[List[int]]:
        """Get the (unique) users the underlying predictor was trained on.

        Returns:
            a list of unique user IDs or None if the algorithm is not trained yet.
        """
        return self.predictor.get_users()

    def on_train(self) -> None:
        """Train the underlying predictor on the train set."""
        self.predictor.train(self.train_set)

    def on_recommend(self, user: int, num_items: int) -> pd.DataFrame:
        """Compute item recommendations using the underlying predictor.

        Go through all user-item combinations for the specified user and
        predict a score. Sort in descending order and return the topK items.

        Args:
            user: the user ID to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            dataframe with the columns: 'item' and 'score'.
        """
        # filter items that are rated by the user already
        if self.rated_items_filter:
            is_user = self.train_set['user'] == user
            user_item_ratings = self.train_set.loc[is_user]['item'].tolist()
            items = [i for i in self.items if i not in user_item_ratings]

        # TODO this is not very efficient, but works (also should utilize available num_threads)
        # compute recommendations for all items and truncate to the top num_items
        item_ratings = list(map(lambda i: (i, self.predictor.predict(user, i)), self.items))
        item_ratings.sort(key=lambda i: i[1], reverse=True)

        return pd.DataFrame(item_ratings[:num_items], columns=['item', 'score'])
