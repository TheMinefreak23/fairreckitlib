"""This module contains a model pipeline that recommends items based on rating predictions.

Classes:

    RecommendationPipeline: can batch recommendations from multiple models for a specific API.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable

import pandas as pd

from ..algorithms.base_recommender import BaseRecommender
from .model_pipeline import ModelPipeline


class RecommendationPipeline(ModelPipeline):
    """Recommendation Pipeline that computes item recommendations.

    The topK item recommendations will be computed for each user that is present in the test set.
    """

    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Get the dataframe that contains the original ratings.

        For the recommendation pipeline the ratings is the combination of the train and test set.

        Returns:
            dataframe containing the 'user', 'item', 'rating', columns.
        """
        return pd.concat([self.train_set, self.test_set])

    def test_model_ratings(self,
                           model: BaseRecommender,
                           output_path: str,
                           batch_size: int,
                           is_running: Callable[[], bool],
                           **kwargs) -> None:
        """Test the specified model for rating recommendations.

        Produce a top K number of item scores for each user that is present in the test set.

        Args:
            model: the model that needs to be tested.
            output_path: path to the file where the ratings will be stored.
            batch_size: number of users to test ratings for in a batch.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce.
        """
        test_users = self.test_set.user.unique()
        start_index = 0
        while start_index < len(test_users):
            if not is_running():
                return

            user_batch = test_users[start_index : start_index + batch_size]
            recommendations = model.recommend_batch(user_batch, num_items=kwargs['num_items'])
            if not is_running():
                return

            self.write_dataframe(output_path, recommendations, start_index == 0)
            start_index += batch_size
