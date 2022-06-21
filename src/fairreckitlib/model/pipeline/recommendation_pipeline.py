"""This module contains a model pipeline that recommends items based on rating predictions.

Classes:

    RecommendationPipeline: can batch recommendations from multiple models for a specific API.
    RecommendationPipelineCSR: recommendation pipeline with a csr matrix instead of a dataframe.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import List

import pandas as pd

from ..algorithms.base_recommender import BaseRecommender
from ..algorithms.matrix import Matrix, MatrixCSR
from .model_pipeline import ModelPipeline


class RecommendationPipeline(ModelPipeline):
    """Recommendation Pipeline that computes item recommendations.

    The topK item recommendations will be computed for each user that is present in the test set.
    """

    def load_test_set_users(self) -> None:
        """Load the test set users that all models can use for testing.

        Recommendations are made for every user in the test set.

        Raises:
            FileNotFoundError: when the test set file is not found.
        """
        test_set = self.load_test_set_dataframe('recommendation test set')
        self.test_set_users = test_set['user'].unique()

    def test_model_ratings(
            self,
            model: BaseRecommender,
            user_batch: List[int],
            **kwargs) -> pd.DataFrame:
        """Test the specified model for rating recommendations.

        Produce a top K number of item scores for each user that is present in the test set.

        Args:
            model: the model that needs to be tested.
            user_batch: the user batch to compute model ratings for.

        Keyword Args:
            num_items(int): the number of item recommendations to produce.

        Raises:
            ArithmeticError: possibly raised by a recommender model on testing.
            MemoryError: possibly raised by a recommender model on testing.
            RuntimeError: possibly raised by a recommender model on testing.

        Returns:
            a dataframe containing the computed item recommendations.
        """
        return model.recommend_batch(user_batch, num_items=kwargs['num_items'])


class RecommendationPipelineCSR(RecommendationPipeline):
    """Recommendation Pipeline implementation for a CSR matrix train set."""

    def on_load_train_set_matrix(self) -> Matrix:
        """Load the train set matrix that all models can use for training.

        Raises:
            FileNotFoundError: when the train set file is not found.

        Returns:
            the loaded train set csr matrix.
        """
        return MatrixCSR(self.data_transition.train_set_path)
