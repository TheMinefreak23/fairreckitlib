"""This module contains a model pipeline that predicts known item ratings.

Classes:

    PredictionPipeline: can batch predictions from multiple models for a specific API.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import List

import pandas as pd

from ..algorithms.base_predictor import BasePredictor
from .model_pipeline import ModelPipeline


class PredictionPipeline(ModelPipeline):
    """Prediction Pipeline that computes user/item rating predictions.

    The (user,item) prediction will be computed and for each pair that is present in the test set.
    """

    def load_test_set_users(self) -> None:
        """Load the test set users that all models can use for testing.

        Predictions are made for every user-item pair in the test set.

        Raises:
            FileNotFoundError: when the test set file is not found.
        """
        self.test_set_users = self.load_test_set_dataframe('prediction test set')

    def test_model_ratings(
            self,
            model: BasePredictor,
            user_batch: List[int],
            **kwargs) -> pd.DataFrame:
        """Test the specified model for rating predictions.

        Predict ratings for each user-item pair that is present in the test set.

        Args:
            model: the model that needs to be tested.
            user_batch: the user batch to compute model ratings for.

        Raises:
            ArithmeticError: possibly raised by a predictor model on testing.
            MemoryError: possibly raised by a predictor model on testing.
            RuntimeError: possibly raised by a predictor model on testing.

        Returns:
            a dataframe containing the computed rating predictions.
        """
        return model.predict_batch(user_batch)
