"""This module contains a model pipeline that predicts known item ratings.

Classes:

    PredictionPipeline: can batch predictions from multiple models for a specific API.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable

import pandas as pd

from ..algorithms.base_predictor import BasePredictor
from .model_pipeline import ModelPipeline, write_computed_ratings


class PredictionPipeline(ModelPipeline):
    """Prediction Pipeline that computes user/item rating predictions.

    The (user,item) prediction will be computed and for each pair that is present in the test set.
    """

    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Get the dataframe that contains the original ratings.

        For the prediction pipeline only the ratings from the test set are necessary.

        Returns:
            dataframe containing the 'user', 'item', 'rating', columns.
        """
        return self.test_set

    def test_model_ratings(self,
                           model: BasePredictor,
                           output_path: str,
                           batch_size: int,
                           is_running: Callable[[], bool],
                           **kwargs) -> None:
        """Test the specified model for rating predictions.

        Predict ratings for each user-item pair that is present in the test set.

        Args:
            model: the model that needs to be tested.
            output_path: path to the file where the ratings will be stored.
            batch_size: number of users to test ratings for in a batch.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.
        """
        start_index = 0
        while start_index < len(self.test_set):
            if not is_running():
                return

            user_batch = self.test_set[start_index : start_index + batch_size]
            predictions = model.predict_batch(user_batch)
            if not is_running():
                return

            write_computed_ratings(self.event_dispatcher, output_path, predictions,
                start_index == 0)
            start_index += batch_size
