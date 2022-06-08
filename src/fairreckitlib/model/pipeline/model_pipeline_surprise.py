"""This module contains the model pipelines for the Surprise package.

Classes:

    PredictionPipelineSurprise: prediction pipeline that uses a surprise matrix.
    RecommendationPipelineSurprise: recommendation pipeline that uses a surprise matrix.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..algorithms.surprise.surprise_matrix import MatrixSurprise
from ..algorithms.matrix import Matrix
from .prediction_pipeline import PredictionPipeline
from .recommendation_pipeline import RecommendationPipeline


class PredictionPipelineSurprise(PredictionPipeline):
    """Prediction Pipeline implementation for a surprise matrix train set."""

    def on_load_train_set_matrix(self) -> Matrix:
        """Load the train set matrix that all models can use for training.

        Raises:
            FileNotFoundError: when the train set file is not found.

        Returns:
            the loaded surprise train set matrix.
        """
        return MatrixSurprise(
            self.data_transition.train_set_path,
            self.data_transition.rating_scale
        )


class RecommendationPipelineSurprise(RecommendationPipeline):
    """Recommendation Pipeline implementation for a surprise matrix train set."""

    def on_load_train_set_matrix(self) -> Matrix:
        """Load the train set matrix that all models can use for training.

        Raises:
            FileNotFoundError: when the train set file is not found.

        Returns:
            the loaded surprise train set matrix.
        """
        return MatrixSurprise(
            self.data_transition.train_set_path,
            self.data_transition.rating_scale
        )
