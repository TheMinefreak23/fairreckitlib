"""This module contains the lenskit rating metric and creation functions.

Classes:

    LensKitRatingMetric: rating metric implementation for lenskit.

Functions:

    create_mae: create the MAE rating metric (factory creation compatible).
    create_rmse: create the RMSE rating metric (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

from lenskit.metrics import predict
import pandas as pd

from ...evaluation_sets import EvaluationSets
from ..metric_base import ColumnMetric


class LensKitRatingMetric(ColumnMetric):
    """Rating metric implementation for the LensKit framework."""

    def on_evaluate(self, eval_sets: EvaluationSets) -> float:
        """Evaluate the sets for the performance of the metric.

        Args:
            eval_sets: the sets to use for computing the performance of the metric.

        Returns:
            the evaluated performance.
        """
        lenskit_ratings = eval_sets.ratings.drop('rating', axis=1)
        score_column = 'score' if 'score' in lenskit_ratings else 'prediction'
        scores = pd.merge(eval_sets.test, lenskit_ratings, how='left', on=['user', 'item'])
        return predict.user_metric(scores, score_column=score_column, metric=self.eval_func)


def create_mae(name: str, params: Dict[str, Any], **_) -> LensKitRatingMetric:
    """Create the MAE rating metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the LensKitAccuracyMetric wrapper of MAE.
    """
    return LensKitRatingMetric(name, params, predict.mae)


def create_rmse(name: str, params: Dict[str, Any], **_) -> LensKitRatingMetric:
    """Create the RMSE rating metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the LensKitAccuracyMetric wrapper of RMSE.
    """
    return LensKitRatingMetric(name, params, predict.rmse)
