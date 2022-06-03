"""This module contains the rexmex rating metric and creation functions.

Classes:

    RexmexRatingMetric: rating metric implementation for rexmex.

Functions:

    create_mape: create the MAPE rating metric (factory creation compatible).
    create_mse: create the MSE rating metric (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

import pandas as pd
from rexmex.metrics import mean_absolute_percentage_error, mean_squared_error

from ...evaluation_sets import EvaluationSets
from ..metric_base import ColumnMetric


class RexmexRatingMetric(ColumnMetric):
    """Rating metric implementation for the Rexmex framework."""

    def on_evaluate(self, eval_sets: EvaluationSets) -> float:
        """Evaluate the sets for the performance of the metric.

        Args:
            eval_sets: the sets to use for computing the performance of the metric.

        Returns:
            the evaluated performance.
        """
        eval_sets.ratings.drop('rating', inplace=True, axis=1)
        score_column = 'score' if 'score' in eval_sets.ratings else 'prediction'
        scores = pd.merge(eval_sets.test, eval_sets.ratings, how='left', on=['user', 'item'])
        scores.dropna(subset=[score_column], axis=0, inplace=True)
        return self.eval_func(scores['rating'], scores[score_column])


def create_mape(name: str, params: Dict[str, Any], **_) -> RexmexRatingMetric:
    """Create the MAPE rating metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the RexmexRatingMetric wrapper of MAPE.
    """
    return RexmexRatingMetric(name, params, mean_absolute_percentage_error)


def create_mse(name: str, params: Dict[str, Any], **_) -> RexmexRatingMetric:
    """Create the MSE rating metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the RexmexRatingMetric wrapper of MSE.
    """
    return RexmexRatingMetric(name, params, mean_squared_error)
