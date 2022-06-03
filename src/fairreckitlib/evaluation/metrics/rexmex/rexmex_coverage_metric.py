"""This module contains the rexmex coverage metric and creation functions.

Classes:

    RexmexCoverageMetric: coverage metric implementation for rexmex.

Functions:

    create_item_coverage: create the Item coverage metric (factory creation compatible).
    create_user_coverage: create the User coverage metric (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, Dict

import pandas as pd
from rexmex.metrics import item_coverage, user_coverage

from ...evaluation_sets import EvaluationSets
from ..metric_base import ColumnMetric


class RexmexCoverageMetric(ColumnMetric):
    """Coverage metric implementation for the Rexmex framework."""

    def __init__(
            self,
            name: str,
            params: Dict[str, Any],
            eval_func: Callable[[pd.Series, pd.Series], float]):
        """Construct the Rexmex coverage metric.

        Args:
            name: the name of the metric.
            params: the parameters of the metric.
            eval_func: the function that uses the user and item columns to compute the evaluation.
        """
        ColumnMetric.__init__(
            self,
            name,
            params,
            eval_func,
            requires_test_set=False,
            requires_train_set=True
        )

    def on_evaluate(self, eval_sets: EvaluationSets) -> float:
        """Evaluate the sets for the performance of the metric.

        Args:
            eval_sets: the sets to use for computing the performance of the metric.

        Returns:
            the evaluated performance.
        """
        # Convert recommended user, item columns to list of tuples.
        tuple_recs = [tuple(r) for r in eval_sets.ratings[['user', 'item']].to_numpy()]
        # The possible users and items are in the train set
        possible_users_items = (eval_sets.train['user'], eval_sets.train['item'])
        return self.eval_func(possible_users_items, tuple_recs)


def create_item_coverage(name: str, params: Dict[str, Any], **_) -> RexmexCoverageMetric:
    """Create the Item coverage metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the RexmexCoverageMetric wrapper of Item coverage.
    """
    return RexmexCoverageMetric(name, params, item_coverage)


def create_user_coverage(name: str, params: Dict[str, Any], **_) -> RexmexCoverageMetric:
    """Create the User coverage metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the RexmexCoverageMetric wrapper of User coverage.
    """
    return RexmexCoverageMetric(name, params, user_coverage)
