"""This module contains the lenskit accuracy metric and creation functions.

Classes:

    LensKitAccuracyMetric: accuracy metric implementation for lenskit.

Functions:

    create_ndcg: create the NDCG@K accuracy metric (factory creation compatible).
    create_hit_ratio: create the HR@K accuracy metric (factory creation compatible).
    create_precision: create the P@K accuracy metric (factory creation compatible).
    create_recall: create the R@K accuracy metric (factory creation compatible).
    create_mean_recip_rank: create the MRR accuracy metric (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Callable, Dict

from lenskit import topn
import pandas as pd

from ...evaluation_sets import EvaluationSets
from ..metric_base import BaseMetric
from ..metric_constants import KEY_METRIC_PARAM_K


class LensKitAccuracyMetric(BaseMetric):
    """Accuracy metric implementation for the LensKit framework."""

    def __init__(
            self,
            name: str,
            params: Dict[str, Any],
            eval_func: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
            group: str):
        """Construct the lenskit accuracy metric.

        Args:
            name: the name of the metric.
            params: the parameters of the metric.
            eval_func: the lenskit evaluation function.
            group: the group name of the lenskit evaluation function.
        """
        BaseMetric.__init__(self, name, params)
        self.eval_func = eval_func
        self.group = group

    def on_evaluate(self, eval_sets: EvaluationSets) -> float:
        """Evaluate the sets for the performance of the metric.

        Args:
            eval_sets: the sets to use for computing the performance of the metric.

        Returns:
            the evaluated performance.
        """
        # Drop the rating column as it is not needed and will be used by lenskit internally
        eval_sets.ratings.drop('rating', inplace=True, axis=1)
        # Lenskit needs this column
        eval_sets.ratings['Algorithm'] = 'APPROACHNAME'

        analysis = topn.RecListAnalysis()
        k = self.params.get(KEY_METRIC_PARAM_K)
        if k:
            analysis.add_metric(self.eval_func, k=k)
        else:
            analysis.add_metric(self.eval_func)

        results = analysis.compute(eval_sets.ratings, eval_sets.test)
        return results.groupby('Algorithm')[self.group].mean()[0]


def create_ndcg(name: str, params: Dict[str, Any], **_) -> LensKitAccuracyMetric:
    """Create the NDCG@K accuracy metric.

    Args:
        name: the name of the metric.
        params: containing the following name-value pairs:
            K(int): the number of item recommendations to test on.

    Returns:
        the LensKitAccuracyMetric wrapper of NDCG@K.
    """
    return LensKitAccuracyMetric(name, params, topn.ndcg, 'ndcg')


def create_hit_ratio(name: str, params: Dict[str, Any], **_) -> LensKitAccuracyMetric:
    """Create the HR@K accuracy metric.

    Args:
        name: the name of the metric.
        params: containing the following name-value pairs:
            K(int): the number of item recommendations to test on.

    Returns:
        the LensKitAccuracyMetric wrapper of HR@K.
    """
    return LensKitAccuracyMetric(name, params, topn.hit, 'hit')


def create_precision(name: str, params: Dict[str, Any], **_) -> LensKitAccuracyMetric:
    """Create the P@K accuracy metric.

    Args:
        name: the name of the metric.
        params: containing the following name-value pairs:
            K(int): the number of item recommendations to test on.

    Returns:
        the LensKitAccuracyMetric wrapper of P@K.
    """
    return LensKitAccuracyMetric(name, params, topn.precision, 'precision')


def create_recall(name: str, params: Dict[str, Any], **_) -> LensKitAccuracyMetric:
    """Create the R@K accuracy metric.

    Args:
        name: the name of the metric.
        params: containing the following name-value pairs:
            K(int): the number of item recommendations to test on.

    Returns:
        the LensKitAccuracyMetric wrapper of R@K.
    """
    return LensKitAccuracyMetric(name, params, topn.recall, 'recall')


def create_mean_recip_rank(name: str, params: Dict[str, Any], **_) -> LensKitAccuracyMetric:
    """Create the MRR accuracy metric.

    Args:
        name: the name of the metric.
        params: there are no parameters for this metric.

    Returns:
        the LensKitAccuracyMetric wrapper of MRR.
    """
    return LensKitAccuracyMetric(name, params, topn.recip_rank, 'recip_rank')
