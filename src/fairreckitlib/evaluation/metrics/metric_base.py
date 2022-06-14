"""This module contains the base class for all metrics.

Classes:

    BaseMetric: the base class for metrics.
    ColumnMetric: metric that uses two columns to compute the evaluation.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd

from ..evaluation_sets import EvaluationSets


class BaseMetric(metaclass=ABCMeta):
    """Base class for FairRecKit metrics.

    A metric is used for evaluating the performance of recommender system experiments.
    Derived metrics are expected to implement the abstract interface.

    Abstract methods:

    on_evaluate

    Public methods:

    evaluate
    get_name
    get_params
    """

    def __init__(
            self,
            name: str,
            params: Dict[str, Any],
            *,
            requires_test_set: bool=True,
            requires_train_set: bool=False):
        """Construct the metric.

        Args:
            name: the name of the metric.
            params: the parameters of the metric.
            requires_test_set: whether the metric requires the test set for evaluation.
            requires_train_set: whether the metric requires the train set for evaluation.
        """
        self.metric_name = name
        self.params = params
        self.requires_test_set = requires_test_set
        self.requires_train_set = requires_train_set

    @abstractmethod
    def on_evaluate(self, eval_sets: EvaluationSets) -> float:
        """Evaluate the sets for the performance of the metric.

        Derived classes should implement the evaluation logic of the metric.

        Args:
            eval_sets: the sets to use for computing the performance of the metric.

        Returns:
            the evaluated performance.
        """
        raise NotImplementedError()

    def evaluate(self, eval_sets: EvaluationSets) -> float:
        """Evaluate the sets for the performance of the metric.

        Args:
            eval_sets: the sets to use for computing the performance of the metric.

        Raises:
            ArithmeticError: possibly raised by a metric on evaluation.
            MemoryError: possibly raised by a metric on evaluation.
            RuntimeError: possibly raised by a metric on evaluation.

        Returns:
            the evaluated performance.
        """
        return self.on_evaluate(eval_sets)

    def get_name(self):
        """Get the name of the metric.

        Returns:
            the metric name.
        """
        return self.metric_name

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the metric.

        Returns:
            the metric parameters.
        """
        return dict(self.params)


class ColumnMetric(BaseMetric, metaclass=ABCMeta):
    """Metric that uses two columns to produce the performance evaluation.

    The intended use of this class is to provide a base implementation for
    computing the evaluation using two pandas Series columns. The actual data
    that is provided in these columns can be anything depending on the derived class.
    """

    def __init__(
            self,
            name: str,
            params: Dict[str, Any],
            eval_func: Callable[[pd.Series, pd.Series], float],
            *,
            requires_test_set: bool=True,
            requires_train_set: bool= False):
        """Construct the column metric.

        Args:
            name: the name of the metric.
            params: the parameters of the metric.
            eval_func: the evaluation function that uses two columns to compute the evaluation.
            requires_test_set: whether the metric requires the test set for evaluation.
            requires_train_set: whether the metric requires the train set for evaluation.
        """
        BaseMetric.__init__(
            self,
            name,
            params,
            requires_test_set=requires_test_set,
            requires_train_set=requires_train_set
        )
        self.eval_func = eval_func
