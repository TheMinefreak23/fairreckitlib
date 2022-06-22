"""This module tests the metric interface of all available metrics.

Classes:

    DummyMetric: dummy metric implementation to test various errors.

Functions:

    test_metric_interface_construction_and_error: test metric interface construction and error.
    test_metric_creation: test creation and parameters creation to match for all metrics.
    test_metric_evaluation: test the evaluation of all metrics for various evaluation sets.
    assert_metric_evaluation: assert the evaluation of all metrics with an evaluation set.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
import os
from typing import Any, Dict

import pandas as pd
import pytest

from src.fairreckitlib.core.core_constants import DEFAULT_TOP_K, MODEL_RATINGS_FILE, \
    TYPE_PREDICTION, TYPE_RECOMMENDATION, VALID_TYPES
from src.fairreckitlib.evaluation.evaluation_factory import create_evaluation_factory
from src.fairreckitlib.evaluation.evaluation_sets import EvaluationSets
from src.fairreckitlib.evaluation.metrics.metric_base import BaseMetric
from src.fairreckitlib.evaluation.metrics.metric_constants import MetricCategory, KEY_METRIC_PARAM_K

eval_factory = create_evaluation_factory()


class DummyMetric(BaseMetric):
    """Dummy metric to test various errors."""

    def __init__(self, name: str, params: Dict[str, Any], **kwargs):
        """Construct dummy metric."""
        BaseMetric.__init__(self, name, params,
                            requires_train_set=kwargs['requires_train_set'],
                            requires_test_set=kwargs['requires_test_set'])
        self.eval_error = kwargs.get('eval_error')
        if kwargs.get('const_error', False):
            raise kwargs['const_error']()

    def on_evaluate(self, eval_sets: EvaluationSets) -> float:
        """Raise error."""
        if self.eval_error is not None:
            raise self.eval_error()

        return BaseMetric.on_evaluate(self, eval_sets)


def test_metric_interface_construction_and_error() -> None:
    """Test metric interface construction and error for not implemented function."""
    metric = DummyMetric('metric', {}, requires_test_set=True, requires_train_set=False)

    assert metric.requires_test_set, \
        'expected metric to require the test set after construction'
    assert not metric.requires_train_set, \
        'did not expect metric to require the train set after construction'

    # test failure for not implemented interface
    pytest.raises(NotImplementedError, metric.on_evaluate, None)


@pytest.mark.parametrize('eval_type', VALID_TYPES)
def test_metric_creation(eval_type: str) -> None:
    """Test creation and parameters creation to match for all metrics."""
    eval_type_factory = eval_factory.get_factory(eval_type)

    # check existence of accuracy metric category
    has_accuracy_metrics = MetricCategory.ACCURACY.value in eval_type_factory.get_available_names()
    if eval_type == TYPE_RECOMMENDATION:
        assert has_accuracy_metrics, \
            'expected recommendation metrics to have an accuracy category'
    elif eval_type == TYPE_PREDICTION:
        assert not has_accuracy_metrics, \
            'did not expect prediction metrics to have an accuracy category'
    else:
        raise TypeError('Unknown evaluation type')

    for metric_category_name in eval_type_factory.get_available_names():
        metric_category_factory = eval_type_factory.get_factory(metric_category_name)

        for metric_name in metric_category_factory.get_available_names():
            metric_params = metric_category_factory.create_params(metric_name).get_defaults()

            # check name / params
            metric = metric_category_factory.create(metric_name)
            assert metric.get_name() == metric_name, \
                'expected metric name to be the same after creation'
            assert metric.get_params() == metric_params, \
                'expected metric params to be the same after creation'

            if metric.get_params().get(KEY_METRIC_PARAM_K, False):
                assert metric.get_params()[KEY_METRIC_PARAM_K] == DEFAULT_TOP_K, \
                    'expected K param to be the default top K'
                assert metric_category_name == MetricCategory.ACCURACY.value, \
                    'expected metric that has K param to be categorized as accuracy'

            # check required sets
            if metric_category_name == MetricCategory.COVERAGE.value:
                assert not metric.requires_test_set, \
                    'did not expect coverage metric to need a test set'
                assert metric.requires_train_set, \
                    'expected coverage metric to need a train set'
            else:
                assert metric.requires_test_set, \
                    'expected metric to need a test set'
                assert not metric.requires_train_set, \
                    'did not expect metric to need a train set'


@pytest.mark.parametrize('eval_type', VALID_TYPES)
def test_metric_evaluation(eval_type: str) -> None:
    """Test the evaluation of all available metrics for various evaluation sets."""
    eval_set_dir = os.path.join('tests', 'evaluation_sets', eval_type)

    for data_dir in os.listdir(eval_set_dir):
        data_dir = os.path.join(eval_set_dir, data_dir)
        if not os.path.isdir(data_dir):
            continue

        train_set = pd.read_table(os.path.join(data_dir, 'train_set.tsv'),
                                  names=['user', 'item', 'rating'])
        test_set = pd.read_table(os.path.join(data_dir, 'test_set.tsv'),
                                 names=['user', 'item', 'rating'])

        for model_dir in os.listdir(data_dir):
            model_dir = os.path.join(data_dir, model_dir)
            if not os.path.isdir(model_dir):
                continue

            rating_set = pd.read_table(os.path.join(model_dir, MODEL_RATINGS_FILE))

            assert_metric_evaluation(eval_type, EvaluationSets(rating_set, train_set, test_set))


def assert_metric_evaluation(eval_type: str, eval_sets: EvaluationSets) -> None:
    """Assert the evaluation of all available metrics with the specified evaluation sets."""
    eval_type_factory = eval_factory.get_factory(eval_type)

    for metric_category_name in eval_type_factory.get_available_names():
        metric_category_factory = eval_type_factory.get_factory(metric_category_name)

        for metric_name in metric_category_factory.get_available_names():
            metric = metric_category_factory.create(metric_name)

            evaluation = metric.evaluate(eval_sets)
            assert isinstance(evaluation, float) and not math.isnan(evaluation), \
                'expected metric to compute (floating-point) evaluation'

            if metric.requires_train_set:
                # test failure when the metric needs the train set for evaluation
                eval_no_train = EvaluationSets(eval_sets.ratings, None, eval_sets.test)
                pytest.raises(RuntimeError, metric.evaluate, eval_no_train)

            if metric.requires_test_set:
                # test failure when the metric needs the test set for evaluation
                eval_no_test = EvaluationSets(eval_sets.ratings, eval_sets.train, None)
                pytest.raises(RuntimeError, metric.evaluate, eval_no_test)
