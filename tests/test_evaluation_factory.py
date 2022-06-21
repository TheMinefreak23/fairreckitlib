"""This module tests the evaluation (type/category) factories.

Functions:

    test_evaluation_factory: test metrics and factories to be derived from the correct base class.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.core.config.config_factories import Factory, GroupFactory
from src.fairreckitlib.core.core_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.evaluation.evaluation_factory import create_evaluation_factory
from src.fairreckitlib.evaluation.metrics.metric_base import BaseMetric


def test_evaluation_factory() -> None:
    """Test metrics and factories in the factory to be derived from the correct base class."""
    eval_factory = create_evaluation_factory()
    assert isinstance(eval_factory, GroupFactory), 'expected evaluation group factory.'

    assert bool(eval_factory.get_factory(TYPE_PREDICTION)), 'missing prediction metrics.'
    assert bool(eval_factory.get_factory(TYPE_RECOMMENDATION)), 'missing recommender metrics.'

    for eval_type in eval_factory.get_available_names():
        eval_type_factory = eval_factory.get_factory(eval_type)
        assert isinstance(eval_type_factory, GroupFactory), 'expected category group factory.'

        for category_name in eval_type_factory.get_available_names():
            category_factory = eval_type_factory.get_factory(category_name)
            assert isinstance(category_factory, Factory), 'expected metric factory.'

            for metric_name in category_factory.get_available_names():
                metric = category_factory.create(metric_name)

                assert isinstance(metric, BaseMetric), 'expected base metric.'
