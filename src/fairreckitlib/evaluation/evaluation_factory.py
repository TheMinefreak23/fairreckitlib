"""This module contains functionality to create an evaluation factory.

Constants:

    KEY_EVALUATION: key that is used to identify evaluation.

Functions:

    create_evaluation_factory: create factory with prediction/recommendation factories.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.config.config_factories import GroupFactory
from ..core.core_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from .metrics.metric_factory import \
    create_accuracy_metric_factory, create_coverage_metric_factory, create_rating_metric_factory

KEY_EVALUATION = 'evaluation'


def create_evaluation_factory() -> GroupFactory:
    """Create a factory with all predictor and recommender metric category factories.

    All the metric category factories are shared between prediction and recommendation,
    except for ths accuracy category which only applies to recommendation evaluation.

    Returns:
        the group factory with available predictor and recommender factories.
    """
    shared_categories = [create_coverage_metric_factory, create_rating_metric_factory]

    prediction_factory = GroupFactory(TYPE_PREDICTION)

    recommendation_factory = GroupFactory(TYPE_RECOMMENDATION)
    recommendation_factory.add_factory(create_accuracy_metric_factory())

    for func_create in shared_categories:
        category_factory = func_create()
        prediction_factory.add_factory(category_factory)
        recommendation_factory.add_factory(category_factory)

    evaluation_factory = GroupFactory(KEY_EVALUATION)
    evaluation_factory.add_factory(prediction_factory)
    evaluation_factory.add_factory(recommendation_factory)
    return evaluation_factory
