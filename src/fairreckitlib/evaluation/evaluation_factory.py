"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.config_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from ..core.factories import GroupFactory
from .pipeline.evaluation_config import KEY_EVALUATION
from .metrics.metric_factory import create_accuracy_metric_factory
from .metrics.metric_factory import create_coverage_metric_factory
from .metrics.metric_factory import create_diversity_metric_factory
from .metrics.metric_factory import create_novelty_metric_factory
from .metrics.metric_factory import create_rating_metric_factory


def create_evaluation_factory():
    shared_categories = [
        create_coverage_metric_factory,
        create_diversity_metric_factory,
        create_novelty_metric_factory,
        create_rating_metric_factory,
    ]

    prediction_factory = GroupFactory(TYPE_PREDICTION)
    recommendation_factory = GroupFactory(TYPE_RECOMMENDATION)
    recommendation_factory.add_factory(create_accuracy_metric_factory())
    # TODO document this (shared factory pointers)
    for _, func_create in enumerate(shared_categories):
        category_factory = func_create()
        prediction_factory.add_factory(category_factory)
        recommendation_factory.add_factory(category_factory)

    evaluation_factory = GroupFactory(KEY_EVALUATION)
    evaluation_factory.add_factory(prediction_factory)
    evaluation_factory.add_factory(recommendation_factory)
    return evaluation_factory
