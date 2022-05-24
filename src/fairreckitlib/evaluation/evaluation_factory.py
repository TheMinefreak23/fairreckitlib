"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from typing import Callable

from .pipeline.evaluation_pipeline import EvaluationPipeline
from ..core.config_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from ..core.event_dispatcher import EventDispatcher
from ..core.factories import GroupFactory, Factory
from .metrics.metric_factory import create_accuracy_metric_factory
from .metrics.metric_factory import create_coverage_metric_factory
from .metrics.metric_factory import create_diversity_metric_factory
from .metrics.metric_factory import create_novelty_metric_factory
from .metrics.metric_factory import create_rating_metric_factory

KEY_EVALUATION = 'evaluation'


def create_metric_pipeline_factory(
        metric_factory: Factory,
        create_pipeline: Callable[[Factory, EventDispatcher], EvaluationPipeline]) -> Factory:
    
    metric_factory.create_pipeline = create_pipeline
    return metric_factory


def create_evaluation_factory() -> GroupFactory:
    """Create a factory with all predictor and recommender metric category factories.

    Returns:
        the group factory with available predictor and recommender factories.
    """

    shared_categories = [
        create_coverage_metric_factory,
        create_diversity_metric_factory,
        create_novelty_metric_factory,
        create_rating_metric_factory,
    ]

    prediction_factory = GroupFactory(TYPE_PREDICTION)

    # rexmex predictors
    """
    prediction_factory.add_factory(create_metric_pipeline_factory(
        rexmex_factory.create_predictor_factory(),
        EvaluationPipeline
    ))"""
    """
    # lenskit predictors
    prediction_factory.add_factory(create_metric_pipeline_factory(
        lenskit_factory.create_predictor_factory(),
        EvaluationPipeline
    ))
    """

    recommendation_factory = GroupFactory(TYPE_RECOMMENDATION)

    # rexmex recommenders
    """recommendation_factory.add_factory(create_metric_pipeline_factory(
        rexmex_factory.create_recommender_factory(),
        EvaluationPipeline
    ))"""
    """
        # lenskit recommenders
        recommendation_factory.add_factory(create_metric_pipeline_factory(
            lenskit_factory.create_recommender_factory(),
            EvaluationPipeline
        ))
    """
    recommendation_factory.add_factory(
        create_metric_pipeline_factory(create_accuracy_metric_factory(),
                                       EvaluationPipeline)
    )

    # TODO document this (shared factory pointers)
    for _, func_create in enumerate(shared_categories):
        category_factory = func_create()
        print('category factory', category_factory)
        prediction_factory.add_factory(
            create_metric_pipeline_factory(category_factory, EvaluationPipeline)
        )
        recommendation_factory.add_factory(
            create_metric_pipeline_factory(category_factory, EvaluationPipeline)
        )

    evaluation_factory = GroupFactory(KEY_EVALUATION)

    #evaluation_factory.add_factory(factory)
    evaluation_factory.add_factory(prediction_factory)
    evaluation_factory.add_factory(recommendation_factory)
    return evaluation_factory
