"""This module contains functionality to create a model factory.

Constants:

    KEY_MODELS: key that is used to identify models.

Functions:

    create_algorithm_pipeline_factory: wrap algorithm factory with pipeline creation.
    create_model_factory: create factory with prediction/recommendation factories.
    create_prediction_model_factory: create factory with predictor API factories.
    create_recommendation_model_factory: create factory with recommender API factories.

Deprecated:

from .algorithms.elliot import elliot_factory
from .pipeline.recommendation_pipeline_elliot import RecommendationPipelineElliot

def create_recommendation_model_factory() -> GroupFactory:
    ...
    # elliot recommenders
    model_factory.add_factory(create_algorithm_pipeline_factory(
        elliot_factory.create_recommender_factory(),
        RecommendationPipelineElliot
    ))
    ...

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Callable

from ..core.config_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from ..core.events.event_dispatcher import EventDispatcher
from ..core.factories import Factory, GroupFactory
from .algorithms.implicit import implicit_factory
from .algorithms.lenskit import lenskit_factory
from .algorithms.surprise import surprise_factory
from .pipeline.model_pipeline import ModelPipeline
from .pipeline.prediction_pipeline import PredictionPipeline
from .pipeline.recommendation_pipeline import RecommendationPipeline

KEY_MODELS = 'models'


def create_algorithm_pipeline_factory(
        algo_factory: Factory,
        create_pipeline: Callable[[Factory, EventDispatcher], ModelPipeline]) -> Factory:
    """Create an algorithm pipeline factory.

    Args:
        algo_factory: the factory with available algorithms.
        create_pipeline: the pipeline creation function associated with the factory.

    Returns
        the algorithm pipeline factory.
    """
    algo_factory.create_pipeline = create_pipeline
    return algo_factory


def create_model_factory() -> GroupFactory:
    """Create a model factory with all predictor and recommender algorithms.

    Returns:
        the group factory with all predictors and recommenders.
    """
    model_factory = GroupFactory(KEY_MODELS)
    model_factory.add_factory(create_prediction_model_factory())
    model_factory.add_factory(create_recommendation_model_factory())
    return model_factory


def create_prediction_model_factory() -> GroupFactory:
    """Create a model factory with all predictor algorithms.

    Consists of algorithms from two APIs:
        1) LensKit predictor algorithms.
        2) Surprise predictor algorithms.

    Returns:
        the group factory with all predictors.
    """
    model_factory = GroupFactory(TYPE_PREDICTION)

    # lenskit predictors
    model_factory.add_factory(create_algorithm_pipeline_factory(
        lenskit_factory.create_predictor_factory(),
        PredictionPipeline
    ))
    # surprise predictors
    model_factory.add_factory(create_algorithm_pipeline_factory(
        surprise_factory.create_predictor_factory(),
        PredictionPipeline
    ))

    return model_factory


def create_recommendation_model_factory() -> GroupFactory:
    """Create a model factory with all recommender algorithms.

    Consists of algorithms from four APIs:
        1) LensKit recommender algorithms.
        2) Implicit recommender algorithms.
        3) Surprise recommender algorithms.

    Returns:
        the group factory with all recommenders.
    """
    model_factory = GroupFactory(TYPE_RECOMMENDATION)

    # lenskit recommenders
    model_factory.add_factory(create_algorithm_pipeline_factory(
        lenskit_factory.create_recommender_factory(),
        RecommendationPipeline
    ))
    # implicit recommenders
    model_factory.add_factory(create_algorithm_pipeline_factory(
        implicit_factory.create_recommender_factory(),
        RecommendationPipeline
    ))
    # surprise recommenders
    model_factory.add_factory(create_algorithm_pipeline_factory(
        surprise_factory.create_recommender_factory(),
        RecommendationPipeline
    ))

    return model_factory
