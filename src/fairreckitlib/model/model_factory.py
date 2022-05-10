""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..core.config_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from ..core.factories import GroupFactory
from .algorithms.elliot.elliot_factory import create_elliot_recommender_factory
from .algorithms.implicit.implicit_factory import create_implicit_recommender_factory
from .algorithms.lenskit.lenskit_factory import create_lenskit_predictor_factory
from .algorithms.lenskit.lenskit_factory import create_lenskit_recommender_factory
from .algorithms.surprise.surprise_factory import create_surprise_predictor_factory
from .algorithms.surprise.surprise_factory import create_surprise_recommender_factory
from .pipeline.prediction_pipeline import PredictionPipeline
from .pipeline.recommendation_pipeline import RecommendationPipeline
from .pipeline.recommendation_pipeline_elliot import RecommendationPipelineElliot
from .pipeline.model_config import KEY_MODELS


def create_algorithm_factory(func_create_factory, func_create_pipeline):
    algo_factory = func_create_factory()
    algo_factory.func_create_pipeline = func_create_pipeline # TODO document this
    return algo_factory


def create_model_factory():
    model_factory = GroupFactory(KEY_MODELS)
    model_factory.add_factory(create_prediction_model_factory())
    model_factory.add_factory(create_recommendation_model_factory())
    return model_factory


def create_model_factory_from_list(algo_type, algo_factory_tuples):
    model_factory = GroupFactory(algo_type)

    for _, (func_create_factory, func_create_pipeline) in enumerate(algo_factory_tuples):
        algo_factory = create_algorithm_factory(func_create_factory, func_create_pipeline)
        model_factory.add_factory(algo_factory)

    return model_factory


def create_prediction_model_factory():
    """Creates a model factory with all predictor algorithms.

    Consists of algorithms from two APIs:
        1) LensKit predictor algorithms.
        2) Surprise predictor algorithms.

    Returns:
        (GroupFactory) with all predictors.
    """
    return create_model_factory_from_list(TYPE_PREDICTION, [
        (create_lenskit_predictor_factory, PredictionPipeline),
        (create_surprise_predictor_factory, PredictionPipeline)
    ])


def create_recommendation_model_factory():
    """Creates a model factory with all recommender algorithms.

    Consists of algorithms from four APIs:
        1) Elliot recommender algorithms.
        2) LensKit recommender algorithms.
        3) Implicit recommender algorithms.
        4) Surprise recommender algorithms.

    Returns:
        (GroupFactory) with all recommenders.
    """
    return create_model_factory_from_list(TYPE_RECOMMENDATION, [
        (create_elliot_recommender_factory, RecommendationPipelineElliot),
        (create_lenskit_recommender_factory, RecommendationPipeline),
        (create_implicit_recommender_factory, RecommendationPipeline),
        (create_surprise_recommender_factory, RecommendationPipeline)
    ])
