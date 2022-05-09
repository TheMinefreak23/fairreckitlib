""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from src.fairreckitlib.experiment.constants import EXP_TYPE_PREDICTION
from src.fairreckitlib.experiment.constants import EXP_TYPE_RECOMMENDATION
from src.fairreckitlib.experiment.constants import EXP_KEY_MODELS
from ..core.factory import GroupFactory
from .algorithms.elliot_factory import get_elliot_recommender_factory
from .algorithms.implicit_factory import get_implicit_recommender_factory
from .algorithms.lenskit_factory import get_lenskit_predictor_factory
from .algorithms.lenskit_factory import get_lenskit_recommender_factory
from .algorithms.surprise_factory import get_surprise_predictor_factory
from .algorithms.surprise_factory import get_surprise_recommender_factory
from .pipeline.predictor_pipeline import PredictorPipeline
from .pipeline.recommender_pipeline import RecommenderPipeline
from .pipeline.recommender_pipeline_elliot import RecommenderPipelineElliot


def create_model_factory():
    model_factory = GroupFactory(EXP_KEY_MODELS)
    model_factory.add_factory(create_predictor_model_factory())
    model_factory.add_factory(create_recommender_model_factory())
    return model_factory


def create_model_factory_from_list(algo_type, algo_factory_tuples):
    model_factory = GroupFactory(algo_type)

    for _, (func_create_factory, func_create_pipeline) in enumerate(algo_factory_tuples):
        algo_factory = func_create_factory()
        algo_factory.func_create_pipeline = func_create_pipeline

        model_factory.add_factory(algo_factory)

    return model_factory


def create_predictor_model_factory():
    """Creates a model factory with all predictor algorithms.

    Consists of algorithms from two APIs:
        1) LensKit predictor algorithms.
        2) Surprise predictor algorithms.

    Returns:
        (GroupFactory) with all predictors.
    """
    return create_model_factory_from_list(EXP_TYPE_PREDICTION, [
        (get_lenskit_predictor_factory, PredictorPipeline),
        (get_surprise_predictor_factory, PredictorPipeline)
    ])

def create_recommender_model_factory():
    """Creates a model factory with all recommender algorithms.

    Consists of algorithms from three APIs:
        1) Elliot recommender algorithms.
        2) LensKit recommender algorithms.
        3) Implicit recommender algorithms.
        4) Surprise recommender algorithms.

    Returns:
        (GroupFactory) with all recommenders.
    """
    return create_model_factory_from_list(EXP_TYPE_RECOMMENDATION, [
        (get_elliot_recommender_factory, RecommenderPipelineElliot),
        (get_lenskit_recommender_factory, RecommenderPipeline),
        (get_implicit_recommender_factory, RecommenderPipeline),
        (get_surprise_recommender_factory, RecommenderPipeline)
    ])
