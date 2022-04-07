""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..common import *
from .predictor import create_predictor_pipeline_lenskit
from .predictor import create_predictor_pipeline_surprise
from .recommender_elliot import create_recommender_pipeline_elliot
from .recommender import create_recommender_pipeline_implicit
from .recommender import create_recommender_pipeline_lenskit


def get_predictor_pipeline_factory():
    return {
        LENSKIT_API: create_predictor_pipeline_lenskit,
        SURPRISE_API: create_predictor_pipeline_surprise
    }


def get_recommender_pipeline_factory():
    return {
        ELLIOT_API: create_recommender_pipeline_elliot,
        IMPLICIT_API: create_recommender_pipeline_implicit,
        LENSKIT_API: create_recommender_pipeline_lenskit
    }
