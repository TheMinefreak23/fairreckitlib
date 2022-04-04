"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..common import *
from .common import *
from .params import *
from .recommenders import *


def get_lenskit_recommender_factory():
    return LENSKIT_API, {
        LENSKIT_BIASED_MF: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_biased_mf,
            FUNC_CREATE_ALGORITHM: create_recommender_biased_mf
        },
        LENSKIT_IMPLICIT_MF: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_implicit_mf,
            FUNC_CREATE_ALGORITHM: create_recommender_implicit_mf
        },
        LENSKIT_POP_SCORE: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_pop_score,
            FUNC_CREATE_ALGORITHM: create_recommender_pop_score
        },
        LENSKIT_RANDOM: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_random,
            FUNC_CREATE_ALGORITHM: create_recommender_random
        }
    }
