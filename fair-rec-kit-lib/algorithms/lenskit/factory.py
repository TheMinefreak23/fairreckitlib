""""
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
            GET_PARAMS_FUNC: get_params_biased_mf,
            CREATE_FUNC: create_recommender_biased_mf
        },
        LENSKIT_IMPLICIT_MF: {
            GET_PARAMS_FUNC: get_params_implicit_mf,
            CREATE_FUNC: create_recommender_implicit_mf
        },
        LENSKIT_POP_SCORE: {
            GET_PARAMS_FUNC: get_params_pop_score,
            CREATE_FUNC: create_recommender_pop_score
        },
        LENSKIT_RANDOM: {
            GET_PARAMS_FUNC: get_params_random,
            CREATE_FUNC: create_recommender_random
        }
    }
