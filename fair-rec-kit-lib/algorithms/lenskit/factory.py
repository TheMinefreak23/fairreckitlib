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
        BIASED_MF: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_biased_mf,
            CREATE_FUNC: create_recommender_biased_mf
        },
        IMPLICIT_MF: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_implicit_mf,
            CREATE_FUNC: create_recommender_implicit_mf
        },
        POP_SCORE: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_pop_score,
            CREATE_FUNC: create_recommender_pop_score
        },
        RANDOM: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_random,
            CREATE_FUNC: create_recommender_random
        }
    }


def get_lenskit_recommender_names():
    return LENSKIT_API, [
        BIASED_MF,
        IMPLICIT_MF,
        POP_SCORE,
        RANDOM
    ]


def is_lenskit_recommender(recommender_name):
    _, names = get_lenskit_recommender_names()

    for rec_name in names:
        if rec_name is recommender_name:
            return True

    return False
