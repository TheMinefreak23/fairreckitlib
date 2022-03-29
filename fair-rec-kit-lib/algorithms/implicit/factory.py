"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..common import *
from .common import *
from .params import *
from .recommenders import *


def get_implicit_recommender_factory():
    return IMPLICIT_API, {
        ALTERNATING_LEAST_SQUARES: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_alternating_least_squares,
            CREATE_FUNC: create_recommender_alternating_least_squares
        },
        BAYESIAN_PERSONALIZED_RANKING: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_bayesian_personalized_ranking,
            CREATE_FUNC: create_recommender_bayesian_personalized_ranking
        },
        LOGISTIC_MATRIX_FACTORIZATION: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_logistic_matrix_factorization,
            CREATE_FUNC: create_recommender_logistic_matrix_factorization
        }
    }


def get_implicit_recommender_names():
    return IMPLICIT_API, [
        ALTERNATING_LEAST_SQUARES,
        BAYESIAN_PERSONALIZED_RANKING,
        LOGISTIC_MATRIX_FACTORIZATION
    ]


def is_implicit_recommender(recommender_name):
    _, names = get_implicit_recommender_names()

    for rec_name in names:
        if rec_name is recommender_name:
            return True

    return False
