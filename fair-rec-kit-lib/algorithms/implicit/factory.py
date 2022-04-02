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
        IMPLICIT_ALS: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_alternating_least_squares,
            FUNC_CREATE_ALGORITHM: create_recommender_alternating_least_squares
        },
        IMPLICIT_BPR: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_bayesian_personalized_ranking,
            FUNC_CREATE_ALGORITHM: create_recommender_bayesian_personalized_ranking
        },
        IMPLICIT_LMF: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_logistic_matrix_factorization,
            FUNC_CREATE_ALGORITHM: create_recommender_logistic_matrix_factorization
        }
    }
