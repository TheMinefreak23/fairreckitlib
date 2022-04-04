"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..common import *
from .common import *
from .params import *
from .recommenders import *


def get_elliot_recommender_factory():
    return ELLIOT_API, {
        ELLIOT_FUNK_SVD: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_funk_svd,
            FUNC_CREATE_ALGORITHM: create_recommender_funk_svd
        },
        ELLIOT_ITEM_KNN: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_item_knn,
            FUNC_CREATE_ALGORITHM: create_recommender_item_knn
        },
        ELLIOT_MULTI_VAE: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_multi_vae,
            FUNC_CREATE_ALGORITHM: create_recommender_multi_vae
        },
        ELLIOT_MOST_POP: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_most_pop,
            FUNC_CREATE_ALGORITHM: create_recommender_most_pop
        },
        ELLIOT_PURE_SVD: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_pure_svd,
            FUNC_CREATE_ALGORITHM: create_recommender_pure_svd
        },
        ELLIOT_RANDOM: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_random,
            FUNC_CREATE_ALGORITHM: create_recommender_random
        },
        ELLIOT_SVD_PP: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_svd_pp,
            FUNC_CREATE_ALGORITHM: create_recommender_svd_pp
        },
        ELLIOT_USER_KNN: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_user_knn,
            FUNC_CREATE_ALGORITHM: create_recommender_user_knn
        }
    }
