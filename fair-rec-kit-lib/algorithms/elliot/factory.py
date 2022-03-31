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
            GET_PARAMS_FUNC: get_params_funk_svd,
            CREATE_FUNC: create_recommender_funk_svd
        },
        ELLIOT_ITEM_KNN: {
            GET_PARAMS_FUNC: get_params_item_knn,
            CREATE_FUNC: create_recommender_item_knn
        },
        ELLIOT_MULTI_VAE: {
            GET_PARAMS_FUNC: get_params_multi_vae,
            CREATE_FUNC: create_recommender_multi_vae
        },
        ELLIOT_MOST_POP: {
            GET_PARAMS_FUNC: get_params_most_pop,
            CREATE_FUNC: create_recommender_most_pop
        },
        ELLIOT_PURE_SVD: {
            GET_PARAMS_FUNC: get_params_pure_svd,
            CREATE_FUNC: create_recommender_pure_svd
        },
        ELLIOT_RANDOM: {
            GET_PARAMS_FUNC: get_params_random,
            CREATE_FUNC: create_recommender_random
        },
        ELLIOT_SVD_PP: {
            GET_PARAMS_FUNC: get_params_svd_pp,
            CREATE_FUNC: create_recommender_svd_pp
        },
        ELLIOT_USER_KNN: {
            GET_PARAMS_FUNC: get_params_user_knn,
            CREATE_FUNC: create_recommender_user_knn
        }
    }
