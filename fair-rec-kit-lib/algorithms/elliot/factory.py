""""
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
        MULTI_VAE: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_multi_vae,
            CREATE_FUNC: create_recommender_multi_vae
        },
        MOST_POP: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_most_pop,
            CREATE_FUNC: create_recommender_most_pop
        },
        RANDOM: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_random,
            CREATE_FUNC: create_recommender_random
        },
        SVD_PURE: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_svd_pure,
            CREATE_FUNC: create_recommender_svd_pure
        },
        SVD_PP: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_svd_pp,
            CREATE_FUNC: create_recommender_svd_pp
        },
        ITEM_KNN: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_item_knn,
            CREATE_FUNC: create_recommender_item_knn
        },
        USER_KNN: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_user_knn,
            CREATE_FUNC: create_recommender_user_knn
        }
    }


def get_elliot_recommender_names():
    return ELLIOT_API, [
        MULTI_VAE,
        MOST_POP,
        RANDOM,
        SVD_PURE,
        SVD_PP,
        ITEM_KNN,
        USER_KNN
    ]


def is_elliot_recommender(recommender_name):
    _, names = get_elliot_recommender_names()

    for rec_name in names:
        if rec_name is recommender_name:
            return True

    return False
