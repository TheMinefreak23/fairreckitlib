"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..common import *
from .common import *
from .params import *
from .predictors import *


def get_surprise_predictor_factory():
    return SURPRISE_API, {
        SURPRISE_BASELINE_ONLY_ALS: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_baseline_only_als,
            FUNC_CREATE_ALGORITHM: create_predictor_baseline_only_als
        },
        SURPRISE_BASELINE_ONLY_SGD: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_baseline_only_sgd,
            FUNC_CREATE_ALGORITHM: create_predictor_baseline_only_sgd
        },
        SURPRISE_CO_CLUSTERING: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_co_clustering,
            FUNC_CREATE_ALGORITHM: create_predictor_co_clustering
        },
        SURPRISE_NMF: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_nmf,
            FUNC_CREATE_ALGORITHM: create_predictor_nmf
        },
        SURPRISE_NORMAL_PREDICTOR: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_empty,
            FUNC_CREATE_ALGORITHM: create_predictor_normal_predictor
        },
        SURPRISE_SLOPE_ONE: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_empty,
            FUNC_CREATE_ALGORITHM: create_predictor_slope_one
        },
        SURPRISE_SVD: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_svd,
            FUNC_CREATE_ALGORITHM: create_predictor_svd
        },
        SURPRISE_SVD_PP: {
            FUNC_GET_ALGORITHM_PARAMS: get_params_svd_pp,
            FUNC_CREATE_ALGORITHM: create_predictor_svd_pp
        }
    }
