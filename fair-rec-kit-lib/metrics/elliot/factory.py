""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..common import *
from .common import *
from .params import *
from .metrics import *


def get_elliot_metric_factory():
    return ELLIOT_API, {
        NDCG: {
            CREATE_FUNC: create_simple_metric
            CATEGORY: ACCURACY
            NAME: 'nDCG'
        },
        PRECISION: {
            CREATE_FUNC: create_metric_precision
            CATEGORY: ACCURACY
        },
        RECALL: {
            CREATE_FUNC: create_metric_recall
            CATEGORY: ACCURACY
        },
        MAP: {
            CREATE_FUNC: create_metric_map
            CATEGORY: ACCURACY
        },
        MAR: {
            CREATE_FUNC: create_metric_random
            CATEGORY: ACCURACY
        },
        RMSE: {
            CREATE_FUNC: create_recommender_svd_pure
            CATEGORY: RATING
        },
        MAE: {
            CREATE_FUNC: create_recommender_svd_pp
            CATEGORY: RATING
        },
        MRR: {
            CREATE_FUNC: create_recommender_item_knn
            CATEGORY: ACCURACY
        },
        USER_COVERAGE_N: {
            CREATE_FUNC: create_recommender_user_knn
            CATEGORY: COVERAGE
        },
        USER_COVERAGE: {
            CREATE_FUNC: create_recommender_user_knn
            CATEGORY: COVERAGE
        },
        ITEM_COVERAGE: {
            CREATE_FUNC: create_recommender_user_knn
            CATEGORY: COVERAGE
        },
        BIAS_DISPARITY_BD: {
            CREATE_FUNC: create_recommender_user_knn
            CATEGORY: FAIRNESS
        },
        GINI_INDEX: {
            CREATE_FUNC: create_recommender_user_knn
            CATEGORY: DIVERSITY
        },
        EFD: {
            CREATE_FUNC: create_recommender_user_knn
            CATEGORY: NOVELTY
        },
    }


def get_elliot_metric_names():
    return ELLIOT_API, [
        NDCG,
        MAP,
        MAR,
        RMSE,
        MAE,
        MRR,
        USER_COVERAGE_N,
        USER_COVERAGE,
        ITEM_COVERAGE,
        BIAS_DISPARITY_BD,
        GINI_INDEX,
        EFD
    ]


def is_elliot_metric(input_name):
    _, names = get_elliot_metric_names()

    for metric_name in names:
        if metric_name is input_name:
            return True

    return False
