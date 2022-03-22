""""
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
            GET_DEFAULT_PARAMS_FUNC: get_defaults_als,
            CREATE_FUNC: create_recommender_als
        },
        BAYESIAN_PERSONALIZED_RANKING: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_bpr,
            CREATE_FUNC: create_recommender_bpr
        },
        LOGISTIC_MATRIX_FACTORIZATION: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_lmf,
            CREATE_FUNC: create_recommender_lmf
        },
        COSINE_RECOMMENDER: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_cosine,
            CREATE_FUNC: create_recommender_cosine
        },
        TFIDF_RECOMMENDER: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_tfidf,
            CREATE_FUNC: create_recommender_tfidf
        },
        BM25_RECOMMENDER: {
            GET_DEFAULT_PARAMS_FUNC: get_defaults_bm25,
            CREATE_FUNC: create_recommender_bm25
        }
    }


def get_implicit_recommender_names():
    return IMPLICIT_API, [
        ALTERNATING_LEAST_SQUARES,
        BAYESIAN_PERSONALIZED_RANKING,
        LOGISTIC_MATRIX_FACTORIZATION,
        COSINE_RECOMMENDER,
        TFIDF_RECOMMENDER,
        BM25_RECOMMENDER
    ]


def is_implicit_recommender(recommender_name):
    _, names = get_implicit_recommender_names()

    for rec_name in names:
        if rec_name is recommender_name:
            return True

    return False
