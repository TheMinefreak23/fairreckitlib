"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

from ..factory import create_algorithm_factory_from_list
from ..params import get_params_empty
from .params import get_elliot_params_funk_svd
from .params import get_elliot_params_item_knn
from .params import get_elliot_params_multi_vae
from .params import get_elliot_params_pure_svd
from .params import get_elliot_params_random
from .params import get_elliot_params_svd_pp
from .params import get_elliot_params_user_knn
from .recommender import ElliotRecommender

ELLIOT_API = 'Elliot'

ELLIOT_FUNK_SVD = 'FunkSVD'
ELLIOT_ITEM_KNN = 'ItemKNN'
ELLIOT_MULTI_VAE = 'MultiVAE'
ELLIOT_MOST_POP = 'MostPop'
ELLIOT_RANDOM = 'Random'
ELLIOT_PURE_SVD = 'PureSVD'
ELLIOT_SVD_PP = 'SVDpp'
ELLIOT_USER_KNN = 'UserKNN'


def get_elliot_recommender_factory():
    """Gets the algorithm factory with Elliot recommenders.

    Returns:
        (AlgorithmFactory) with available recommenders.
    """
    return create_algorithm_factory_from_list(ELLIOT_API, [
        (ELLIOT_FUNK_SVD,
         _create_recommender_funk_svd,
         get_elliot_params_funk_svd
         ),
        (ELLIOT_ITEM_KNN,
         _create_recommender_item_knn,
         get_elliot_params_item_knn
         ),
        (ELLIOT_MOST_POP,
         _create_recommender_most_pop,
         get_params_empty
         ),
        (ELLIOT_MULTI_VAE,
         _create_recommender_multi_vae,
         get_elliot_params_multi_vae
         ),
        (ELLIOT_PURE_SVD,
         _create_recommender_pure_svd,
         get_elliot_params_pure_svd
         ),
        (ELLIOT_RANDOM,
         _create_recommender_random,
         get_elliot_params_random
         ),
        (ELLIOT_SVD_PP,
         _create_recommender_svd_pp,
         get_elliot_params_svd_pp
         ),
        (ELLIOT_USER_KNN,
         _create_recommender_user_knn,
         get_elliot_params_user_knn
         )
    ])


def _create_recommender_funk_svd(params, **kwargs):
    """Creates the FunkSVD recommender.

    Args:
        params(dict): with the entries:
            epochs,
            batch_size,
            factors,
            learning_rate,
            reg_w,
            reg_b,
            seed

    Returns:
        (ElliotRecommender) wrapper of FunkSVD.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ElliotRecommender(params, **kwargs)


def _create_recommender_item_knn(params, **kwargs):
    """Creates the ItemKNN recommender.

    Args:
        params(dict): with the entries:
            neighbours,
            similarity,
            implementation

    Returns:
        (ElliotRecommender) wrapper of ItemKNN.
    """
    return ElliotRecommender(params, **kwargs)


def _create_recommender_most_pop(params, **kwargs):
    """Creates the MostPop recommender.

    Returns:
        (ElliotRecommender) wrapper of MostPop.
    """
    return ElliotRecommender(params, **kwargs)


def _create_recommender_multi_vae(params, **kwargs):
    """Creates the MultiVAE recommender.

    Args:
        params(dict): with the entries:
            batch_size,
            epochs,
            learning_rate,
            reg_lambda,
            intermediate_dim,
            latent_dim,
            dropout_keep,
            seed

    Returns:
        (ElliotRecommender) wrapper of MultiVAE.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ElliotRecommender(params, **kwargs)


def _create_recommender_pure_svd(params, **kwargs):
    """Creates the PureSVD recommender.

    Args:
        params(dict): with the entries:
            factors,
            seed

    Returns:
        (ElliotRecommender) wrapper of PureSVD.
    """
    return ElliotRecommender(params, **kwargs)


def _create_recommender_random(params, **kwargs):
    """Creates the Random recommender.

    Args:
        params(dict): with the entries:
            random_seed

    Returns:
        (ElliotRecommender) wrapper of Random.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return ElliotRecommender(params, **kwargs)


def _create_recommender_svd_pp(params, **kwargs):
    """Creates the SVDpp recommender.

    Args:
        params(dict): with the entries:
            epochs,
            batch_size,
            factors,
            learning_rate,
            reg_w,
            reg_b,
            seed

    Returns:
        (ElliotRecommender) wrapper of SVDpp.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ElliotRecommender(params, **kwargs)


def _create_recommender_user_knn(params, **kwargs):
    """Creates the UserKNN recommender.

    Args:
        params(dict): with the entries:
            neighbours,
            similarity,
            implementation

    Returns:
        (ElliotRecommender) wrapper of UserKNN.
    """
    return ElliotRecommender(params, **kwargs)
