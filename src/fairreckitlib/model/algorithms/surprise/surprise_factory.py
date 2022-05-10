"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

from surprise.prediction_algorithms import BaselineOnly
from surprise.prediction_algorithms import CoClustering
from surprise.prediction_algorithms import NMF
from surprise.prediction_algorithms import NormalPredictor
from surprise.prediction_algorithms import SlopeOne
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import SVDpp

from ....core.apis import SURPRISE_API
from ....core.factories import create_factory_from_list
from ..top_k_recommender import TopK
from .surprise_params import create_surprise_params_baseline_only_als
from .surprise_params import create_surprise_params_baseline_only_sgd
from .surprise_params import create_surprise_params_co_clustering
from .surprise_params import create_surprise_params_nmf
from .surprise_params import create_surprise_params_svd
from .surprise_params import create_surprise_params_svd_pp
from .surprise_predictor import SurprisePredictor

SURPRISE_BASELINE_ONLY_ALS = 'BaselineOnlyALS'
SURPRISE_BASELINE_ONLY_SGD = 'BaselineOnlySGD'
SURPRISE_CO_CLUSTERING = 'CoClustering'
SURPRISE_NMF = 'NMF'
SURPRISE_NORMAL_PREDICTOR = 'NormalPredictor'
SURPRISE_SLOPE_ONE = 'SlopeOne'
SURPRISE_SVD = 'SVD'
SURPRISE_SVD_PP = 'SVDpp'


def create_surprise_predictor_factory():
    """Creates the algorithm factory with Surprise predictors.

    Returns:
        (Factory) with available predictors.
    """
    return create_factory_from_list(SURPRISE_API, [
        (SURPRISE_BASELINE_ONLY_ALS,
         _create_predictor_baseline_only_als,
         create_surprise_params_baseline_only_als
         ),
        (SURPRISE_BASELINE_ONLY_SGD,
         _create_predictor_baseline_only_sgd,
         create_surprise_params_baseline_only_sgd
         ),
        (SURPRISE_CO_CLUSTERING,
         _create_predictor_co_clustering,
         create_surprise_params_co_clustering
         ),
        (SURPRISE_NMF,
         _create_predictor_nmf,
         create_surprise_params_nmf
         ),
        (SURPRISE_NORMAL_PREDICTOR,
         _create_predictor_normal_predictor,
         None
         ),
        (SURPRISE_SLOPE_ONE,
         _create_predictor_slope_one,
         None
         ),
        (SURPRISE_SVD,
         _create_predictor_svd,
         create_surprise_params_svd
         ),
        (SURPRISE_SVD_PP,
         _create_predictor_svd_pp,
         create_surprise_params_svd_pp
         )
    ])


def create_surprise_recommender_factory():
    """Creates the algorithm factory with Surprise recommenders.

    Returns:
        (Factory) with available recommenders.
    """
    return create_factory_from_list(SURPRISE_API, [
        (SURPRISE_BASELINE_ONLY_ALS,
         _create_recommender_baseline_only_als,
         create_surprise_params_baseline_only_als
         ),
        (SURPRISE_BASELINE_ONLY_SGD,
         _create_recommender_baseline_only_sgd,
         create_surprise_params_baseline_only_sgd
         ),
        (SURPRISE_CO_CLUSTERING,
         _create_recommender_co_clustering,
         create_surprise_params_co_clustering
         ),
        (SURPRISE_NMF,
         _create_recommender_nmf,
         create_surprise_params_nmf
         ),
        (SURPRISE_NORMAL_PREDICTOR,
         _create_recommender_normal_predictor,
         None
         ),
        (SURPRISE_SLOPE_ONE,
         _create_recommender_slope_one,
         None
         ),
        (SURPRISE_SVD,
         _create_recommender_svd,
         create_surprise_params_svd
         ),
        (SURPRISE_SVD_PP,
         _create_recommender_svd_pp,
         create_surprise_params_svd_pp
         )
    ])


def _create_predictor_baseline_only_als(name, params, **kwargs):
    """Creates the BaselineOnly ALS predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            reg_i,
            reg_u,
            epochs

    Returns:
        (SurprisePredictor) wrapper of BaselineOnly with method 'als'.
    """
    algo = BaselineOnly(
        bsl_options={
            'method': 'als',
            'reg_i': params['reg_i'],
            'reg_u': params['reg_u'],
            'n_epochs': params['epochs']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def _create_recommender_baseline_only_als(name, params, **kwargs):
    predictor = _create_predictor_baseline_only_als(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_baseline_only_sgd(name, params, **kwargs):
    """Creates the BaselineOnly SGD predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            regularization,
            learning_rate,
            epochs

    Returns:
        (SurprisePredictor) wrapper of BaselineOnly with method 'sgd'.
    """
    algo = BaselineOnly(
        bsl_options={
            'method': 'sgd',
            'reg': params['regularization'],
            'learning_rate': params['learning_rate'],
            'n_epochs': params['epochs']
         },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def _create_recommender_baseline_only_sgd(name, params, **kwargs):
    predictor = _create_predictor_baseline_only_sgd(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_co_clustering(name, params, **kwargs):
    """Creates the CoClustering predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            user_clusters,
            item_clusters,
            epochs,
            random_seed

    Returns:
        (SurprisePredictor) wrapper of CoClustering.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = CoClustering(
        n_cltr_u=params['user_clusters'],
        n_cltr_i=params['item_clusters'],
        n_epochs=params['epochs'],
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def _create_recommender_co_clustering(name, params, **kwargs):
    predictor = _create_predictor_co_clustering(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_nmf(name, params, **kwargs):
    """Creates the NMF predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors,
            epochs,
            biased,
            reg_pu,
            reg_qi,
            reg_bu,
            reg_bi,
            init_low,
            init_high,
            random_seed

    Returns:
        (SurprisePredictor) wrapper of NMF.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = NMF(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        biased=params['biased'],
        reg_pu=params['reg_pu'],
        reg_qi=params['reg_qi'],
        reg_bu=params['reg_bu'],
        reg_bi=params['reg_bi'],
        lr_bu=params['lr_bu'],
        lr_bi=params['lr_bi'],
        init_low=params['init_low'],
        init_high=params['init_high'],
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def _create_recommender_nmf(name, params, **kwargs):
    predictor = _create_predictor_nmf(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_normal_predictor(name, params, **kwargs):
    """Creates the NormalPredictor.

    Args:
        name(str): the name of the algorithm.

    Returns:
        (SurprisePredictor) wrapper of NormalPredictor.
    """
    return SurprisePredictor(NormalPredictor(), name, params, **kwargs)


def _create_recommender_normal_predictor(name, params, **kwargs):
    predictor = _create_predictor_normal_predictor(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_slope_one(name, params, **kwargs):
    """Creates the SlopeOne predictor.

    Args:
        name(str): the name of the algorithm.

    Returns:
        (SurprisePredictor) wrapper of SlopeOne.
    """
    return SurprisePredictor(SlopeOne(), name, params, **kwargs)


def _create_recommender_slope_one(name, params, **kwargs):
    predictor = _create_predictor_slope_one(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_svd(name, params, **kwargs):
    """Creates the SVD predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors,
            epochs,
            biased,
            init_mean,
            init_std_dev,
            learning_rate,
            regularization,
            random_seed

    Returns:
        (SurprisePredictor) wrapper of SVD.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = SVD(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        biased=params['biased'],
        init_mean=params['init_mean'],
        init_std_dev=params['init_std_dev'],
        lr_all=params['learning_rate'],
        reg_all=params['regularization'],
        lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
        reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def _create_recommender_svd(name, params, **kwargs):
    predictor =_create_predictor_svd(name, params, **kwargs)
    return TopK(predictor, **kwargs)


def _create_predictor_svd_pp(name, params, **kwargs):
    """Creates the SVDpp predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors,
            epochs,
            init_mean,
            init_std_dev,
            learning_rate,
            regularization,
            random_seed

    Returns:
        (SurprisePredictor) wrapper of SVDpp.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = SVDpp(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        init_mean=params['init_mean'],
        init_std_dev=params['init_std_dev'],
        lr_all=params['learning_rate'],
        reg_all=params['regularization'],
        lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None,
        reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None,
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def _create_recommender_svd_pp(name, params, **kwargs):
    predictor = _create_predictor_svd_pp(name, params, **kwargs)
    return TopK(predictor, **kwargs)
