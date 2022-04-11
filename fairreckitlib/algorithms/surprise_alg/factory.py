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

from ..factory import create_algorithm_factory_from_list
from ..params import get_params_empty
from .params import get_surprise_params_baseline_only_als
from .params import get_surprise_params_baseline_only_sgd
from .params import get_surprise_params_co_clustering
from .params import get_surprise_params_nmf
from .params import get_surprise_params_svd
from .params import get_surprise_params_svd_pp
from .predictor import SurprisePredictor

SURPRISE_API = 'Surprise'

SURPRISE_BASELINE_ONLY_ALS = 'BaselineOnlyALS'
SURPRISE_BASELINE_ONLY_SGD = 'BaselineOnlySGD'
SURPRISE_CO_CLUSTERING = 'CoClustering'
SURPRISE_NMF = 'NMF'
SURPRISE_NORMAL_PREDICTOR = 'NormalPredictor'
SURPRISE_SLOPE_ONE = 'SlopeOne'
SURPRISE_SVD = 'SVD'
SURPRISE_SVD_PP = 'SVDpp'


def get_surprise_predictor_factory():
    """Gets the algorithm factory with Surprise predictors.

    Returns:
        (AlgorithmFactory) with available predictors.
    """
    return create_algorithm_factory_from_list(SURPRISE_API, [
        (SURPRISE_BASELINE_ONLY_ALS,
         _create_predictor_baseline_only_als,
         get_surprise_params_baseline_only_als
         ),
        (SURPRISE_BASELINE_ONLY_SGD,
         _create_predictor_baseline_only_sgd,
         get_surprise_params_baseline_only_sgd
         ),
        (SURPRISE_CO_CLUSTERING,
         _create_predictor_co_clustering,
         get_surprise_params_co_clustering
         ),
        (SURPRISE_NMF,
         _create_predictor_nmf,
         get_surprise_params_nmf
         ),
        (SURPRISE_NORMAL_PREDICTOR,
         _create_predictor_normal_predictor,
         get_params_empty
         ),
        (SURPRISE_SLOPE_ONE,
         _create_predictor_slope_one,
         get_params_empty
         ),
        (SURPRISE_SVD,
         _create_predictor_svd,
         get_surprise_params_svd
         ),
        (SURPRISE_SVD_PP,
         _create_predictor_svd_pp,
         get_surprise_params_svd_pp
         )
    ])


def _create_predictor_baseline_only_als(params, **kwargs):
    """Creates the BaselineOnly ALS predictor.

    Args:
        params(dict): with the entries:
            reg_i,
            reg_u,
            epochs

    Returns:
        (SurprisePredictor) wrapper of BaselineOnly with method 'als'.
    """
    bsl_options = {
        'method': 'als',
        'reg_i': params['reg_i'],
        'reg_u': params['reg_u'],
        'n_epochs': params['epochs']
    }

    return SurprisePredictor(BaselineOnly(
        bsl_options=bsl_options,
        verbose=False
    ), params, **kwargs)


def _create_predictor_baseline_only_sgd(params, **kwargs):
    """Creates the BaselineOnly SGD predictor.

    Args:
        params(dict): with the entries:
            regularization,
            learning_rate,
            epochs

    Returns:
        (SurprisePredictor) wrapper of BaselineOnly with method 'sgd'.
    """
    bsl_options = {
        'method': 'sgd',
        'reg': params['regularization'],
        'learning_rate': params['learning_rate'],
        'n_epochs': params['epochs']
     }

    return SurprisePredictor(BaselineOnly(
        bsl_options=bsl_options,
        verbose=False
    ), params, **kwargs)


def _create_predictor_co_clustering(params, **kwargs):
    """Creates the CoClustering predictor.

    Args:
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

    return SurprisePredictor(CoClustering(
        n_cltr_u=params['user_clusters'],
        n_cltr_i=params['item_clusters'],
        n_epochs=params['epochs'],
        random_state=params['random_seed'],
        verbose=False
    ), params, **kwargs)


def _create_predictor_nmf(params, **kwargs):
    """Creates the NMF predictor.

    Args:
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

    return SurprisePredictor(NMF(
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
    ), params, **kwargs)


def _create_predictor_normal_predictor(params, **kwargs):
    """Creates the NormalPredictor.

    Returns:
        (SurprisePredictor) wrapper of NormalPredictor.
    """
    return SurprisePredictor(NormalPredictor(), params, **kwargs)


def _create_predictor_slope_one(params, **kwargs):
    """Creates the SlopeOne predictor.

    Returns:
        (SurprisePredictor) wrapper of SlopeOne.
    """
    return SurprisePredictor(SlopeOne(), params, **kwargs)


def _create_predictor_svd(params, **kwargs):
    """Creates the SVD predictor.

    Args:
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

    return SurprisePredictor(SVD(
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
    ), params, **kwargs)


def _create_predictor_svd_pp(params, **kwargs):
    """Creates the SVDpp predictor.

    Args:
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

    return SurprisePredictor(SVDpp(
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
    ), params, **kwargs)
