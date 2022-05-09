"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

from ..core.factory import create_factory_from_list
from .implicit_params import get_implicit_params_alternating_least_squares
from .implicit_params import get_implicit_params_bayesian_personalized_ranking
from .implicit_params import get_implicit_params_logistic_matrix_factorization
from .implicit_recommender import ImplicitRecommender

IMPLICIT_API = 'Implicit'

IMPLICIT_ALS = 'AlternatingLeastSquares'
IMPLICIT_BPR = 'BayesianPersonalizedRanking'
IMPLICIT_LMF = 'LogisticMatrixFactorization'


def get_implicit_recommender_factory():
    """Gets the algorithm factory with Implicit recommenders.

    Returns:
        (BaseFactory) with available recommenders.
    """
    return create_factory_from_list(IMPLICIT_API, [
        (IMPLICIT_ALS,
         _create_recommender_alternating_least_squares,
         get_implicit_params_alternating_least_squares
         ),
        (IMPLICIT_BPR,
         _create_recommender_bayesian_personalized_ranking,
         get_implicit_params_bayesian_personalized_ranking
         ),
        (IMPLICIT_LMF,
         _create_recommender_logistic_matrix_factorization,
         get_implicit_params_logistic_matrix_factorization
         )
    ])


def _create_recommender_alternating_least_squares(name, params, **kwargs):
    """Creates the AlternatingLeastSquares recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors,
            regularization,
            use_native,
            use_cg,
            iterations,
            calculate_training_loss,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (ImplicitRecommender) wrapper of AlternatingLeastSquares.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = AlternatingLeastSquares(
        factors=params['factors'],
        regularization=params['regularization'],
        dtype=np.float32,
        use_native=params['use_native'],
        use_cg=params['use_cg'],
        iterations=params['iterations'],
        calculate_training_loss=params['calculate_training_loss'],
        num_threads=kwargs['num_threads'],
        random_state=params['random_seed']
    )

    return ImplicitRecommender(algo, name, params, **kwargs)


def _create_recommender_bayesian_personalized_ranking(name, params, **kwargs):
    """Creates the BayesianPersonalizedRanking recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors,
            learning_rate,
            regularization,
            iterations,
            verify_negative_samples,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (ImplicitRecommender) wrapper of BayesianPersonalizedRanking.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = BayesianPersonalizedRanking(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        num_threads=kwargs['num_threads'],
        verify_negative_samples=params['verify_negative_samples'],
        random_state=params['random_seed']
    )

    return ImplicitRecommender(algo, name, params, **kwargs)


def _create_recommender_logistic_matrix_factorization(name, params, **kwargs):
    """Creates the LogisticMatrixFactorization recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            factors,
            learning_rate,
            regularization,
            iterations,
            neg_prop,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (ImplicitRecommender) wrapper of LogisticMatrixFactorization.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = LogisticMatrixFactorization(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        neg_prop=params['neg_prop'],
        num_threads=kwargs['num_threads'],
        random_state=params['random_seed']
    )

    return ImplicitRecommender(algo, name, params, **kwargs)
