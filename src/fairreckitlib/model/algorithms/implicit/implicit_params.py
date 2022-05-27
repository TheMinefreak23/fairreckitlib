"""This module contains the parameter creation functions for implicit recommenders.

Functions:

    create_params_als: create AlternatingLeastSquares config parameters.
    create_params_bpr: create BayesianPersonalizedRanking config parameters.
    create_params_lmf: create LogisticMatrixFactorization config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.params.config_parameters import ConfigParameters


def create_params_als() -> ConfigParameters:
    """Create the parameters of the AlternatingLeastSquares algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 100, (1, 100))
    params.add_number('iterations', int, 15, (1, 50))
    params.add_number('regularization', float, 0.01, (0.0001, 1.0))
    params.add_random_seed('random_seed')
    params.add_bool('calculate_training_loss', False)
    params.add_bool('use_cg', True)
    params.add_bool('use_native', True)
    return params


def create_params_bpr() -> ConfigParameters:
    """Create the parameters of the BayesianPersonalizedRanking algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 100, (1, 100))
    params.add_number('iterations', int, 100, (1, 1000))
    params.add_number('regularization', float, 0.01, (0.0001, 1.0))
    params.add_number('learning_rate', float, 0.01, (0.0001, 1.0))
    params.add_random_seed('random_seed')
    params.add_bool('verify_negative_samples', True)
    return params


def create_params_lmf() -> ConfigParameters:
    """Create the parameters of the LogisticMatrixFactorization algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 30, (1, 100))
    params.add_number('iterations', int, 30, (1, 100))
    params.add_number('regularization', float, 0.6, (0.0001, 1.0))
    params.add_number('learning_rate', float, 1.0, (0.0001, 1.0))
    params.add_number('neg_prop', int, 30, (1, 50))
    params.add_random_seed('random_seed')
    return params
