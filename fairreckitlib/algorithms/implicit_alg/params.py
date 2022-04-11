"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..params import PARAM_KEY_OPTIONS
from ..params import PARAM_KEY_VALUES
from ..params import create_param_factors
from ..params import create_param_bool
from ..params import create_param_seed
from ..params import create_value_param


def get_implicit_params_alternating_least_squares():
    """Gets the parameters of the AlternatingLeastSquares algorithm.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [
            create_param_factors(100),
            _create_param_iterations(15, 50),
            _create_param_regularization(0.01),
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: [
            create_param_bool('calculate_training_loss', False),
            create_param_bool('use_cg', True),
            create_param_bool('use_native', True)
        ]
    }


def get_implicit_params_bayesian_personalized_ranking():
    """Gets the parameters of the BayesianPersonalizedRanking algorithm.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [
            create_param_factors(100),
            _create_param_iterations(100, 1000),
            _create_param_regularization(0.01),
            _create_param_learning_rate(0.01),
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: [
            create_param_bool('verify_negative_samples', True)
        ]
    }


def get_implicit_params_logistic_matrix_factorization():
    """Gets the parameters of the LogisticMatrixFactorization algorithm.

    Returns:
        (dict) containing array like values and options.
    """
    return {
        PARAM_KEY_VALUES: [
            create_param_factors(30),
            _create_param_iterations(30, 100),
            _create_param_regularization(0.6),
            _create_param_learning_rate(1.00),
            create_value_param('neg_prop', 1, 50, 30),
            create_param_seed('random_seed')
        ],
        PARAM_KEY_OPTIONS: []
    }


def _create_param_iterations(iterations, max_iterations):
    return create_value_param('iterations', 1, max_iterations, iterations)


def _create_param_learning_rate(learning_rate):
    return create_value_param('learning_rate', 0.0001, 1.0, learning_rate)


def _create_param_regularization(regularization):
    return create_value_param('regularization', 0.0001, 1.0, regularization)
