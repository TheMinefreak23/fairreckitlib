"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..params import AlgorithmParameters


def get_implicit_params_alternating_least_squares():
    """Gets the parameters of the AlternatingLeastSquares algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    params = AlgorithmParameters()

    params.add_value('factors', int, 100, 1, 100)
    params.add_value('iterations', int, 15, 1, 50)
    params.add_value('regularization', float, 0.01, 0.0001, 1.0)
    params.add_random_seed('random_seed')
    params.add_bool('calculate_training_loss', False)
    params.add_bool('use_cg', True)
    params.add_bool('use_native', True)

    return params


def get_implicit_params_bayesian_personalized_ranking():
    """Gets the parameters of the BayesianPersonalizedRanking algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    params = AlgorithmParameters()

    params.add_value('factors', int, 100, 1, 100)
    params.add_value('iterations', int, 100, 1, 1000)
    params.add_value('regularization', float, 0.01, 0.0001, 1.0)
    params.add_value('learning_rate', float, 0.01, 0.0001, 1.0)
    params.add_random_seed('random_seed')
    params.add_bool('verify_negative_samples', True)

    return params


def get_implicit_params_logistic_matrix_factorization():
    """Gets the parameters of the LogisticMatrixFactorization algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    params = AlgorithmParameters()

    params.add_value('factors', int, 30, 1, 100)
    params.add_value('iterations', int, 30, 1, 100)
    params.add_value('regularization', float, 0.6, 0.0001, 1.0)
    params.add_value('learning_rate', float, 1.0, 0.0001, 1.0)
    params.add_value('neg_prop', int, 30, 1, 50)
    params.add_random_seed('random_seed')

    return params
