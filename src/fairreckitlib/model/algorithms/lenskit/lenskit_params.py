"""This module contains the parameter creation functions for lenskit predictors/recommenders.

Functions:

    create_params_biased_mf: create BiasedMF config parameters.
    create_params_implicit_mf: create ImplicitMF config parameters.
    create_params_knn: create ItemItem/UserUser config parameters.
    create_params_pop_score: create PopScore config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.config_params import ConfigParameters


def create_params_biased_mf() -> ConfigParameters:
    """Create the parameters of the BiasedMF algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    methods = ['cd', 'lu']

    params = ConfigParameters()
    params.add_value('features', int, 10, (1, 50))
    params.add_value('iterations', int, 20, (1, 50))
    params.add_value('user_reg', float, 0.1, (0.0001, 1.0))
    params.add_value('item_reg', float, 0.1, (0.0001, 1.0))
    params.add_value('damping', float, 5.0, (0.0, 1000.0))
    params.add_random_seed('seed')
    params.add_option('method', str, methods[0], methods)
    return params


def create_params_implicit_mf() -> ConfigParameters:
    """Create the parameters of the ImplicitMF algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    methods = ['cg', 'lu']

    params = ConfigParameters()
    params.add_value('features', int, 3, (1, 50))
    params.add_value('iterations', int, 20, (1, 50))
    params.add_value('reg', float, 0.1, (0.0001, 1.0))
    params.add_value('weight', float, 40.0, (1.0, 10000.0))
    params.add_random_seed('seed')
    params.add_option('method', str, methods[0], methods)
    params.add_bool('use_ratings', False)
    return params


def create_params_knn() -> ConfigParameters:
    """Create the parameters of the k-NN algorithms.

    Returns:
        the configuration parameters of the algorithm.
    """
    feedback = ['explicit', 'implicit']

    params = ConfigParameters()
    params.add_value('max_neighbors', int, 10, (1, 50))
    params.add_value('min_neighbors', int, 1, (1, 50))
    params.add_value('min_similarity', float, 1e-06, (0.0, 10.0))
    params.add_option('feedback', str, feedback[0], feedback)
    return params


def create_params_pop_score() -> ConfigParameters:
    """Create the parameters of the PopScore algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    options = ['quantile', 'rank', 'count']

    params = ConfigParameters()
    params.add_option('score_method', str, options[0], options)
    return params
