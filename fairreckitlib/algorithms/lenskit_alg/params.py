"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..params import AlgorithmParameters


def get_lenskit_params_biased_mf():
    """Gets the params of the BiasedMF algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    methods = [
        'cd',
        'lu'
    ]

    params = AlgorithmParameters()

    params.add_value('features', int, 10, 1, 50)
    params.add_value('iterations', int, 20, 1, 50)
    params.add_value('user_reg', float, 0.1, 0.0001, 1.0)
    params.add_value('item_reg', float, 0.1, 0.0001, 1.0)
    params.add_value('damping', float, 5.0, 0.0, 1000.0)
    params.add_random_seed('random_seed')
    params.add_option('method', str, methods[0], methods)

    return params


def get_lenskit_params_implicit_mf():
    """Gets the params of the ImplicitMF algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    methods = [
        'cg',
        'lu'
    ]

    params = AlgorithmParameters()

    params.add_value('features', int, 3, 1, 50)
    params.add_value('iterations', int, 20, 1, 50)
    params.add_value('reg', float, 0.1, 0.0001, 1.0)
    params.add_value('weight', float, 40.0, 1.0, 10000.0)
    params.add_random_seed('random_seed')
    params.add_option('method', str, methods[0], methods)
    params.add_bool('use_ratings', False)

    return params


def get_lenskit_params_item_item():
    """Gets the params of the ItemItem algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    params = AlgorithmParameters()

    params.add_value('max_nnbrs', int, 10, 1, 50)
    params.add_value('min_nbrs', int, 1, 1, 50)
    params.add_value('min_sim', float, 1e-06, 0.0, 10.0)

    return params


def get_lenskit_params_pop_score():
    """Gets the params of the PopScore algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    options = [
        'quantile',
        'rank',
        'count'
    ]

    params = AlgorithmParameters()

    params.add_option('score_method', str, options[0], options)

    return params


def get_lenskit_params_random():
    """Gets the params of the Random algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    params = AlgorithmParameters()

    params.add_random_seed('random_seed')

    return params


def get_lenskit_params_user_user():
    """Gets the params of the UserUser algorithm.

    Returns:
        params(AlgorithmParameters) the params of the algorithm.
    """
    params = AlgorithmParameters()

    params.add_value('max_nnbrs', int, 10, 1, 50)
    params.add_value('min_nbrs', int, 1, 1, 50)
    params.add_value('min_sim', float, 0.0, 0.0, 10.0)

    return params
