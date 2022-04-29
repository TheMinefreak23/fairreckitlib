"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.experiment.params import ConfigParameters


def get_surprise_params_baseline_only_als():
    """Gets the parameters of the BaselineOnly ALS algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('epochs', int, 10, (1, 50))
    params.add_value('reg_i', int, 10, (1, 100))
    params.add_value('reg_u', int, 15, (1, 100))

    return params


def get_surprise_params_baseline_only_sgd():
    """Gets the parameters of the BaselineOnly SGD algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('epochs', int, 20, (1, 50))
    params.add_value('regularization', float, 0.02, (0.00001, 1.0))
    params.add_value('learning_rate', float, 0.005, (0.0001, 1.0))

    return params


def get_surprise_params_co_clustering():
    """Gets the parameters of the CoClustering algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('epochs', int, 20, (1, 50))
    params.add_value('user_clusters', int, 3, (0, 30))
    params.add_value('item_clusters', int, 3, (0, 30))
    params.add_random_seed('random_seed')

    return params

def get_surprise_params_nmf():
    """Gets the parameters of the NMF algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('factors', int, 15, (1, 100))
    params.add_value('epochs', int, 50, (1, 50))
    params.add_value('reg_pu', float, 0.06, (0.00001, 1.0))
    params.add_value('reg_qi', float, 0.06, (0.00001, 1.0))
    params.add_value('reg_bu', float, 0.02, (0.00001, 1.0))
    params.add_value('reg_bi', float, 0.02, (0.00001, 1.0))
    params.add_value('lr_bu', float, 0.005, (0.00001, 1.0))
    params.add_value('lr_bi', float, 0.005, (0.00001, 1.0))
    params.add_value('init_low', int, 0, (0, 100))
    params.add_value('init_high', int, 1, (0, 100))
    params.add_random_seed('random_seed')
    params.add_bool('biased', False)

    return params


def get_surprise_params_svd():
    """Gets the parameters of the SVD algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('factors', int, 100, (1, 100))
    params.add_value('epochs', int, 20, (1, 50))
    params.add_value('init_mean', int, 0, (-1000, 1000))
    params.add_value('init_std_dev', float, 0.1, (0.0, 1.0))
    params.add_value('learning_rate', float, 0.005, (0.00001, 1.0))
    params.add_value('regularization', float, 0.02, (0.00001, 1.0))
    params.add_random_seed('random_seed')
    params.add_bool('biased', True)

    return params


def get_surprise_params_svd_pp():
    """Gets the parameters of the SVDpp algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('factors', int, 20, (1, 100))
    params.add_value('epochs', int, 20, (1, 50))
    params.add_value('init_mean', int, 0, (-1000, 1000))
    params.add_value('init_std_dev', float, 0.1, (0.0, 1.0))
    params.add_value('learning_rate', float, 0.007, (0.00001, 1.0))
    params.add_value('regularization', float, 0.02, (0.00001, 1.0))
    params.add_random_seed('random_seed')

    return params
