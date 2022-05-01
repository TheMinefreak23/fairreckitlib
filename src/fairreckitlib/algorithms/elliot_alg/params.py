"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from fairreckitlib.experiment.params import ConfigParameters

_implementations = [
    'aiolli',
    'classical'
]

_similarities = [
    'cosine',
    'adjusted',
    'asymmetric',
    'pearson',
    'jaccard',
    'dice',
    'tversky',
    'tanimoto'
]

def get_elliot_params_funk_svd():
    """Gets the parameters of the FunkSVD algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('epochs', int, 10, (1, 50))
    params.add_value('batch_size', int, 512, (100, 10000))
    params.add_value('factors', int, 10, (1, 100))
    params.add_value('lr', float, 0.001, (0.0001, 1.0))
    params.add_value('reg_w', float, 0.1, (0.0001, 1.0))
    params.add_value('reg_b', float, 0.001, (0.0001, 1.0))
    params.add_random_seed('seed')

    return params


def get_elliot_params_item_knn():
    """Gets the parameters of the ItemKNN algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('neighbours', int, 40, (1, 50))
    params.add_option('similarity', str, _similarities[0], _similarities)
    params.add_option('implementation', str, _implementations[0], _implementations)

    return params


def get_elliot_params_multi_vae():
    """Gets the parameters of the MultiVAE algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('epochs', int, 10, (1, 50))
    params.add_value('batch_size', int, 512, (100, 10000))
    params.add_value('lr', float, 0.001, (0.0001, 1.0))
    params.add_value('reg_lambda', float, 0.01, (0.0001, 1.0))
    params.add_value('intermediate_dim', int, 100, (1000, 600))
    params.add_value('latent_dim', int, 200, (100, 500))
    params.add_value('dropout_keep', float, 1.0, (0.0, 1.0))
    params.add_random_seed('seed')

    return params


def get_elliot_params_pure_svd():
    """Gets the parameters of the PureSVD algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('factors', int, 10, (1, 100))
    params.add_random_seed('seed')

    return params


def get_elliot_params_random():
    """Gets the parameters of the Random algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_random_seed('random_seed')

    return params


def get_elliot_params_svd_pp():
    """Gets the parameters of the SVDpp algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('epochs', int, 10, (1, 50))
    params.add_value('batch_size', int, 512, (100, 10000))
    params.add_value('factors', int, 50, (1, 100))
    params.add_value('lr', float, 0.001, (0.0001, 1.0))
    params.add_value('reg_w', float, 0.1, (0.0001, 1.0))
    params.add_value('reg_b', float, 0.001, (0.0001, 1.0))
    params.add_random_seed('seed')

    return params


def get_elliot_params_user_knn():
    """Gets the parameters of the UserKNN algorithm.

    Returns:
        params(ConfigParameters) the params of the algorithm.
    """
    params = ConfigParameters()

    params.add_value('neighbours', int, 40, (1, 50))
    params.add_option('similarity', str, _similarities[0], _similarities)
    params.add_option('implementation', str, _implementations[0], _implementations)

    return params
