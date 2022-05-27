"""This module contains the parameter creation functions for elliot recommenders.

Functions:

    create_params_funk_svd: create FunkSVD config parameters.
    create_params_knn: create ItemKNN/UserKNN config parameters.
    create_params_multi_vae: create MultiVAE config parameters.
    create_params_pure_svd: create PureSVD config parameters.
    create_params_svd_pp: create SVDpp config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.params.config_parameters import ConfigParameters


def create_params_funk_svd() -> ConfigParameters:
    """Create the parameters of the FunkSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('iterations', int, 10, (1, 50))
    params.add_number('factors', int, 10, (1, 100))
    params.add_number('learning_rate', float, 0.001, (0.0001, 1.0))
    params.add_number('regularization_factors', float, 0.1, (0.0001, 1.0))
    params.add_number('regularization_bias', float, 0.001, (0.0001, 1.0))
    params.add_random_seed('seed')
    return params


def create_params_knn() -> ConfigParameters:
    """Create the parameters of the ItemKNN/UserKNN algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    implementations = ['aiolli', 'classical']
    similarities = [
        'cosine',
        'adjusted',
        'asymmetric',
        'pearson',
        'jaccard',
        'dice',
        'tversky',
        'tanimoto'
    ]

    params = ConfigParameters()
    params.add_number('neighbors', int, 40, (1, 50))
    params.add_single_option('similarity', str, similarities[0], similarities)
    params.add_single_option('implementation', str, implementations[0], implementations)
    return params


def create_params_multi_vae() -> ConfigParameters:
    """Create the parameters of the MultiVAE algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('iterations', int, 10, (1, 50))
    params.add_number('factors', int, 200, (100, 500))
    params.add_number('learning_rate', float, 0.001, (0.0001, 1.0))
    params.add_number('regularization_factors', float, 0.01, (0.0001, 1.0))
    params.add_number('intermediate_dimensions', int, 100, (1000, 600))
    params.add_number('dropout_probability', float, 1.0, (0.0, 1.0))
    params.add_random_seed('seed')
    return params


def create_params_pure_svd() -> ConfigParameters:
    """Create the parameters of the PureSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 10, (1, 100))
    params.add_random_seed('seed')
    return params


def create_params_svd_pp() -> ConfigParameters:
    """Create the parameters of the SVDpp algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('iterations', int, 10, (1, 50))
    params.add_number('factors', int, 50, (1, 100))
    params.add_number('learning_rate', float, 0.001, (0.0001, 1.0))
    params.add_number('regularization_factors', float, 0.1, (0.0001, 1.0))
    params.add_number('regularization_bias', float, 0.001, (0.0001, 1.0))
    params.add_random_seed('seed')
    return params
