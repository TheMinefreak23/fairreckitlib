"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.config_params import ConfigParameters


def create_params_funk_svd() -> ConfigParameters:
    """Create the parameters of the FunkSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
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
    params.add_value('neighbours', int, 40, (1, 50))
    params.add_option('similarity', str, similarities[0], similarities)
    params.add_option('implementation', str, implementations[0], implementations)
    return params


def create_params_multi_vae() -> ConfigParameters:
    """Create the parameters of the MultiVAE algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_value('epochs', int, 10, (1, 50))
    params.add_value('batch_size', int, 512, (100, 10000))
    params.add_value('lr', float, 0.001, (0.0001, 1.0))
    params.add_value('reg_lambda', float, 0.01, (0.0001, 1.0))
    params.add_value('intermediate_dim', int, 100, (1000, 600))
    params.add_value('latent_dim', int, 200, (100, 500))
    params.add_value('dropout_pkeep', float, 1.0, (0.0, 1.0))
    params.add_random_seed('seed')
    return params


def create_params_pure_svd() -> ConfigParameters:
    """Create the parameters of the PureSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_value('factors', int, 10, (1, 100))
    params.add_random_seed('seed')
    return params


def create_params_svd_pp() -> ConfigParameters:
    """Create the parameters of the SVDpp algorithm.

    Returns:
        the configuration parameters of the algorithm.
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
