"""This module contains the parameter creation functions for surprise predictors/recommenders.

Functions:

    add_baseline_als_params_to: add baseline config parameters that use ALS.
    add_baseline_sgd_params_to: add baseline config parameters that use SGD.
    create_params_baseline_only_als: create BaselineOnly config parameters with ALS.
    create_params_baseline_only_sgd: create BaselineOnly config parameters with SGD.
    create_params_co_clustering: create CoClustering config parameters.
    create_params_knn: create KNN base config parameters.
    create_params_knn_baseline: create KNNBaseline base config parameters.
    create_params_knn_baseline_als: create KNNBaseline config parameters with ALS.
    create_params_knn_baseline_sgd: create KNNBaseline config parameters with SGD.
    create_params_knn_similarities: create KNN config parameters with similarities.
    create_params_nmf: create NMF config parameters.
    create_params_svd: create SVD config parameters.
    create_params_svd_pp: create SVDpp config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.config.config_parameters import ConfigParameters


def add_baseline_als_params_to(params: ConfigParameters) -> ConfigParameters:
    """Add the parameters of the Baseline options with ALS.

    Args:
        params: the parameters to add the baseline options to.

    Returns:
        the configuration parameters of the algorithm.
    """
    params.add_number('epochs', int, 10, (1, 50))
    params.add_number('reg_i', int, 10, (1, 100))
    params.add_number('reg_u', int, 15, (1, 100))
    return params


def add_baseline_sgd_params_to(params: ConfigParameters) -> ConfigParameters:
    """Add the parameters of the Baseline options with SGD.

    Args:
        params: the parameters to add the baseline options to.

    Returns:
        the configuration parameters of the algorithm.
    """
    params.add_number('epochs', int, 20, (1, 50))
    params.add_number('regularization', float, 0.02, (0.00001, 1.0))
    params.add_number('learning_rate', float, 0.005, (0.0001, 1.0))
    return params


def create_params_baseline_only_als() -> ConfigParameters:
    """Create the parameters of the BaselineOnly ALS algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    return add_baseline_als_params_to(ConfigParameters())


def create_params_baseline_only_sgd() -> ConfigParameters:
    """Create the parameters of the BaselineOnly SGD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    return add_baseline_sgd_params_to(ConfigParameters())


def create_params_co_clustering() -> ConfigParameters:
    """Create the parameters of the CoClustering algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('epochs', int, 20, (1, 50))
    params.add_number('user_clusters', int, 3, (0, 30))
    params.add_number('item_clusters', int, 3, (0, 30))
    params.add_random_seed('random_seed')
    return params


def create_params_knn() -> ConfigParameters:
    """Create the base parameters of all KNN algorithms.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('max_k', int, 40, (1, 100))
    params.add_number('min_k', int, 1, (1, 100))
    params.add_bool('user_based', True)
    params.add_number('min_support', int, 1, (1, 100))
    return params


def create_params_knn_baseline() -> ConfigParameters:
    """Create the base parameters of both KNN Baseline algorithms.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = create_params_knn()
    params.add_number('shrinkage', int, 100, (1, 1000))
    return params


def create_params_knn_baseline_als() -> ConfigParameters:
    """Create the parameters of the KNN Baseline algorithm with ALS.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = create_params_knn_baseline()
    return add_baseline_als_params_to(params)


def create_params_knn_baseline_sgd() -> ConfigParameters:
    """Create the parameters of the KNN Baseline algorithm with SGD.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = create_params_knn_baseline()
    return add_baseline_sgd_params_to(params)


def create_params_knn_similarities() -> ConfigParameters:
    """Create the parameters of the KNN algorithm with similarities.

    Returns:
        the configuration parameters of the algorithm.
    """
    similarities = ['MSD', 'cosine', 'pearson']

    params = create_params_knn()
    params.add_single_option('similarity', str, similarities[0], similarities)
    return params


def create_params_nmf() -> ConfigParameters:
    """Create the parameters of the NMF algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 15, (1, 100))
    params.add_number('epochs', int, 50, (1, 50))
    params.add_number('reg_pu', float, 0.06, (0.00001, 1.0))
    params.add_number('reg_qi', float, 0.06, (0.00001, 1.0))
    params.add_number('init_low', int, 0, (0, 100))
    params.add_number('init_high', int, 1, (0, 100))
    params.add_random_seed('random_seed')
    return params


def create_params_svd() -> ConfigParameters:
    """Create the parameters of the SVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 100, (1, 100))
    params.add_number('epochs', int, 20, (1, 50))
    params.add_number('init_mean', int, 0, (-1000, 1000))
    params.add_number('init_std_dev', float, 0.1, (0.0, 1.0))
    params.add_number('learning_rate', float, 0.005, (0.00001, 1.0))
    params.add_number('regularization', float, 0.02, (0.00001, 1.0))
    params.add_random_seed('random_seed')
    params.add_bool('biased', True)
    return params


def create_params_svd_pp() -> ConfigParameters:
    """Create the parameters of the SVDpp algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('factors', int, 20, (1, 100))
    params.add_number('epochs', int, 20, (1, 50))
    params.add_number('init_mean', int, 0, (-1000, 1000))
    params.add_number('init_std_dev', float, 0.1, (0.0, 1.0))
    params.add_number('learning_rate', float, 0.007, (0.00001, 1.0))
    params.add_number('regularization', float, 0.02, (0.00001, 1.0))
    params.add_random_seed('random_seed')
    return params
