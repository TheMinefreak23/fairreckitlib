"""This module contains the surprise top k predictor creation functions.

Functions:

    create_baseline_only_als: create BaselineOnly ALS recommender (factory creation compatible).
    create_baseline_only_sgd: create BaselineOnly SGD recommender (factory creation compatible).
    create_co_clustering: create CoClustering recommender (factory creation compatible).
    create_knn_basic: create KNNBasic recommender (factory creation compatible).
    create_knn_baseline_als: create KNNBaseline ALS recommender (factory creation compatible).
    create_knn_baseline_sgd: create KNNBaseline SGD recommender (factory creation compatible).
    create_knn_with_means: create KNNWithMeans recommender (factory creation compatible).
    create_knn_with_zscore: create KNNWithZScore recommender (factory creation compatible).
    create_nmf: create NMF recommender (factory creation compatible).
    create_normal_predictor: create NormalPredictor recommender (factory creation compatible).
    create_slope_one: create SlopeOne recommender (factory creation compatible).
    create_svd: create SVD recommender (factory creation compatible).
    create_svd_pp: create SVDpp recommender (factory creation compatible).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Any, Dict

from ..top_k_recommender import TopK
from . import surprise_predictor


def create_baseline_only_als(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the BaselineOnly ALS recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): The number of iteration of the ALS procedure.
            reg_i(int): the regularization parameter for items.
            reg_u(int): The regularization parameter for items.

    Returns:
        the SurprisePredictor wrapper of BaselineOnly with method 'als' as a TopK recommender.
    """
    predictor = surprise_predictor.create_baseline_only_als(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_baseline_only_sgd(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the BaselineOnly SGD recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): the number of iteration of the SGD procedure.
            regularization(float): the regularization parameter
                of the cost function that is optimized.
            learning_rate(float): the learning rate of SGD.

    Returns:
        the SurprisePredictor wrapper of BaselineOnly with method 'sgd' as a TopK recommender.
    """
    predictor = surprise_predictor.create_baseline_only_sgd(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_co_clustering(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the CoClustering recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): number of iteration of the optimization loop.
            user_clusters(int): number of user clusters.
            item_clusters(int): number of item clusters.
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the SurprisePredictor wrapper of CoClustering as a TopK recommender.
    """
    predictor = surprise_predictor.create_co_clustering(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_knn_basic(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the KNNBasic recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_k(int): the maximum number of neighbors to take into account for aggregation.
            min_k(int): the minimum number of neighbors to take into account for aggregation.
            user_based(bool): whether similarities will be computed between users or between
                items, this has a huge impact on the performance.
            min_support(int): the minimum number of common items or users, depending on the
                user_based parameter.
            similarity(str): the name of the similarity to use ('MSD', 'cosine' or 'pearson').

    Returns:
        the SurprisePredictor wrapper of KNNBasic as a TopK recommender.
    """
    predictor = surprise_predictor.create_knn_basic(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_knn_baseline_als(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the KNNBaseline ALS recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_k(int): the maximum number of neighbors to take into account for aggregation.
            min_k(int): the minimum number of neighbors to take into account for aggregation.
            user_based(bool): whether similarities will be computed between users or between
                items, this has a huge impact on the performance.
            min_support(int): the minimum number of common items or users, depending on the
                user_based parameter.
            epochs(int): The number of iteration of the ALS procedure.
            reg_i(int): the regularization parameter for items.
            reg_u(int): The regularization parameter for items.

    Returns:
        the SurprisePredictor wrapper of KNNBaseline with method 'als' as a TopK recommender.
    """
    predictor = surprise_predictor.create_knn_baseline_als(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_knn_baseline_sgd(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the KNNBaseline SGD recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_k(int): the maximum number of neighbors to take into account for aggregation.
            min_k(int): the minimum number of neighbors to take into account for aggregation.
            user_based(bool): whether similarities will be computed between users or between
                items, this has a huge impact on the performance.
            min_support(int): the minimum number of common items or users, depending on the
                user_based parameter.
            shrinkage(int): shrinkage parameter to apply.
            epochs(int): the number of iteration of the SGD procedure.
            regularization(float): the regularization parameter
                of the cost function that is optimized.
            learning_rate(float): the learning rate of SGD.

    Returns:
        the SurprisePredictor wrapper of KNNBaseline with method 'sgd' as a TopK recommender.
    """
    predictor = surprise_predictor.create_knn_baseline_sgd(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_knn_with_means(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the KNNWithMeans recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_k(int): the maximum number of neighbors to take into account for aggregation.
            min_k(int): the minimum number of neighbors to take into account for aggregation.
            user_based(bool): whether similarities will be computed between users or between
                items, this has a huge impact on the performance.
            min_support(int): the minimum number of common items or users, depending on the
                user_based parameter.
            similarity(str): the name of the similarity to use ('MSD', 'cosine' or 'pearson').

    Returns:
        the SurprisePredictor wrapper of KNNWithMeans as a TopK recommender.
    """
    predictor = surprise_predictor.create_knn_with_means(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_knn_with_zscore(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the KNNWithZScore recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            max_k(int): the maximum number of neighbors to take into account for aggregation.
            min_k(int): the minimum number of neighbors to take into account for aggregation.
            user_based(bool): whether similarities will be computed between users or between
                items, this has a huge impact on the performance.
            min_support(int): the minimum number of common items or users, depending on the
                user_based parameter.
            similarity(str): the name of the similarity to use ('MSD', 'cosine' or 'pearson').

    Returns:
        the SurprisePredictor wrapper of KNNWithZScore as a TopK recommender.
    """
    predictor = surprise_predictor.create_knn_with_zscore(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_nmf(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the NMF recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): the number of factors.
            epochs(int): the number of iteration of the SGD procedure.
            reg_pu(float): the regularization term for users.
            reg_qi(float): the regularization term for items.
            init_low(int): lower bound for random initialization of factors.
            init_high(int): higher bound for random initialization of factors.
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the SurprisePredictor wrapper of NMF as a TopK recommender.
    """
    predictor = surprise_predictor.create_nmf(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_normal_predictor(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the NormalPredictor recommender.

    Args:
        name: the name of the algorithm.
        params: there are no parameters for this algorithm.

    Returns:
        the SurprisePredictor wrapper of NormalPredictor as a TopK recommender.
    """
    predictor = surprise_predictor.create_normal_predictor(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_slope_one(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the SlopeOne recommender.

    Args:
        name: the name of the algorithm.
        params: there are no parameters for this algorithm.

    Returns:
        the SurprisePredictor wrapper of SlopeOne as a TopK recommender.
    """
    predictor = surprise_predictor.create_slope_one(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_svd(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the SVD recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): the number of factors.
            epochs(int): the number of iteration of the SGD procedure.
            biased(bool): whether to use baselines (or biases).
            init_mean(int): the mean of the normal distribution for factor vectors initialization.
            init_std_dev(float): the standard deviation of the normal distribution for
                factor vectors initialization.
            learning_rate(float): the learning rate for users and items.
            regularization(float): the regularization term for users and items.
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the SurprisePredictor wrapper of SVD as a TopK recommender.
    """
    predictor = surprise_predictor.create_svd(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])


def create_svd_pp(name: str, params: Dict[str, Any], **kwargs) -> TopK:
    """Create the SVDpp recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): the number of factors.
            epochs(int): the number of iteration of the SGD procedure.
            init_mean(int): the mean of the normal distribution for factor vectors initialization.
            init_std_dev(float): the standard deviation of the normal distribution for
                factor vectors initialization.
            learning_rate(float): the learning rate for users and items.
            regularization(float): the regularization term for users and items.
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the SurprisePredictor wrapper of SVDpp as a TopK recommender.
    """
    predictor = surprise_predictor.create_svd_pp(name, params, **kwargs)
    return TopK(predictor, kwargs['rated_items_filter'])
