"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Dict

import pandas as pd
from surprise.dataset import Dataset
from surprise.prediction_algorithms import AlgoBase
from surprise.prediction_algorithms import BaselineOnly
from surprise.prediction_algorithms import CoClustering
from surprise.prediction_algorithms import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise.prediction_algorithms import NMF
from surprise.prediction_algorithms import NormalPredictor
from surprise.prediction_algorithms import SlopeOne
from surprise.prediction_algorithms import SVD, SVDpp
from surprise.reader import Reader

from ..base_predictor import Predictor


class SurprisePredictor(Predictor):
    """Predictor implementation for the Surprise package."""

    def __init__(self, algo: AlgoBase, name: str, params: Dict[str, Any], **kwargs):
        """Construct the surprise predictor.

        Args:
            algo: the surprise prediction algorithm.
            name: the name of the predictor.
            params: the parameters of the predictor.

        Keyword Args:
            num_threads(int): the max number of threads the predictor can use.
            rating_scale(Tuple[float, float]): consisting of the min_rating and max_rating on
                which the algorithm will perform training.
        """
        Predictor.__init__(self, name, params, kwargs['num_threads'])
        self.algo = algo
        self.rating_scale = kwargs['rating_scale']

    def train(self, train_set: pd.DataFrame) -> None:
        """Train the algorithm on the specified train set.

        Surprise predictors expect the train set as a Dataset class,
        so the original train set needs to be converted before training can be done.

        Args:
            train_set: with at least three columns: 'user', 'item', 'rating'.
        """
        reader = Reader(rating_scale=self.rating_scale)
        dataset = Dataset.load_from_df(train_set, reader)
        self.algo.fit(dataset.build_full_trainset())

    def predict(self, user: int, item: int) -> float:
        """Compute a prediction for the specified user and item.

        Surprise predictors clip the predicted ratings by default to the original rating scale
        that is provided during training. It is turned off to conform with the expected interface.

        Args:
            user: the user ID.
            item: the item ID.

        Returns:
            the prediction rating.
        """
        prediction = self.algo.predict(user, item, clip=False)
        return prediction.est


def create_baseline_only_als(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the BaselineOnly ALS predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): The number of iteration of the ALS procedure.
            reg_i(int): the regularization parameter for items.
            reg_u(int): The regularization parameter for items.

    Returns:
        the SurprisePredictor wrapper of BaselineOnly with method 'als'.
    """
    algo = BaselineOnly(
        bsl_options={
            'method': 'als',
            'reg_i': params['reg_i'],
            'reg_u': params['reg_u'],
            'n_epochs': params['epochs']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)



def create_baseline_only_sgd(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the BaselineOnly SGD predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): the number of iteration of the SGD procedure.
            regularization(float): the regularization parameter
                of the cost function that is optimized.
            learning_rate(float): the learning rate of SGD.

    Returns:
        the SurprisePredictor wrapper of BaselineOnly with method 'sgd'.
    """
    algo = BaselineOnly(
        bsl_options={
            'method': 'sgd',
            'reg': params['regularization'],
            'learning_rate': params['learning_rate'],
            'n_epochs': params['epochs']
         },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def create_co_clustering(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the CoClustering predictor.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            epochs(int): number of iteration of the optimization loop.
            user_clusters(int): number of user clusters.
            item_clusters(int): number of item clusters.
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the SurprisePredictor wrapper of CoClustering.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = CoClustering(
        n_cltr_u=params['user_clusters'],
        n_cltr_i=params['item_clusters'],
        n_epochs=params['epochs'],
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)



def create_knn_basic(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the KNNBasic predictor.

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
        the SurprisePredictor wrapper of KNNBasic.
    """
    algo = KNNBasic(
        k=params['max_k'],
        min_k=params['min_k'],
        sim_options={
            'name': params['similarity'],
            'user_based': params['user_based'],
            'min_support': params['min_support']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def create_knn_baseline_als(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the KNNBaseline ALS predictor.

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
        the SurprisePredictor wrapper of KNNBaseline with method 'als'.
    """
    algo = KNNBaseline(
        k=params['max_k'],
        min_k=params['min_k'],
        bsl_options={
            'name': 'als',
            'reg_i': params['reg_i'],
            'reg_u': params['reg_u'],
            'n_epochs': params['epochs']
        },
        sim_options={
            'name': 'pearson_baseline',
            'user_based': params['user_based'],
            'min_support': params['min_support']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)



def create_knn_baseline_sgd(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the KNNBaseline SGD predictor.

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
        the SurprisePredictor wrapper of KNNBaseline with method 'sgd'.
    """
    algo = KNNBaseline(
        k=params['max_k'],
        min_k=params['min_k'],
        bsl_options={
            'method': 'sgd',
            'reg': params['regularization'],
            'learning_rate': params['learning_rate'],
            'n_epochs': params['epochs']
         },
        sim_options={
            'name': 'pearson_baseline',
            'user_based': params['user_based'],
            'min_support': params['min_support'],
            'shrinkage': params['shrinkage']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)



def create_knn_with_means(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the KNNWithMeans predictor.

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
        the SurprisePredictor wrapper of KNNWithMeans.
    """
    algo = KNNWithMeans(
        k=params['max_k'],
        min_k=params['min_k'],
        sim_options={
            'name': params['similarity'],
            'user_based': params['user_based'],
            'min_support': params['min_support']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)



def create_knn_with_zscore(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the KNNWithZScore predictor.

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
        the SurprisePredictor wrapper of KNNWithZScore.
    """
    algo = KNNWithZScore(
        k=params['max_k'],
        min_k=params['min_k'],
        sim_options={
            'name': params['similarity'],
            'user_based': params['user_based'],
            'min_support': params['min_support']
        },
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def create_nmf(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the NMF predictor.

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
        the SurprisePredictor wrapper of NMF.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = NMF(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        biased=False,
        reg_pu=params['reg_pu'],
        reg_qi=params['reg_qi'],
        init_low=params['init_low'],
        init_high=params['init_high'],
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def create_normal_predictor(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the NormalPredictor.

    Args:
        name: the name of the algorithm.
        params: there are no parameters for this algorithm.

    Returns:
        the SurprisePredictor wrapper of NormalPredictor.
    """
    return SurprisePredictor(NormalPredictor(), name, params, **kwargs)


def create_slope_one(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the SlopeOne predictor.

    Args:
        name: the name of the algorithm.
        params: there are no parameters for this algorithm.

    Returns:
        the SurprisePredictor wrapper of SlopeOne.
    """
    return SurprisePredictor(SlopeOne(), name, params, **kwargs)


def create_svd(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the SVD predictor.

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
        the SurprisePredictor wrapper of SVD.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = SVD(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        biased=params['biased'],
        init_mean=params['init_mean'],
        init_std_dev=params['init_std_dev'],
        lr_all=params['learning_rate'],
        reg_all=params['regularization'],
        lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
        reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)


def create_svd_pp(name: str, params: Dict[str, Any], **kwargs) -> SurprisePredictor:
    """Create the SVDpp predictor.

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
        the SurprisePredictor wrapper of SVDpp.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = SVDpp(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        init_mean=params['init_mean'],
        init_std_dev=params['init_std_dev'],
        lr_all=params['learning_rate'],
        reg_all=params['regularization'],
        lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None,
        reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None,
        random_state=params['random_seed'],
        verbose=False
    )

    return SurprisePredictor(algo, name, params, **kwargs)
