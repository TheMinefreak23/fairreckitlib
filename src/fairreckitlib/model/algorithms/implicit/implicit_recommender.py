"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Dict, List

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.recommender_base import RecommenderBase

import numpy as np
import pandas as pd
from scipy import sparse

from ..base_recommender import Recommender


class ImplicitRecommender(Recommender):
    """Recommender implementation for the Implicit package."""

    def __init__(self, algo: RecommenderBase, name: str, params: Dict[str, Any], **kwargs):
        """Construct the implicit recommender.

        Args:
            algo: the implicit recommender algorithm.
            name: the name of the recommender.
            params: the parameters of the recommender.

        Keyword Args:
            num_threads(int): the max number of threads the recommender can use.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.
        """
        Recommender.__init__(self, name, params, kwargs['num_threads'],
                             kwargs['rated_items_filter'])
        self.algo = algo
        self.csr_train_set = None

    def on_train(self) -> None:
        """Train the algorithm on the train set.

        Implicit recommenders expect a CSR matrix, convert the train set and store
        it for recommending items.
        """
        self.csr_train_set = sparse.csr_matrix(
            (self.train_set['rating'], (self.train_set['user'], self.train_set['item']))
        )

        self.algo.fit(self.csr_train_set, False)

    def on_recommend(self, user: int, num_items: int) -> pd.DataFrame:
        """Compute item recommendations for the specified user.

        Implicit recommenders use the stored CSR train set to produce item recommendations.

        Args:
            user: the user ID to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            dataframe with the columns: 'item' and 'score'.
        """
        items, scores = self.algo.recommend(
            user,
            self.csr_train_set[user],
            N=num_items,
            filter_already_liked_items=self.rated_items_filter
        )

        return pd.DataFrame({ 'item': items, 'score': scores })

    def on_recommend_batch(self, users: List[int], num_items: int) -> pd.DataFrame:
        """Compute the items recommendations for each of the specified users.

        Implicit recommenders use the stored CSR train set to produce item recommendations.
        Moreover, they allow for batching multiple users at the same time using multiple threads.

        Args:
            users: the user ID's to compute recommendations for.
            num_items: the number of item recommendations to produce.

        Returns:
            dataframe with the columns: 'rank', 'user', 'item', 'score'.
        """
        items, scores = self.algo.recommend(
            users,
            self.csr_train_set[users],
            N=num_items,
            filter_already_liked_items=True
        )

        result = pd.DataFrame()
        num_users = len(users)
        for i in range(num_users):
            result = result.append(pd.DataFrame({
                'rank': np.arange(1, 1 + num_items),
                'user': np.full(num_items, users[i]),
                'item': items[i],
                'score': scores[i]
            }), ignore_index=True)

        return result


def create_als(name: str, params: Dict[str, Any], **kwargs) -> ImplicitRecommender:
    """Create the AlternatingLeastSquares recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): the number of latent factors to compute.
            regularization(float): the regularization factor to use.
            use_native(bool): use native extensions to speed up model fitting.
            use_cg(bool): use a faster Conjugate Gradient solver to calculate factors.
            iterations(int): the number of ALS iterations to use when fitting data.
            calculate_training_loss(bool): whether to log out the training loss at each iteration.
            random_seed(int): the random seed or None for the current time as seed.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the ImplicitRecommender wrapper of AlternatingLeastSquares.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = AlternatingLeastSquares(
        factors=params['factors'],
        regularization=params['regularization'],
        dtype=np.float32,
        use_native=params['use_native'],
        use_cg=params['use_cg'],
        iterations=params['iterations'],
        calculate_training_loss=params['calculate_training_loss'],
        num_threads=kwargs['num_threads'],
        random_state=params['random_seed']
    )

    return ImplicitRecommender(algo, name, params, **kwargs)


def create_bpr(name: str, params: Dict[str, Any], **kwargs) -> ImplicitRecommender:
    """Create the BayesianPersonalizedRanking recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): the number of latent factors to compute.
            learning_rate(float): the learning rate to apply for SGD updates during training.
            regularization(float): the regularization factor to use.
            iterations(int): the number of training epochs to use when fitting the data.
            verify_negative_samples(bool): when sampling negative items, check if the randomly
                picked negative item has actually been liked by the user. This check increases
                the time needed to train but usually leads to better predictions.
            random_seed(int): the random seed or None for the current time as seed.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the ImplicitRecommender wrapper of BayesianPersonalizedRanking.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = BayesianPersonalizedRanking(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        num_threads=kwargs['num_threads'],
        verify_negative_samples=params['verify_negative_samples'],
        random_state=params['random_seed']
    )

    return ImplicitRecommender(algo, name, params, **kwargs)


def create_lmf(name: str, params: Dict[str, Any], **kwargs) -> ImplicitRecommender:
    """Create the LogisticMatrixFactorization recommender.

    Args:
        name: the name of the algorithm.
        params: containing the following name-value pairs:
            factors(int): the number of latent factors to compute.
            learning_rate(float): the learning rate to apply for updates during training.
            regularization(float): the regularization factor to use.
            iterations(int): the number of training epochs to use when fitting the data.
            neg_prop(int): the proportion of negative samples.
            random_seed(int): the random seed or None for the current time as seed.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        the ImplicitRecommender wrapper of LogisticMatrixFactorization.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    algo = LogisticMatrixFactorization(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        neg_prop=params['neg_prop'],
        num_threads=kwargs['num_threads'],
        random_state=params['random_seed']
    )

    return ImplicitRecommender(algo, name, params, **kwargs)
