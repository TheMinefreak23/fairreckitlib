"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import numpy as np
import pandas as pd
from scipy import sparse

from ..recommender import RecommenderAlgorithm


class RecommenderImplicit(RecommenderAlgorithm):

    def __init__(self, algo, params, **kwargs):
        RecommenderAlgorithm.__init__(self, algo, params, **kwargs)
        self.__train_user_items = None

    def train(self, train_set):
        self.__train_user_items = sparse.csr_matrix(
            (train_set['rating'], (train_set['user'], train_set['item']))
        )

        self.algo.fit(self.__train_user_items, False)

    def recommend(self, user, num_items=10):
        items, scores = self.algo.recommend(
            user,
            self.__train_user_items[user],
            N=num_items,
            filter_already_liked_items=True
        )

        return pd.DataFrame({ 'item': items, 'score': scores })

    def recommend_batch(self, users, num_items=10):
        items, scores = self.algo.recommend(
            users,
            self.__train_user_items[users],
            N=num_items,
            filter_already_liked_items=True
        )

        result = pd.DataFrame()

        for i in range(len(users)):
            result = result.append(pd.DataFrame({
                'user': np.full(num_items, users[i]),
                'item': items[i],
                'score': scores[i]
            }), ignore_index=True)

        return result


def create_recommender_alternating_least_squares(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return RecommenderImplicit(AlternatingLeastSquares(
        factors=params['factors'],
        regularization=params['regularization'],
        dtype=np.float32,
        use_native=params['use_native'],
        use_cg=params['use_cg'],
        iterations=params['iterations'],
        calculate_training_loss=params['calculate_training_loss'],
        num_threads=kwargs['num_threads'],
        random_state=params['random_seed']
    ), params, **kwargs)


def create_recommender_bayesian_personalized_ranking(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return RecommenderImplicit(BayesianPersonalizedRanking(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        num_threads=kwargs['num_threads'],
        verify_negative_samples=params['verify_negative_samples'],
        random_state=params['random_seed']
    ), params, **kwargs)


def create_recommender_logistic_matrix_factorization(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return RecommenderImplicit(LogisticMatrixFactorization(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        neg_prop=params['neg_prop'],
        num_threads=kwargs['num_threads'],
        random_state=params['random_seed']
    ), params, **kwargs)
