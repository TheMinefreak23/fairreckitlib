""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender, BM25Recommender
import numpy as np
import pandas as pd
from scipy import sparse

from ..recommender import RecommenderAlgorithm


class RecommenderImplicit(RecommenderAlgorithm):

    def __init__(self, algo, params):
        RecommenderAlgorithm.__init__(self, algo, params)
        self._train_user_items = None

    def train(self, train_set):
        self._train_user_items = sparse.csr_matrix(
            (train_set['rating'], (train_set['user'], train_set['item']))
        ).T.tocsr()

        self._algo.fit(self._train_user_items)

    def recommend(self, user, num_items=10):
        items, scores = self._algo.recommend(
            user,
            self._train_user_items[user],
            N=num_items,
            filter_already_liked_items=True
        )

        return pd.DataFrame({ 'item': items, 'score': scores })


def create_recommender_als(params):
    return RecommenderImplicit(AlternatingLeastSquares(
        factors=params['factors'],
        regularization=params['regularization'],
        dtype=np.float32,
        use_native=params['use_native'],
        use_cg=params['use_cg'],
        iterations=params['iterations'],
        calculate_training_loss=params['calculate_training_loss'],
        num_threads=0,
        random_state=None
    ), params)


def create_recommender_bpr(params):
    return RecommenderImplicit(BayesianPersonalizedRanking(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        num_threads=0,
        verify_negative_samples=params['verify_negative_samples'],
        random_state=None
    ), params)


def create_recommender_lmf(params):
    return RecommenderImplicit(LogisticMatrixFactorization(
        factors=params['factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        dtype=np.float32,
        iterations=params['iterations'],
        neg_prop=params['neg_prop'],
        num_threads=0,
        random_state=None
    ), params)


def create_recommender_cosine(params):
    return RecommenderImplicit(CosineRecommender(
        K=params['K'],
        num_threads=0
    ), params)


def create_recommender_tfidf(params):
    return RecommenderImplicit(TFIDFRecommender(
        K=params['K'],
        num_threads=0
    ), params)


def create_recommender_bm25(params):
    return RecommenderImplicit(BM25Recommender(
        K=params['K'],
        K1=params['K1'],
        B=params['B'],
        num_threads=0
    ), params)
