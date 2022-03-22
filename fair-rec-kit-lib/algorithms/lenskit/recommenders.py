""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from lenskit.algorithms import Recommender

from ..recommender import RecommenderAlgorithm
from .algorithms import *


class RecommenderLensKit(RecommenderAlgorithm):

    def __init__(self, algo, params):
        RecommenderAlgorithm.__init__(self, Recommender.adapt(algo), params)

    def train(self, train_set):
        self._algo.fit(train_set)

    def recommend(self, user, num_items=10):
        return self._algo.recommend(user, n=num_items)


def create_recommender_als_biased_mf(params):
    return RecommenderLensKit(create_als_biased_mf(params), params)


def create_recommender_als_implicit_mf(params):
    return RecommenderLensKit(create_als_implicit_mf(params), params)


def create_recommender_pop_score(params):
    return RecommenderLensKit(create_pop_score(params), params)


def create_recommender_bias(params):
    return RecommenderLensKit(create_bias(params), params)


def create_recommender_funk_svd(params):
    return RecommenderLensKit(create_funk_svd(params), params)


def create_recommender_knn_item_item(params):
    return RecommenderLensKit(create_knn_item_item(params), params)


def create_recommender_knn_user_user(params):
    return RecommenderLensKit(create_knn_user_user(params), params)
