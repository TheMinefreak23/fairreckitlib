"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import numpy as np
from lenskit.algorithms import Recommender
from lenskit.batch import recommend

from ..recommender import RecommenderAlgorithm
from .algorithms import *


class RecommenderLensKit(RecommenderAlgorithm):

    def __init__(self, algo, params):
        RecommenderAlgorithm.__init__(self, Recommender.adapt(algo), params)

    def train(self, train_set):
        self._algo.fit(train_set)

    def recommend(self, user, num_items=10):
        return self._algo.recommend(user, n=num_items)

    def recommend_batch(self, users, num_items=10):
        recs = recommend(self._algo, users, num_items)

        # random algo does not produce a score
        if recs.get('score') is None:
            recs['score'] = np.full(len(users) * num_items, 1)

        return recs[['user', 'item', 'score']]


def create_recommender_biased_mf(params):
    return RecommenderLensKit(create_biased_mf(params), params)


def create_recommender_implicit_mf(params):
    return RecommenderLensKit(create_implicit_mf(params), params)


def create_recommender_pop_score(params):
    return RecommenderLensKit(create_pop_score(params), params)


def create_recommender_random(params):
    return RecommenderLensKit(create_random(params), params)
