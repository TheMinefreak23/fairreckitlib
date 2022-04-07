"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import numpy as np
from lenskit.algorithms import Recommender
import lenskit.batch as batch

from ..recommender import RecommenderAlgorithm
from .algorithms import *


class RecommenderLensKit(RecommenderAlgorithm):

    def __init__(self, algo, params, **kwargs):
        RecommenderAlgorithm.__init__(self, Recommender.adapt(algo), params, **kwargs)

    def train(self, train_set):
        self.algo.fit(train_set)

    def recommend(self, user, num_items=10):
        return self.algo.recommend(user, n=num_items)

    def recommend_batch(self, users, num_items=10):
        n_jobs = self.num_threads if self.num_threads > 0 else None
        recs = batch.recommend(self.algo, users, num_items, n_jobs=n_jobs)

        # random algo does not produce a score
        if recs.get('score') is None:
            recs['score'] = np.full(len(users) * num_items, 1)

        return recs[['user', 'item', 'score']]


def create_recommender_biased_mf(params, **kwargs):
    return RecommenderLensKit(create_biased_mf(params), params, **kwargs)


def create_recommender_implicit_mf(params, **kwargs):
    return RecommenderLensKit(create_implicit_mf(params), params, **kwargs)


def create_recommender_item_item(params, **kwargs):
    return RecommenderLensKit(create_item_item(params, kwargs['rating_type']), params, **kwargs)


def create_recommender_pop_score(params, **kwargs):
    return RecommenderLensKit(create_pop_score(params), params, **kwargs)


def create_recommender_random(params, **kwargs):
    return RecommenderLensKit(create_random(params), params, **kwargs)


def create_recommender_user_user(params, **kwargs):
    return RecommenderLensKit(create_user_user(params, kwargs['rating_type']), params, **kwargs)
