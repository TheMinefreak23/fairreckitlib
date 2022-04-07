"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

from ..recommender import RecommenderAlgorithm


class RecommenderElliot(RecommenderAlgorithm):

    def __init__(self, algo, params, **kwargs):
        RecommenderAlgorithm.__init__(self, algo, params, **kwargs)

    def train(self, train_set):
        raise NotImplementedError()

    def recommend(self, user, num_items=10):
        raise NotImplementedError()

    def recommend_batch(self, users, num_items=10):
        raise NotImplementedError()


def create_recommender_funk_svd(params, **kwargs):
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return RecommenderElliot(None, params, **kwargs)


def create_recommender_item_knn(params, **kwargs):
    return RecommenderElliot(None, params, **kwargs)


def create_recommender_most_pop(params, **kwargs):
    return RecommenderElliot(None, params, **kwargs)


def create_recommender_multi_vae(params, **kwargs):
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return RecommenderElliot(None, params, **kwargs)


def create_recommender_pure_svd(params, **kwargs):
    return RecommenderElliot(None, params, **kwargs)


def create_recommender_random(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return RecommenderElliot(None, params, **kwargs)


def create_recommender_svd_pp(params, **kwargs):
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return RecommenderElliot(None, params, **kwargs)


def create_recommender_user_knn(params, **kwargs):
    return RecommenderElliot(None, params, **kwargs)
