"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import numpy as np
from lenskit import batch

from ..base_recommender import Recommender


class LensKitRecommender(Recommender):
    """Recommender implementation for LensKit.

    Args:
        recommender(lenskit.Recommender): the recommender algorithm.
        name(str): the name of the algorithm.
        params(dict): the parameters of the algorithm.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.
    """
    def __init__(self, recommender, name, params, **kwargs):
        Recommender.__init__(self, **kwargs)
        self.__recommender = recommender
        self.__name = name
        self.__params = params

    def get_name(self):
        return self.__name

    def get_params(self):
        return dict(self.__params)

    def train(self, train_set):
        self.__recommender.fit(train_set)

    def recommend(self, user, num_items=10):
        return self.__recommender.recommend(user, n=num_items)

    def recommend_batch(self, users, num_items=10):
        n_jobs = self.num_threads if self.num_threads > 0 else None
        recs = batch.recommend(self.__recommender, users, num_items, n_jobs=n_jobs)

        # random algo does not produce a score
        if recs.get('score') is None:
            recs['score'] = np.full(len(users) * num_items, 1)

        return recs[['user', 'item', 'score']]
