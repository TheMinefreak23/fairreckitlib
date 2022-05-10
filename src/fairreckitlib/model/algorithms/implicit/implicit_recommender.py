"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import numpy as np
import pandas as pd
from scipy import sparse

from ..base_recommender import Recommender


class ImplicitRecommender(Recommender):
    """Recommender implementation for Implicit.

    Args:
        recommender(implicit.RecommenderBase): the recommender algorithm.
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
        self.__train_user_items = None

    def get_name(self):
        return self.__name

    def get_params(self):
        return dict(self.__params)

    def train(self, train_set):
        self.__train_user_items = sparse.csr_matrix(
            (train_set['rating'], (train_set['user'], train_set['item']))
        )

        self.__recommender.fit(self.__train_user_items, False)

    def recommend(self, user, num_items=10):
        items, scores = self.__recommender.recommend(
            user,
            self.__train_user_items[user],
            N=num_items,
            filter_already_liked_items=self.rated_items_filter
        )

        return pd.DataFrame({ 'item': items, 'score': scores })

    def recommend_batch(self, users, num_items=10):
        items, scores = self.__recommender.recommend(
            users,
            self.__train_user_items[users],
            N=num_items,
            filter_already_liked_items=True
        )

        result = pd.DataFrame()
        num_users = len(users)
        for i in range(num_users):
            result = result.append(pd.DataFrame({
                'user': np.full(num_items, users[i]),
                'item': items[i],
                'score': scores[i]
            }), ignore_index=True)

        return result
