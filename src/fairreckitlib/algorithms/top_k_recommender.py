"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import pandas as pd

from .base_recommender import Recommender


class TopK(Recommender):
    """Recommender that implements top K recommendation using a predictor.

    Args:
        predictor(Predictor): the underlying predictor to use for recommendations.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.
    """
    def __init__(self, predictor, **kwargs):
        Recommender.__init__(self, **kwargs)
        self.__predictor = predictor
        self.__train_set = None

    def get_name(self):
        return self.__predictor.get_name()

    def get_params(self):
        return self.__predictor.get_params()

    def train(self, train_set):
        self.__predictor.train(train_set)
        self.__train_set = train_set

    def recommend(self, user, num_items=10):
        items = self.__train_set['item'].unique()

        # filter items that are rated by the user already
        if self.rated_items_filter:
            is_user = self.__train_set['user'] == user
            user_item_ratings = self.__train_set.loc[is_user]['item'].tolist()
            items = [i for i in items if i not in user_item_ratings]

        # compute recommendations for all items and truncate to the top num_items
        item_ratings = list(map(lambda i: (i, self.__predictor.predict(user, i)), items))
        item_ratings.sort(key=lambda i: i[1], reverse=True)

        return pd.DataFrame(item_ratings[:num_items], columns=['item', 'score'])
