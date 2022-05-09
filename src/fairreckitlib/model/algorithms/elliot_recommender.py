"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .base_recommender import Recommender


class ElliotRecommender(Recommender):
    """Recommender implementation for Elliot.

    Args:
        name(str): the name of the algorithm.
        params(dict): the parameters of the algorithm.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.
    """
    def __init__(self, name, params, **kwargs):
        Recommender.__init__(self, **kwargs)
        self.__name = name
        self.__params = params
        if not self.rated_items_filter:
            raise RuntimeError('Elliot: not supported.')

    def get_name(self):
        return self.__name

    def get_params(self):
        return dict(self.__params)

    def train(self, train_set):
        # not used, training is done by running the framework
        raise NotImplementedError()

    def recommend(self, user, num_items=10):
        # not used, recommending is done by running the framework
        raise NotImplementedError()
