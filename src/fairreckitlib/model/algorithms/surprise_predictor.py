"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from surprise.dataset import Dataset
from surprise.reader import Reader

from .base_predictor import Predictor


class SurprisePredictor(Predictor):
    """Predictor implementation for Surprise.

    Args:
        algo(surprise.AlgoBase): the prediction algorithm.
        name(str): the name of the algorithm.
        params(dict): the parameters of the algorithm.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rating_scale(tuple): consisting of (min_rating, max_rating) on
            which the algorithm will perform training.
    """
    def __init__(self, algo, name, params, **kwargs):
        Predictor.__init__(self, **kwargs)

        self.__algo = algo
        self.__name = name
        self.__params = params

        self.rating_scale = kwargs['rating_scale']

    def get_name(self):
        return self.__name

    def get_params(self):
        return dict(self.__params)

    def train(self, train_set):
        reader = Reader(rating_scale=self.rating_scale)
        dataset = Dataset.load_from_df(train_set, reader)
        self.__algo.fit(dataset.build_full_trainset())

    def predict(self, user, item):
        prediction = self.__algo.predict(user, item, clip=False)
        return prediction.est
