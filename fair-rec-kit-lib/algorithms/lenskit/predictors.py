""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..predictor import PredictorAlgorithm
from .algorithms import *


class PredictorLensKit(PredictorAlgorithm):

    def train(self, train_set):
        self._algo.fit(train_set)

    def predict(self, user, items):
        raise NotImplementedError


def create_predictor_pop_score(params):
    return PredictorLensKit(create_pop_score(params), params)


def create_predictor_bias(params):
    return PredictorLensKit(create_bias(params), params)


def create_predictor_funk_svd(params):
    return PredictorLensKit(create_funk_svd(params), params)


def create_predictor_knn_item_item(params):
    return PredictorLensKit(create_knn_item_item(params), params)


def create_predictor_knn_user_user(params):
    return PredictorLensKit(create_knn_user_user(params), params)
