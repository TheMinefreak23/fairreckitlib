"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import lenskit.batch as batch

from ..predictor import PredictorAlgorithm
from .algorithms import *


class PredictorLensKit(PredictorAlgorithm):

    def train(self, train_set):
        self.algo.fit(train_set)

    def predict(self, user, item):
        prediction = self.algo.predict_for_user(user, [item])
        return prediction[item]

    def predict_batch(self, user_item_pairs):
        n_jobs = self.num_threads if self.num_threads > 0 else None
        predictions = batch.predict(self.algo, user_item_pairs, n_jobs=n_jobs)
        return predictions[['user', 'item', 'prediction']]


def create_predictor_biased_mf(params, **kwargs):
    return PredictorLensKit(create_biased_mf(params), params, **kwargs)


def create_predictor_implicit_mf(params, **kwargs):
    return PredictorLensKit(create_implicit_mf(params), params, **kwargs)


def create_predictor_item_item(params, **kwargs):
    return PredictorLensKit(create_item_item(params, kwargs['rating_type']), params, **kwargs)


def create_predictor_pop_score(params, **kwargs):
    return PredictorLensKit(create_pop_score(params), params, **kwargs)


def create_predictor_user_user(params, **kwargs):
    return PredictorLensKit(create_user_user(params, kwargs['rating_type']), params, **kwargs)
