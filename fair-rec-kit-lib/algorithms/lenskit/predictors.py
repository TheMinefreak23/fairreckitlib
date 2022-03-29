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
