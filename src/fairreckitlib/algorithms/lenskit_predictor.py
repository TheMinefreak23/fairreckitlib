"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from lenskit import batch

from .base_predictor import Predictor


class LensKitPredictor(Predictor):
    """Predictor implementation for LensKit.

    Args:
        predictor(lenskit.Predictor): the prediction algorithm.
        name(str): the name of the algorithm.
        params(dict): the parameters of the algorithm.

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
    """
    def __init__(self, predictor, name, params, **kwargs):
        Predictor.__init__(self, **kwargs)
        self.__predictor = predictor
        self.__name = name
        self.__params = params

    def get_name(self):
        return self.__name

    def get_params(self):
        return dict(self.__params)

    def train(self, train_set):
        self.__predictor.fit(train_set)

    def predict(self, user, item):
        prediction = self.__predictor.predict_for_user(user, [item])
        return prediction[item]

    def predict_batch(self, user_item_pairs):
        n_jobs = self.num_threads if self.num_threads > 0 else None
        predictions = batch.predict(self.__predictor, user_item_pairs, n_jobs=n_jobs)
        return predictions[['user', 'item', 'prediction']]
