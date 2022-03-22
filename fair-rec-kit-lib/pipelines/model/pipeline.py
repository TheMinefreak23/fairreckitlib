""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import time

import pandas as pd


class ModelPipeline(metaclass=ABCMeta):

    def __init__(self, api_name, factory):
        self._api_name = api_name
        self._factory = factory
        self.train_set = None
        self.test_set = None

    def run(self, train_set_path, test_set_path, models, callback, **kwargs):
        callback.on_begin_pipeline(self._api_name)

        self._load_train_set(train_set_path, callback)
        self._load_test_set(test_set_path, callback)

        self._run_batch(models, callback, **kwargs)

        callback.on_end_pipeline(self._api_name)

    def _create_model(self, model_name, params, callback):
        callback.on_create_model(model_name, params)
        return self._factory[model_name]['create'](params)

    def _load_train_set(self, train_set_path, callback):
        callback.on_begin_load_train_set(train_set_path)

        start = time.time()
        self.train_set = pd.read_csv(train_set_path, sep='\t', header=None, names=['user', 'item', 'rating'])
        end = time.time()

        callback.on_end_load_train_set(train_set_path, self.train_set, end - start)

    def _load_test_set(self, test_set_path, callback):
        callback.on_begin_load_test_set(test_set_path)

        start = time.time()
        self.test_set = pd.read_csv(test_set_path, sep='\t', header=None, names=['user', 'item', 'rating'])
        end = time.time()

        callback.on_end_load_test_set(test_set_path, self.test_set, end - start)

    def _run_batch(self, models, callback, **kwargs):
        for model_name in models:
            self._run_model(model_name, models[model_name], callback, **kwargs)

    @abstractmethod
    def _run_model_test(self, model, callback, **kwargs):
        raise NotImplementedError()

    def _run_model(self, model_name, params, callback, **kwargs):
        model = self._create_model(model_name, params, callback)
        self._train_model(model, callback)
        self._run_model_test(model, callback, **kwargs)

    def _train_model(self, model, callback):
        callback.on_begin_train_model(model, self.train_set)

        start = time.time()
        model.train(self.train_set)
        end = time.time()

        callback.on_end_train_model(model, self.train_set, end - start)

