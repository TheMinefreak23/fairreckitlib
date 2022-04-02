"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import os
import time

import pandas as pd

from algorithms.common import FUNC_CREATE_ALGORITHM
from experiment.config import EXP_KEY_MODEL_NAME
from experiment.config import EXP_KEY_MODEL_PARAMS


class ModelPipeline(metaclass=ABCMeta):

    def __init__(self, api_name, factory):
        self.api_name = api_name
        self.dataset_name = None
        self.factory = factory
        self.tested_models = dict()

        self.train_set = None
        self.test_set = None

    def run(self, dataset_name, train_set_path, test_set_path, output_dir, models, callback, **kwargs):
        callback.on_begin_pipeline(self.api_name)

        self.dataset_name = dataset_name
        self.load_train_test_set(train_set_path, test_set_path, callback)
        model_dirs = self.run_batch(models, output_dir, callback, **kwargs)

        callback.on_end_pipeline(self.api_name)

        return model_dirs

    def create_model(self, model_name, params, callback):
        callback.on_create_model(model_name, params)

        model = self.factory[model_name][FUNC_CREATE_ALGORITHM](params)
        model.name = model_name

        return model

    def create_model_output_dir(self, output_dir, model_name, callback):
        index = self.tested_models[model_name]
        model_dir = os.path.join(output_dir, self.api_name + '_' + model_name + '_' + str(index))
        if not os.path.isdir(model_dir):
            callback.on_create_folder(model_dir)
            os.mkdir(model_dir)

        return model_dir

    def load_train_set(self, train_set_path, callback):
        callback.on_begin_load_train_set(train_set_path)

        start = time.time()
        self.train_set = pd.read_csv(train_set_path, sep='\t', header=None, names=['user', 'item', 'rating'])
        end = time.time()

        callback.on_end_load_train_set(train_set_path, self.train_set, end - start)

    def load_test_set(self, test_set_path, callback):
        callback.on_begin_load_test_set(test_set_path)

        start = time.time()
        self.test_set = pd.read_csv(test_set_path, sep='\t', header=None, names=['user', 'item', 'rating'])
        end = time.time()

        callback.on_end_load_test_set(test_set_path, self.test_set, end - start)

    def load_train_test_set(self, train_set_path, test_set_path, callback):
        self.load_train_set(train_set_path, callback)
        self.load_test_set(test_set_path, callback)

    def run_batch(self, models, output_dir, callback, **kwargs):
        model_dirs = []
        for _, model in enumerate(models):
            model_dirs.append(self.run_model(
                model[EXP_KEY_MODEL_NAME],
                model[EXP_KEY_MODEL_PARAMS],
                output_dir,
                callback,
                **kwargs
            ))

        return model_dirs

    def run_model(self, model_name, model_params, output_dir, callback, **kwargs):
        model_dir, model = self.begin_model(model_name, model_params, output_dir, callback)
        self.train_test_model(model, model_dir, callback, **kwargs)
        self.end_model(model_name, callback)

        return model_dir

    def begin_model(self, model_name, model_params, output_dir, callback):
        if self.tested_models.get(model_name) is None:
            self.tested_models[model_name] = 0

        callback.on_begin_model(model_name)

        model_dir = self.create_model_output_dir(output_dir, model_name, callback)
        model = self.create_model(model_name, model_params, callback)

        return model_dir, model

    def train_model(self, model, callback):
        callback.on_begin_train_model(model, self.train_set)

        start = time.time()
        model.train(self.train_set)
        end = time.time()

        callback.on_end_train_model(model, self.train_set, end - start)

    @abstractmethod
    def test_model(self, model, model_dir, callback, **kwargs):
        raise NotImplementedError()

    def train_test_model(self, model, model_dir, callback, **kwargs):
        self.train_model(model, callback)
        self.test_model(model, model_dir, callback, **kwargs)

    def end_model(self, model_name, callback):
        self.tested_models[model_name] += 1

        callback.on_end_model(model_name)

    @staticmethod
    def stringify_model(model):
        result = model.name
        params = model.get_params()
        for param_name in params:
            result = result + "_" + param_name + "=" + str(params[param_name])

        return result
