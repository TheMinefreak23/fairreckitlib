"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import os
import time

import pandas as pd

from fairreckitlib.algorithms.common import FUNC_CREATE_ALGORITHM
from fairreckitlib.experiment.config import EXP_KEY_MODEL_NAME
from fairreckitlib.experiment.config import EXP_KEY_MODEL_PARAMS


class ModelPipeline(metaclass=ABCMeta):

    def __init__(self, api_name, factory):
        self.api_name = api_name
        self.dataset_name = None
        self.factory = factory
        self.tested_models = dict()

        self.train_set = None
        self.test_set = None

    def run(self, dataset_name, train_path, test_path, rating_scale, rating_type, output_dir, models, num_threads,
            callback, **kwargs):
        callback.on_begin_pipeline(self.api_name)

        self.dataset_name = dataset_name
        self.load_train_test_set(train_path, test_path, callback)
        model_dirs = self.run_batch(models, rating_scale, rating_type, output_dir, num_threads, callback, **kwargs)

        callback.on_end_pipeline(self.api_name)

        return model_dirs

    def create_model(self, model_name, params, rating_scale, rating_type, num_threads, callback):
        callback.on_create_model(model_name, params)

        model = self.factory[model_name][FUNC_CREATE_ALGORITHM](
            params,
            num_threads=num_threads,
            rating_scale=rating_scale,
            rating_type=rating_type
        )
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

    def load_train_test_set(self, train_path, test_path, callback):
        self.load_train_set(train_path, callback)
        self.load_test_set(test_path, callback)

    def run_batch(self, models, rating_scale, rating_type, output_dir, num_threads, callback, **kwargs):
        model_dirs = []
        for _, model in enumerate(models):
            model_dirs.append(self.run_model(
                model[EXP_KEY_MODEL_NAME],
                model[EXP_KEY_MODEL_PARAMS],
                rating_scale,
                rating_type,
                output_dir,
                num_threads,
                callback,
                **kwargs
            ))

        return model_dirs

    def run_model(self, model_name, model_params, rating_scale, rating_type, output_dir, num_threads, callback, **kwargs):
        model_dir, model = self.begin_model(model_name, model_params, rating_scale, rating_type, output_dir,
                                            num_threads, callback)
        self.train_test_model(model, model_dir, callback, **kwargs)
        self.end_model(model_name, callback)

        return model_dir

    def begin_model(self, model_name, model_params, rating_scale, rating_type, output_dir, num_threads, callback):
        if self.tested_models.get(model_name) is None:
            self.tested_models[model_name] = 0

        callback.on_begin_model(model_name)

        model_dir = self.create_model_output_dir(output_dir, model_name, callback)
        model = self.create_model(model_name, model_params, rating_scale, rating_type, num_threads, callback)

        return model_dir, model

    def train_model(self, model, callback):
        callback.on_begin_train_model(model, self.train_set)

        start = time.time()
        model.train(self.train_set)
        end = time.time()

        callback.on_end_train_model(model, self.train_set, end - start)

    def test_model(self, model, model_dir, callback, **kwargs):
        callback.on_begin_test_model(model, self.test_set)
        start = time.time()

        batch_size = 10000
        file_path = os.path.join(model_dir, 'ratings.tsv')

        self.test_model_ratings(
            model,
            file_path,
            batch_size,
            **kwargs
        )

        end = time.time()
        callback.on_end_test_model(model, self.test_set, end - start)

    @abstractmethod
    def test_model_ratings(self, model, output_path, batch_size, **kwargs):
        raise NotImplementedError()

    def train_test_model(self, model, model_dir, callback, **kwargs):
        self.train_model(model, callback)
        self.test_model(model, model_dir, callback, **kwargs)

    def end_model(self, model_name, callback):
        self.tested_models[model_name] += 1

        callback.on_end_model(model_name)
