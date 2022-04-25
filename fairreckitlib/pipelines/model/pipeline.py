"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import os
import time
from typing import Any

import json
import pandas as pd

from fairreckitlib.events import io_event
from fairreckitlib.events import model_event

MODEL_USER_BATCH_SIZE = 10000
RATING_OUTPUT_FILE = 'ratings.tsv'


@dataclass
class ModelConfig:
    """Model Configuration."""

    name: str
    params: {str: Any}


class ModelPipeline(metaclass=ABCMeta):
    """Model Pipeline to run computations for algorithms from a specific API.

    Args:
        api_name(str): name of the API associated with the algorithm factory.
        algo_factory(AlgorithmFactory): factory of available algorithms for this API.
        event_dispatcher(EventDispatcher): used to dispatch model/IO events
            when running the pipeline.
    """

    def __init__(self, api_name, algo_factory, event_dispatcher):
        self.api_name = api_name
        self.algo_factory = algo_factory
        self.tested_models = {}

        self.train_set = None
        self.test_set = None

        self.event_dispatcher = event_dispatcher

    def run(self, output_dir, data_transition, models_config, is_running, **kwargs):
        """Runs the entire pipeline from beginning to end.

        Effectively running all computations of the specified models.

        Args:
            output_dir(str): the path of the directory to store the output.
            data_transition(DataTransition): data input.
            models_config(array like): list of ModelConfig objects.
            is_running(func -> bool): function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.

        Returns:
            result_dirs(array like): list of model directories
                where computation results are stored.
        """
        self.event_dispatcher.dispatch(
            model_event.ON_BEGIN_MODEL_PIPELINE,
            api_name=self.api_name,
            num_models=len(models_config)
        )

        start = time.time()

        self.load_train_and_test_set(
            data_transition.train_set_path,
            data_transition.test_set_path
        )
        if not is_running():
            return None

        result_dirs = []

        for _, model in enumerate(models_config):
            result_dirs.append(self.run_model(
                output_dir,
                data_transition,
                model,
                is_running,
                **kwargs
            ))
            if not is_running():
                return None

        end = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_END_MODEL_PIPELINE,
            api_name=self.api_name,
            num_models=len(models_config),
            elapsed_time=end - start
        )

        return result_dirs

    def run_model(self, output_dir, data_transition, model_config, is_running, **kwargs):
        """Runs the computation for the specified model configuration.

        Args:
            output_dir(str): the path of the directory to store the output.
            data_transition(DataTransition): data input.
            model_config(ModelConfig): the algorithm model configuration.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.

        Returns:
            model_dir(str): the directory where computed ratings are stored.
        """
        model_dir, model, start = self.begin_model(
            model_config.name,
            model_config.params,
            output_dir,
            rating_scale=data_transition.rating_scale,
            rating_type=data_transition.rating_type,
            num_threads=kwargs['num_threads']
        )
        self.train_and_test_model(model, model_dir, is_running, **kwargs)
        self.end_model(model, start)

        return model_dir

    def begin_model(self, model_name, model_params, output_dir, **kwargs):
        """Prepares the model computation.

        Resolves the output directory to create for the model computation,
        so that it is unique and creates the model.

        Args:
            model_name(str): name of the model's algorithm.
            model_params(dict): parameters of the algorithm.
            output_dir(str): the path of the directory to store the output.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.

        Returns:
            model_dir(str): the path of the directory where the computed ratings can be stored.
            model(Algorithm): the created model according the specified name and parameters.
            start(float): the time when the model computation started.
        """
        if self.tested_models.get(model_name) is None:
            self.tested_models[model_name] = 0

        start = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_BEGIN_MODEL,
            model_name=model_name
        )

        model_dir = self.create_model_output_dir(
            output_dir,
            model_name
        )

        model = self.algo_factory.create(
            model_name,
            model_params,
            **kwargs
        )

        self.write_settings_file(model_dir, model.get_params())

        return model_dir, model, start

    def create_model_output_dir(self, output_dir, model_name):
        """Creates the output directory for a model.

        Args:
            output_dir(str): the path of the directory to store the output.
            model_name(str): name of the model's algorithm.

        Returns:
            model_dir(str): the path of the directory where the computed ratings can be stored.
        """
        index = self.tested_models[model_name]
        model_dir = os.path.join(output_dir, self.api_name + '_' + model_name + '_' + str(index))
        if not os.path.isdir(model_dir):
            self.event_dispatcher.dispatch(
                io_event.ON_MAKE_DIR,
                dir=model_dir
            )
            os.mkdir(model_dir)

        return model_dir

    def end_model(self, model, start):
        """Finalizes the model computation.

        Updates the number of tested models so that additional
        computations remain unique for this model.

        Args:
            model(Algorithm): the model that finished.
            start(float): the time when the model computation started.
        """
        self.tested_models[model.name] += 1

        end = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_END_MODEL,
            model=model,
            elapsed_time=end - start
        )

    def load_test_set(self, test_set_path):
        """Loads the test set that all models can use for testing.

        Args:
            test_set_path(str): path to where the test set is stored.
        """
        self.event_dispatcher.dispatch(
            model_event.ON_BEGIN_LOAD_TEST_SET,
            test_set_path=test_set_path
        )

        start = time.time()
        self.test_set = pd.read_csv(
            test_set_path,
            sep='\t',
            header=None,
            names=['user', 'item', 'rating']
        )
        end = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_END_LOAD_TEST_SET,
            test_set_path=test_set_path,
            test_set=self.test_set,
            elapsed_time=end - start
        )

    def load_train_set(self, train_set_path):
        """Loads the train set that all models can use for training.

        Args:
            train_set_path(str): path to where the train set is stored.
        """
        self.event_dispatcher.dispatch(
            model_event.ON_BEGIN_LOAD_TRAIN_SET,
            train_set_path=train_set_path
        )

        start = time.time()
        self.train_set = pd.read_csv(
            train_set_path,
            sep='\t',
            header=None,
            names=['user', 'item', 'rating']
        )
        end = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_END_LOAD_TRAIN_SET,
            train_set_path=train_set_path,
            train_set=self.train_set,
            elapsed_time=end - start
        )

    def load_train_and_test_set(self, train_set_path, test_set_path):
        """Loads the train and test set that all models can use.

        Args:
            train_set_path(str): path to where the train set is stored.
            test_set_path(str): path to where the test set is stored.
        """
        self.load_train_set(train_set_path)
        self.load_test_set(test_set_path)

    def write_settings_file(self, settings_dir, model_params):
        """Write model params in settings file in the model's result directory.

        Args:
            settings_dir(str): directory in which to store the model parameter settings.
            model_params(dict): dictionary containing the model parameter name-value pairs.
        """
        settings_path = os.path.join(settings_dir, 'settings.json')

        with open(settings_path, 'w', encoding='utf-8') as file:
            json.dump(model_params, file, indent=4)

        self.event_dispatcher.dispatch(
            model_event.ON_SAVE_MODEL_SETTINGS,
            settings_path=settings_path,
            model_params=model_params
        )

    def test_model(self, model, model_dir, is_running, **kwargs):
        """Tests the specified model using the test set.

        This function wraps the event dispatching and functionality
        that both predictor and recommender models have in common.

        Args:
            model(Algorithm): the model that needs to be tested.
            model_dir(str): the path of the directory where the computed ratings can be stored.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
        """
        self.event_dispatcher.dispatch(
            model_event.ON_BEGIN_TEST_MODEL,
            model=model,
            test_set=self.test_set
        )

        start = time.time()

        self.test_model_ratings(
            model,
            os.path.join(model_dir, RATING_OUTPUT_FILE),
            MODEL_USER_BATCH_SIZE,
            is_running,
            **kwargs
        )

        end = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_END_TEST_MODEL,
            model=model,
            test_set=self.test_set,
            elapsed_time=end - start
        )

    @abstractmethod
    def test_model_ratings(self, model, output_path, batch_size, is_running, **kwargs):
        """Tests the specified model for rating predictions or recommendations.

        Args:
            model(Algorithm): the model that needs to be tested.
            output_path(str): path to the file where the ratings will be stored.
            batch_size(int): number of users to test ratings for in a batch.
            is_running(func -> bool): function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
        """
        raise NotImplementedError()

    def train_model(self, model):
        """Trains the specified model using the train set.

        Args:
            model(Algorithm): the model that needs to be trained.
        """
        self.event_dispatcher.dispatch(
            model_event.ON_BEGIN_TRAIN_MODEL,
            model=model,
            train_set=self.train_set
        )

        start = time.time()
        model.train(self.train_set)
        end = time.time()

        self.event_dispatcher.dispatch(
            model_event.ON_END_TRAIN_MODEL,
            model=model,
            train_set=self.train_set,
            elapsed_time=end - start
        )

    def train_and_test_model(self, model, model_dir, is_running, **kwargs):
        """Trains and test the specified model.

        Args:
            model(Algorithm): the model that needs to be trained.
            model_dir(str): the path of the directory where the computed ratings can be stored.
            is_running(func -> bool): function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
        """
        self.train_model(model)
        if not is_running():
            return

        self.test_model(model, model_dir, is_running, **kwargs)
