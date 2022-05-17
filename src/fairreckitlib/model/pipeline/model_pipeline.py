"""This module contains base functionality of the complete model pipeline.

Classes:

    ModelPipeline: class that batches multiple model computations for a specific API.

Functions:

    write_computed_ratings: append computed ratings to a result file.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import json
import pandas as pd

from ...core.config_constants import MODEL_RATINGS_FILE, MODEL_USER_BATCH_SIZE
from ...core.factories import Factory
from ...core.event_dispatcher import EventDispatcher
from ...core.event_io import ON_MAKE_DIR
from ...core.event_error import ON_FAILURE_ERROR, ON_RAISE_ERROR
from ...data.data_transition import DataTransition
from ..algorithms.base_algorithm import BaseAlgorithm
from .model_config import ModelConfig
from .model_event import ON_BEGIN_LOAD_TEST_SET, ON_END_LOAD_TEST_SET
from .model_event import ON_BEGIN_LOAD_TRAIN_SET, ON_END_LOAD_TRAIN_SET
from .model_event import ON_BEGIN_MODEL_PIPELINE, ON_END_MODEL_PIPELINE
from .model_event import ON_BEGIN_TEST_MODEL, ON_END_TEST_MODEL
from .model_event import ON_BEGIN_TRAIN_MODEL, ON_END_TRAIN_MODEL
from .model_event import ON_BEGIN_MODEL, ON_END_MODEL
from .model_event import ON_SAVE_MODEL_SETTINGS


class ModelPipeline(metaclass=ABCMeta):
    """Model Pipeline to run computations for algorithms from a specific API.

    Wraps the common functionality that applies to all models disregarding the type.
    Loading the train and test is only done once each time the pipeline is run.
    After the previously mentioned sets are done loading, the pipeline loops
    through all specified models and executes the following steps:

    1) create the output directory.
    2) create the model.
    3) save the model's creation settings.
    4) train the model using the train set.
    5) test the model using the test set.
    6) add original ratings to the result.

    Abstract methods:

    get_ratings_dataframe
    test_model_ratings

    Public methods:

    run
    """

    def __init__(self, algo_factory: Factory, event_dispatcher: EventDispatcher):
        """Construct the model pipeline.

        Args:
            algo_factory: factory of available algorithms for this API.
            event_dispatcher: used to dispatch model/IO events when running the pipeline.
        """
        self.algo_factory = algo_factory
        self.tested_models = {}

        self.train_set = None
        self.test_set = None

        self.event_dispatcher = event_dispatcher

    def run(self,
            output_dir: str,
            data_transition: DataTransition,
            models_config: List[ModelConfig],
            is_running: Callable[[], bool],
            **kwargs) -> List[str]:
        """Run the entire pipeline from beginning to end.

        Effectively running all computations of the specified models.
        The pipeline raises a FileNotFoundError when it fails to load
        the train and/or test set.

        Args:
            output_dir: the path of the directory to store the output.
            data_transition: data input.
            models_config: list of ModelConfig objects.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.

        Returns:
            a list of model directories where computation results are stored.
        """
        result_dirs = []
        if not is_running():
            return result_dirs

        self.event_dispatcher.dispatch(
            ON_BEGIN_MODEL_PIPELINE,
            api_name=self.algo_factory.get_name(),
            num_models=len(models_config)
        )

        start = time.time()

        # this can raise a FileNotFoundError, effectively aborting the pipeline
        self.load_train_and_test_set(
            data_transition.train_set_path,
            data_transition.test_set_path
        )
        if not is_running():
            return result_dirs

        for _, model in enumerate(models_config):
            if not self.algo_factory.is_obj_available(model.name):
                self.event_dispatcher.dispatch(
                    ON_FAILURE_ERROR,
                    msg='Failure: algorithm is not available: ' +
                        self.algo_factory.get_name() + ' ' + model.name
                )
                continue

            try:
                model_dir = self.run_model(
                    output_dir,
                    data_transition,
                    model,
                    is_running,
                    **kwargs
                )
            except ArithmeticError:
                self.event_dispatcher.dispatch(
                    ON_RAISE_ERROR,
                    msg='ArithmeticError: trying to run model ' +
                        self.algo_factory.get_name() + ' ' + model.name
                )
                continue
            except MemoryError:
                self.event_dispatcher.dispatch(
                    ON_RAISE_ERROR,
                    msg='MemoryError: trying to run model ' +
                        self.algo_factory.get_name() + ' ' + model.name
                )
                continue
            except RuntimeError:
                self.event_dispatcher.dispatch(
                    ON_RAISE_ERROR,
                    msg='RuntimeError: trying to run model ' +
                        self.algo_factory.get_name() + ' ' + model.name
                )
                continue

            result_dirs.append(model_dir)
            if not is_running():
                return result_dirs

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_MODEL_PIPELINE,
            api_name=self.algo_factory.get_name(),
            num_models=len(models_config),
            elapsed_time=end - start
        )

        return result_dirs

    def run_model(self,
                  output_dir: str,
                  data_transition: DataTransition,
                  model_config: ModelConfig,
                  is_running: Callable[[], bool],
                  **kwargs) -> str:
        """Run the model computation for the specified model configuration.

        Several possible errors can be raised during the model computation run:
        ArithmeticError, MemoryError and RuntimeError.

        Args:
            output_dir: the path of the directory to store the output.
            data_transition: data input.
            model_config: the algorithm model configuration.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.

        Returns:
            the directory where the model's computed ratings are stored.
        """
        model_dir, model, start = self.begin_model(
            model_config.name,
            model_config.params,
            output_dir,
            rating_scale=data_transition.rating_scale,
            #rating_type=data_transition.rating_type,
            rated_items_filter=kwargs.get('rated_items_filter'),
            num_threads=kwargs['num_threads']
        )
        if not is_running():
            return model_dir

        result_file_path = self.train_and_test_model(model, model_dir, is_running, **kwargs)
        if not is_running():
            return model_dir

        self.reconstruct_ratings(result_file_path)
        if not is_running():
            return model_dir

        self.end_model(model, start)

        return model_dir

    def begin_model(self,
                    model_name: str,
                    model_params: Dict[str, Any],
                    output_dir: str,
                    **kwargs) -> Tuple[str, BaseAlgorithm, float]:
        """Prepare the model computation.

        Resolves the output directory to create for the model computation,
        so that it is unique and creates the model. It raises a RuntimeError
        when the model's algorithm fails to create.

        Args:
            model_name: name of the model's algorithm.
            model_params: parameters of the algorithm.
            output_dir: the path of the directory to store the output.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.

        Returns:
            model_dir: the path of the directory where the computed ratings can be stored.
            model: the created model according the specified name and parameters.
            start: the time when the model computation started.
        """
        if self.tested_models.get(model_name) is None:
            self.tested_models[model_name] = 0

        start = time.time()

        self.event_dispatcher.dispatch(
            ON_BEGIN_MODEL,
            model_name=model_name
        )

        model_dir = self.create_model_output_dir(
            output_dir,
            model_name
        )

        try:
            model = self.algo_factory.create(
                model_name,
                model_params,
                **kwargs
            )
        except RuntimeError as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='RuntimeError: trying to create model ' +
                    self.algo_factory.get_name() + ' ' + model_name + ' => ' + str(err)
            )
            raise err

        self.write_settings_file(model_dir, model.get_params())

        return model_dir, model, start

    def create_model_output_dir(self, output_dir: str, model_name: str) -> str:
        """Create the output directory for a model.

        Args:
            output_dir: the path of the directory to store the output.
            model_name: name of the model's algorithm.

        Returns:
            the path of the directory where the model's computed ratings can be stored.
        """
        index = self.tested_models[model_name]
        model_dir = os.path.join(output_dir, self.algo_factory.get_name() +
                                 '_' + model_name + '_' + str(index))
        if not os.path.isdir(model_dir):
            self.event_dispatcher.dispatch(
                ON_MAKE_DIR,
                dir=model_dir
            )
            os.mkdir(model_dir)

        return model_dir

    def end_model(self, model: BaseAlgorithm, start: float) -> None:
        """Finalize the model computation.

        Updates the number of tested models so that additional
        computations remain unique for this model.

        Args:
            model: the model that finished.
            start: the time when the model computation started.
        """
        self.tested_models[model.get_name()] += 1

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_MODEL,
            model=model,
            elapsed_time=end - start
        )

    @abstractmethod
    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Get the dataframe that contains the original ratings.

        Returns:
            dataframe containing the 'user', 'item', 'rating', columns.
        """
        raise NotImplementedError()

    def load_test_set(self, test_set_path: str) -> None:
        """Load the test set that all models can use for testing.

        Args:
            test_set_path: path to where the test set is stored.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_LOAD_TEST_SET,
            test_set_path=test_set_path
        )

        start = time.time()

        try:
            self.test_set = pd.read_csv(
                test_set_path,
                sep='\t',
                header=None,
                names=['user', 'item', 'rating']
            )
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='FileNotFoundError: raised while trying to load the test set from ' +
                    test_set_path
            )
            # raise again so that the pipeline aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_TEST_SET,
            test_set_path=test_set_path,
            test_set=self.test_set,
            elapsed_time=end - start
        )

    def load_train_set(self, train_set_path: str) -> None:
        """Load the train set that all models can use for training.

        Args:
            train_set_path: path to where the train set is stored.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_LOAD_TRAIN_SET,
            train_set_path=train_set_path
        )

        start = time.time()

        try:
            self.train_set = pd.read_csv(
                train_set_path,
                sep='\t',
                header=None,
                names=['user', 'item', 'rating']
            )
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='FileNotFoundError: raised while trying to load the train set from ' +
                    train_set_path
            )
            # raise again so that the pipeline aborts
            raise err

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_LOAD_TRAIN_SET,
            train_set_path=train_set_path,
            train_set=self.train_set,
            elapsed_time=end - start
        )

    def load_train_and_test_set(self, train_set_path: str, test_set_path: str) -> None:
        """Load the train and test set that all models can use.

        Args:
            train_set_path: path to where the train set is stored.
            test_set_path: path to where the test set is stored.
        """
        self.load_train_set(train_set_path)
        self.load_test_set(test_set_path)

    def reconstruct_ratings(self, result_file_path: str) -> None:
        """Reconstruct the original ratings in the specified result file.

        Args:
            result_file_path: path to the file that needs the ratings to be added.
        """
        result = pd.read_csv(result_file_path, sep='\t')
        result = pd.merge(result, self.get_ratings_dataframe(), how='left', on=['user', 'item'])
        result.to_csv(result_file_path, sep='\t', header=True, index=False)

    def write_settings_file(self, settings_dir: str, model_params: Dict[str, Any]) -> None:
        """Write model params in settings file in the model's result directory.

        Args:
            settings_dir: directory in which to store the model parameter settings.
            model_params: dictionary containing the model parameter name-value pairs.
        """
        settings_path = os.path.join(settings_dir, 'settings.json')

        with open(settings_path, 'w', encoding='utf-8') as file:
            json.dump(model_params, file, indent=4)

        self.event_dispatcher.dispatch(
            ON_SAVE_MODEL_SETTINGS,
            settings_path=settings_path,
            model_params=model_params
        )

    def test_model(self,
                   model: BaseAlgorithm,
                   model_dir: str,
                   is_running: Callable[[], bool],
                   **kwargs) -> str:
        """Test the specified model using the test set.

        This function wraps the event dispatching and functionality
        that both predictor and recommender models have in common.

        Args:
            model: the model that needs to be tested.
            model_dir: the path of the directory where the computed ratings can be stored.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.

        Returns:
            the path to the file where the model's computed ratings are stored.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_TEST_MODEL,
            model=model,
            test_set=self.test_set
        )

        start = time.time()

        result_file_path = os.path.join(model_dir, MODEL_RATINGS_FILE)
        self.test_model_ratings(
            model,
            result_file_path,
            MODEL_USER_BATCH_SIZE,
            is_running,
            **kwargs
        )

        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_TEST_MODEL,
            model=model,
            test_set=self.test_set,
            elapsed_time=end - start
        )

        return result_file_path

    @abstractmethod
    def test_model_ratings(self,
                           model: BaseAlgorithm,
                           output_path: str,
                           batch_size: int,
                           is_running: Callable[[], bool],
                           **kwargs) -> None:
        """Test the specified model for rating predictions or recommendations.

        Args:
            model: the model that needs to be tested.
            output_path: path to the file where the ratings will be stored.
            batch_size: number of users to test ratings for in a batch.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
        """
        raise NotImplementedError()

    def train_model(self, model: BaseAlgorithm) -> None:
        """Train the specified model using the train set.

        Args:
            model: the model that needs to be trained.
        """
        self.event_dispatcher.dispatch(
            ON_BEGIN_TRAIN_MODEL,
            model=model,
            train_set=self.train_set
        )

        start = time.time()
        model.train(self.train_set)
        end = time.time()

        self.event_dispatcher.dispatch(
            ON_END_TRAIN_MODEL,
            model=model,
            train_set=self.train_set,
            elapsed_time=end - start
        )

    def train_and_test_model(self,
                             model: BaseAlgorithm,
                             model_dir: str,
                             is_running: Callable[[], bool],
                             **kwargs) -> str:
        """Train and test the specified model.

        Several possible errors can be raised during the executing of both training and
        testing the model: namely ArithmeticError, MemoryError and RuntimeError.

        Args:
            model: the model that needs to be trained.
            model_dir: the path of the directory where the computed ratings can be stored.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.

        Returns:
            the path to the file where the model's computed ratings are stored.
        """
        try:
            self.train_model(model)
        except (ArithmeticError, MemoryError, RuntimeError) as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='Error: raised while training model ' +
                    self.algo_factory.get_name() + ' ' + model.get_name()
            )
            # raise again so the model run aborts
            raise err

        try:
            result_file_path = self.test_model(model, model_dir, is_running, **kwargs)
        except (ArithmeticError, MemoryError, RuntimeError) as err:
            self.event_dispatcher.dispatch(
                ON_RAISE_ERROR,
                msg='Error: raised while testing model ' +
                    self.algo_factory.get_name() + ' ' + model.get_name()
            )
            # raise again so the model run aborts
            raise err

        return result_file_path


def write_computed_ratings(output_path: str, ratings: pd.DataFrame, header: bool) -> None:
    """Append the ratings to the file specified by the output path.

    Args:
        output_path: path to the file where the ratings are appended to.
        ratings: the computed ratings dataframe that needs to be appended.
        header: whether to write the header as the first line.

    """
    ratings.to_csv(output_path, mode='a', sep='\t', header=header, index=False)
