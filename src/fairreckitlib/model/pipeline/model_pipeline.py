"""This module contains base functionality of the complete model pipeline.

Classes:

    ModelPipeline: class that batches multiple model computations for a specific API.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from abc import ABCMeta, abstractmethod
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from ...core.config.config_factories import Factory
from ...core.core_constants import MODEL_RATINGS_FILE, MODEL_USER_BATCH_SIZE
from ...core.events.event_dispatcher import EventDispatcher
from ...core.events.event_error import ON_FAILURE_ERROR, ON_RAISE_ERROR, ErrorEventArgs
from ...core.io.event_io import DataframeEventArgs, FileEventArgs
from ...core.io.io_create import create_dir, create_json
from ...core.io.io_delete import delete_dir
from ...core.pipeline.core_pipeline import CorePipeline
from ...data.data_transition import DataTransition
from ..algorithms.base_algorithm import BaseAlgorithm
from ..algorithms.matrix import Matrix
from .model_config import ModelConfig
from .model_event import ON_BEGIN_LOAD_TEST_SET, ON_END_LOAD_TEST_SET
from .model_event import ON_BEGIN_LOAD_TRAIN_SET, ON_END_LOAD_TRAIN_SET
from .model_event import ON_BEGIN_MODEL_PIPELINE, ON_END_MODEL_PIPELINE
from .model_event import ON_BEGIN_RECONSTRUCT_RATINGS, ON_END_RECONSTRUCT_RATINGS
from .model_event import ON_BEGIN_TEST_MODEL, ON_END_TEST_MODEL
from .model_event import ON_BEGIN_TRAIN_MODEL, ON_END_TRAIN_MODEL
from .model_event import ON_BEGIN_MODEL, ON_END_MODEL
from .model_event import ModelPipelineEventArgs, ModelEventArgs


class ModelPipeline(CorePipeline, metaclass=ABCMeta):
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

    After all models are trained and tested the computed rating files are updated
    with the original ratings from the train and test set.

    Abstract methods:

    load_test_set_users
    test_model_ratings

    Public methods:

    run
    """

    def __init__(
            self,
            algo_factory: Factory,
            data_transition: DataTransition,
            event_dispatcher: EventDispatcher):
        """Construct the model pipeline.

        Args:
            algo_factory: factory of available algorithms for this API.
            data_transition: data input.
            event_dispatcher: used to dispatch model/IO events when running the pipeline.
        """
        CorePipeline.__init__(self, event_dispatcher)
        self.algo_factory = algo_factory
        self.data_transition = data_transition
        self.tested_models = {}

        self.train_set_matrix = None
        self.test_set_users = None

    def run(self,
            output_dir: str,
            models_config: List[ModelConfig],
            is_running: Callable[[], bool],
            **kwargs) -> List[str]:
        """Run the entire pipeline from beginning to end.

        Effectively running all computations of the specified models.

        Args:
            output_dir: the path of the directory to store the output.
            models_config: list of ModelConfig objects to compute.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.

        Raises:
            FileNotFoundError: when either the train and/or test fails to load.

        Returns:
            a list of model directories where computation results are stored.
        """
        result_dirs = []
        if not is_running():
            return result_dirs

        self.event_dispatcher.dispatch(ModelPipelineEventArgs(
            ON_BEGIN_MODEL_PIPELINE,
            self.algo_factory.get_name(),
            models_config
        ))

        start = time.time()

        # this can raise a FileNotFoundError, effectively aborting the pipeline
        self.load_train_set_matrix()
        if not is_running():
            return result_dirs

        # this can raise a FileNotFoundError, effectively aborting the pipeline
        self.load_test_set_users()
        if not is_running():
            return result_dirs

        for model in models_config:
            # verify that the specified model is available
            if not self.algo_factory.is_obj_available(model.name):
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_FAILURE_ERROR,
                    'Failure: algorithm is not available: ' +
                    self.algo_factory.get_name() + ' ' + model.name
                ))
                continue

            # create model output dir
            model_dir = self.create_model_output_dir(
                output_dir,
                model.name
            )

            # attempt to run the model computation
            try:
                self.run_model(
                    model_dir,
                    model,
                    is_running,
                    **kwargs
                )
            except ArithmeticError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'ArithmeticError: trying to run model ' +
                    self.algo_factory.get_name() + ' ' + model.name
                ))
                delete_dir(model_dir, self.event_dispatcher)
                continue
            except MemoryError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'MemoryError: trying to run model ' +
                    self.algo_factory.get_name() + ' ' + model.name
                ))
                delete_dir(model_dir, self.event_dispatcher)
                continue
            except RuntimeError:
                self.event_dispatcher.dispatch(ErrorEventArgs(
                    ON_RAISE_ERROR,
                    'RuntimeError: trying to run model ' +
                    self.algo_factory.get_name() + ' ' + model.name
                ))
                delete_dir(model_dir, self.event_dispatcher)
                continue

            result_dirs.append(model_dir)
            if not is_running():
                return result_dirs

        # free up some memory because everything is trained and tested
        self.train_set_matrix = None
        self.test_set_users = None

        self.reconstruct_ratings(result_dirs, is_running)

        end = time.time()

        self.event_dispatcher.dispatch(ModelPipelineEventArgs(
            ON_END_MODEL_PIPELINE,
            self.algo_factory.get_name(),
            models_config
        ), elapsed_time=end - start)

        return result_dirs

    def run_model(
            self,
            model_dir: str,
            model_config: ModelConfig,
            is_running: Callable[[], bool],
            **kwargs) -> None:
        """Run the model computation for the specified model configuration.

        Args:
            model_dir: the path of the directory where the computed ratings can be stored.
            model_config: the algorithm model configuration.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.

        Raises:
            ArithmeticError: possibly raised by a model on construction, training or testing.
            MemoryError: possibly raised by a model on construction, training or testing.
            RuntimeError: possibly raised by a model on construction, training or testing.
        """
        model, start = self.begin_model(
            model_config.name,
            model_config.params,
            model_dir,
            **kwargs
        )
        if not is_running():
            return

        self.train_and_test_model(model, model_dir, is_running, **kwargs)
        if not is_running():
            return

        self.end_model(model, start)

    def begin_model(
            self,
            model_name: str,
            model_params: Dict[str, Any],
            model_dir: str,
            **kwargs) -> Tuple[BaseAlgorithm, float]:
        """Prepare the model computation.

        Resolves the output directory to create for the model computation,
        so that it is unique and creates the model.

        Args:
            model_name: name of the model's algorithm.
            model_params: parameters of the algorithm.
            model_dir: the path of the directory where the computed ratings can be stored.

        Keyword Args:
            num_threads(int): the max number of threads an algorithm can use.
            rated_items_filter(bool): whether to filter already rated items when
                producing item recommendations.

        Raises:
            ArithmeticError: possibly raised by a model on construction.
            MemoryError: possibly raised by a model on construction.
            RuntimeError: possibly raised by a model on construction.

        Returns:
            model: the created model according the specified name and parameters.
            start: the time when the model computation started.
        """
        start = time.time()

        self.event_dispatcher.dispatch(ModelEventArgs(
            ON_BEGIN_MODEL,
            model_name,
            model_params
        ))

        # attempt to create model
        kwargs['rating_type'] = self.data_transition.get_rating_type()
        model = self.algo_factory.create(
            model_name,
            model_params,
            **kwargs
        )

        # create settings file
        create_json(
            os.path.join(model_dir, 'settings.json'),
            model.get_params(),
            self.event_dispatcher,
            indent=4
        )

        return model, start

    def create_model_output_dir(self, output_dir: str, model_name: str) -> str:
        """Create the output directory for a model.

        Args:
            output_dir: the path of the directory to store the output.
            model_name: name of the model's algorithm.

        Returns:
            the path of the directory where the model's computed ratings can be stored.
        """
        if self.tested_models.get(model_name) is None:
            # initialize model name counter
            self.tested_models[model_name] = 0

        return create_dir(self.get_model_output_dir(output_dir, model_name), self.event_dispatcher)

    def get_model_output_dir(self, output_dir: str, model_name: str) -> str:
        """Get the model output directory path for the specified model name.

        Args:
            output_dir: the path of the directory to store the output.
            model_name: name of the model's algorithm.

        Returns:
            the path of the directory where the model's computed ratings can be stored.
        """
        index = self.tested_models[model_name]
        return os.path.join(
            output_dir,
            self.algo_factory.get_name() + '_' + model_name + '_' + str(index)
        )

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

        self.event_dispatcher.dispatch(ModelEventArgs(
            ON_END_MODEL,
            model.get_name(),
            model.get_params()
        ), elapsed_time=end - start)

    def on_load_train_set_matrix(self) -> Matrix:
        """Load the train set matrix that all models can use for training.

        The default train set matrix of the model pipeline is a dataframe.
        Derived classes are allowed to override this function to return a different type of matrix.

        Returns:
            the loaded train set matrix dataframe.
        """
        return Matrix(self.data_transition.train_set_path)

    def load_train_set_matrix(self) -> None:
        """Load the train set matrix that all models can use for training.

        Raises:
            FileNotFoundError: when the train set file is not found.
        """
        self.event_dispatcher.dispatch(DataframeEventArgs(
            ON_BEGIN_LOAD_TRAIN_SET,
            self.data_transition.train_set_path,
            'model train set matrix'
        ))

        start = time.time()

        try:
            self.train_set_matrix = self.on_load_train_set_matrix()
        except FileNotFoundError as err:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_RAISE_ERROR,
                'FileNotFoundError: raised while trying to load the matrix train set from ' +
                self.data_transition.train_set_path
            ))
            raise err


        end = time.time()

        self.event_dispatcher.dispatch(DataframeEventArgs(
            ON_END_LOAD_TRAIN_SET,
            self.data_transition.train_set_path,
            'model train set matrix'
        ), elapsed_time=end - start)

    def load_train_set_dataframe(self) -> pd.DataFrame:
        """Load the train set as a dataframe.

        Raises:
            FileNotFoundError: when the train set file is not found.

        Returns:
            the loaded train set dataframe.
        """
        return self.read_dataframe(
            self.data_transition.train_set_path,
            'data train set',
            ON_BEGIN_LOAD_TRAIN_SET,
            ON_END_LOAD_TRAIN_SET,
            names=['user', 'item', 'rating']
        )

    def load_test_set_dataframe(self, test_name: str='data test set') -> pd.DataFrame:
        """Load the test set as a dataframe.

        Args:
            test_name: name of the test set dataframe to dispatch in the dataframe event.

        Raises:
            FileNotFoundError: when the test set file is not found.

        Returns:
            the loaded test set dataframe.
        """
        return self.read_dataframe(
            self.data_transition.test_set_path,
            test_name,
            ON_BEGIN_LOAD_TEST_SET,
            ON_END_LOAD_TEST_SET,
            names=['user', 'item', 'rating']
        )

    @abstractmethod
    def load_test_set_users(self) -> None:
        """Load the test set users that all models can use for testing.

        Raises:
            FileNotFoundError: when the test set file is not found.
        """
        raise NotImplementedError()

    def reconstruct_ratings(
            self,
            result_dirs: List[str],
            is_running: Callable[[], bool]) -> None:
        """Reconstruct the original ratings for all the computed models ratings.

        Args:
            result_dirs: a list of directories that contain a computed rating file.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.
        """
        if not is_running() or len(result_dirs) == 0:
            return

        # TODO should probably move this code to a separate pipeline
        ratings_dataframe = pd.concat([
            self.load_train_set_dataframe(),
            self.load_test_set_dataframe()
        ])

        for model_dir in result_dirs:
            if not is_running():
                return

            result_file_path = os.path.join(model_dir, MODEL_RATINGS_FILE)

            self.event_dispatcher.dispatch(FileEventArgs(
                ON_BEGIN_RECONSTRUCT_RATINGS,
                result_file_path
            ))

            start = time.time()

            result = pd.read_csv(result_file_path, sep='\t')
            result = pd.merge(result, ratings_dataframe, how='left', on=['user', 'item'])
            result.to_csv(result_file_path, sep='\t', header=True, index=False)

            end = time.time()

            self.event_dispatcher.dispatch(FileEventArgs(
                ON_END_RECONSTRUCT_RATINGS,
                result_file_path
            ), elapsed_time=end - start)

    def test_model(
            self,
            model: BaseAlgorithm,
            model_dir: str,
            is_running: Callable[[], bool],
            **kwargs) -> None:
        """Test the specified model using the test set.

        This function wraps the event dispatching and functionality
        that both predictor and recommender models have in common.

        Args:
            model: the model that needs to be tested.
            model_dir: the path of the directory where the computed ratings can be stored.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.

        Raises:
            ArithmeticError: possibly raised by a model on testing.
            MemoryError: possibly raised by a model on testing.
            RuntimeError: possibly raised by a model on testing.
        """
        self.event_dispatcher.dispatch(ModelEventArgs(
            ON_BEGIN_TEST_MODEL,
            model.get_name(),
            model.get_params()
        ))

        start = time.time()

        result_file_path = os.path.join(model_dir, MODEL_RATINGS_FILE)
        start_index = 0
        while start_index < len(self.test_set_users):
            if not is_running():
                return

            user_batch = self.test_set_users[start_index : start_index + MODEL_USER_BATCH_SIZE]
            ratings = self.test_model_ratings(model, user_batch, **kwargs)
            if not is_running():
                return

            self.write_dataframe(result_file_path, ratings, start_index == 0)
            start_index += MODEL_USER_BATCH_SIZE

        end = time.time()

        self.event_dispatcher.dispatch(ModelEventArgs(
            ON_END_TEST_MODEL,
            model.get_name(),
            model.get_params()
        ), elapsed_time=end - start)

    @abstractmethod
    def test_model_ratings(
            self,
            model: BaseAlgorithm,
            user_batch: List[int],
            **kwargs) -> pd.DataFrame:
        """Test the specified model for rating predictions or recommendations.

        Args:
            model: the model that needs to be tested.
            user_batch: the user batch to compute model ratings for.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.

        Raises:
            ArithmeticError: possibly raised by a model on testing.
            MemoryError: possibly raised by a model on testing.
            RuntimeError: possibly raised by a model on testing.

        Returns:
            a dataframe containing the computed model ratings.
        """
        raise NotImplementedError()

    def train_model(self, model: BaseAlgorithm) -> None:
        """Train the specified model using the train set.

        Args:
            model: the model that needs to be trained.

        Raises:
            ArithmeticError: possibly raised by a model on training.
            MemoryError: possibly raised by a model on training.
            RuntimeError: possibly raised by a model on training.
        """
        self.event_dispatcher.dispatch(ModelEventArgs(
            ON_BEGIN_TRAIN_MODEL,
            model.get_name(),
            model.get_params()
        ))

        start = time.time()

        model.train(self.train_set_matrix)

        end = time.time()

        self.event_dispatcher.dispatch(ModelEventArgs(
            ON_END_TRAIN_MODEL,
            model.get_name(),
            model.get_params()
        ), elapsed_time=end - start)

    def train_and_test_model(
            self,
            model: BaseAlgorithm,
            model_dir: str,
            is_running: Callable[[], bool],
            **kwargs) -> None:
        """Train and test the specified model.

        Several possible errors can be raised during the executing of both training and
        testing the model: namely ArithmeticError, MemoryError and RuntimeError.

        Args:
            model: the model that needs to be trained.
            model_dir: the path of the directory where the computed ratings can be stored.
            is_running: function that returns whether the pipeline
                is still running. Stops early when False is returned.

        Raises:
            ArithmeticError: possibly raised by a model on training or testing.
            MemoryError: possibly raised by a model on training or testing.
            RuntimeError: possibly raised by a model on training or testing.

        Keyword Args:
            num_items(int): the number of item recommendations to produce, only
                needed when running the pipeline for recommender algorithms.
        """
        try:
            self.train_model(model)
        except (ArithmeticError, MemoryError, RuntimeError) as err:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_RAISE_ERROR,
                'Error: raised while training model ' +
                self.algo_factory.get_name() + ' ' + model.get_name()
            ))
            # raise again so the model run aborts
            raise err

        try:
            self.test_model(model, model_dir, is_running, **kwargs)
        except (ArithmeticError, MemoryError, RuntimeError) as err:
            self.event_dispatcher.dispatch(ErrorEventArgs(
                ON_RAISE_ERROR,
                'Error: raised while testing model ' +
                self.algo_factory.get_name() + ' ' + model.get_name()
            ))
            # raise again so the model run aborts
            raise err
