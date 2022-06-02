"""This module contains the FairRecKit recommender system.

Classes:

    RecommenderSystem: class that includes the entire recommender system.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import errno
import os
from typing import Any, Dict, Callable, List, Union

from .core.threading.thread_processor import ThreadProcessor
from .data.data_factory import KEY_DATA
from .data.filter.filter_constants import KEY_DATA_FILTERS
from .data.ratings.convert_constants import KEY_RATING_CONVERTER
from .data.set.dataset_registry import DataRegistry
from .data.split.split_constants import KEY_SPLITTING
from .evaluation.evaluation_factory import KEY_EVALUATION
from .experiment.experiment_config import PredictorExperimentConfig, RecommenderExperimentConfig
from .experiment.experiment_config_parsing import Parser
from .experiment.experiment_factory import create_experiment_factory
from .experiment.experiment_run import ExperimentPipelineConfig, resolve_experiment_start_run
from .experiment.experiment_thread import ThreadExperiment
from .model.model_factory import KEY_MODELS


class RecommenderSystem:
    """The FairReckit Recommender System.

    Defines the top level API intended for use by applications.

    Public methods:

    abort_computation
    run_experiment
    run_experiment_from_yml
    validate_experiment
    get_active_computations
    get_available_algorithms
    get_available_datasets
    get_available_data_filters
    get_available_metrics
    get_available_rating_converters
    get_available_splitters
    """

    def __init__(self, data_dir: str, result_dir: str):
        """Construct the RecommenderSystem.

        Initializes the data registry with available datasets on which the
        recommender system depends. It raises an IOError when the specified
        data directory does not exist. The result directory however is
        created when non-existing.

        Args:
            data_dir: path to the directory that contains the datasets.
            result_dir: path to the directory to store computation results.
        """
        try:
            self.data_registry = DataRegistry(data_dir)
        except IOError as err:
            raise IOError('Failed to initialize DataRegistry: '
                          'unknown data directory => ' + data_dir) from err

        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.experiment_factory = create_experiment_factory(self.data_registry)
        self.thread_processor = ThreadProcessor()

    def abort_computation(self, computation_name: str) -> bool:
        """Attempt to abort a running computation thread.

        The name of the computation is the same as specified in the configuration
        file when the computation is run. If the computation with the specified
        name does not exist this function returns False.
        Note that the computation is only requested to abort as soon as possible,
        therefore it might take a while until the computation actually stops.

        Args:
            computation_name: name of the active computation thread to abort.

        Returns:
            whether the computation is successfully requested to abort.
        """
        if not self.thread_processor.is_active_thread(computation_name):
            return False

        self.thread_processor.stop(computation_name)
        return True

    def run_experiment(
            self,
            config: Union[PredictorExperimentConfig, RecommenderExperimentConfig],
            *,
            events: Dict[str, Callable[[Any], None]] = None,
            num_threads: int = 0,
            verbose: bool = True,
            validate_config: bool = True) -> bool:
        """Run an experiment with the specified configuration.

        It is advised to validate the configuration (default) before running the
        experiment, to make sure the configuration describes a valid experiment.
        The configuration is invalid when it contains no selected datasets or
        models, only the evaluation is optional. If the configuration is invalidated
        this function will return False.

        This functions raises an IOError when the result already exists or a TypeError
        when the provided configuration is not a valid experiment configuration.
        Lastly a KeyError is raised when a computation with the same name is already active.

        Args:
            events: the external events to dispatch during the experiment.
            config: the configuration of the experiment.
            num_threads: the max number of threads the experiment can use.
            verbose: whether the internal events should give verbose output.
            validate_config: whether to validate the configuration beforehand.

        Returns:
            whether the experiment successfully started.
        """
        result_dir = os.path.join(self.result_dir, config.name)
        if os.path.isdir(result_dir):
            raise IOError('Result already exists: ' + result_dir)

        if not isinstance(config, (PredictorExperimentConfig, RecommenderExperimentConfig)):
            raise TypeError('Invalid experiment configuration type.')

        if validate_config:
            parser = Parser(verbose)
            config = parser.parse_experiment_config(config.to_yml_format(),
                                                    self.data_registry,
                                                    self.experiment_factory)
            if config is None:
                return False

        self.thread_processor.start(ThreadExperiment(
            config.name,
            events,
            verbose,
            pipeline_config=ExperimentPipelineConfig(
                result_dir,
                self.data_registry,
                self.experiment_factory,
                config,
                0,
                1,
                num_threads
            )
        ))

        return True

    def run_experiment_from_yml(
            self,
            file_path: str,
            *,
            events: Dict[str, Callable[[Any], None]] = None,
            num_threads: int = 0,
            verbose: bool = True) -> bool:
        """Run an experiment from a yml file.

        The configuration in the file is validated before starting the experiment.
        It is invalid when it contains no selected datasets or models,
        only the evaluation is optional.  If the configuration is invalidated
        this function will return False.

        This function raises a FileNotFoundError when the specified yml file
        is not found or an IOError when the result already exists. Lastly a KeyError
        is raised when a computation with the same name is already active.

        Args:
            events: the external events to dispatch during the experiment.
            file_path: path to the yml file without extension.
            num_threads: the max number of threads the experiment can use.
            verbose: whether the internal events should give verbose output.

        Returns:
            whether the experiment successfully started.
        """
        try:
            parser = Parser(verbose)
            config = parser.parse_experiment_config_from_yml(file_path,
                                                             self.data_registry,
                                                             self.experiment_factory)
            if config is None:
                return False
        except FileNotFoundError as err:
            raise FileNotFoundError(errno.ENOENT, 'Config file not found', file_path) from err

        return self.run_experiment(
            config,
            events=events,
            num_threads=num_threads,
            verbose=verbose,
            validate_config=False
        )

    def validate_experiment(
            self,
            result_dir: str,
            num_runs: int,
            *,
            events: Dict[str, Callable[[Any], None]] = None,
            num_threads: int = 0,
            verbose: bool = True) -> bool:
        """Validate an experiment for an additional number of runs.

        The configuration file is expected to be stored in the specified result directory.
        Moreover, the configuration is validated before starting the experiment validation.
        If the configuration is invalidated this function will return False.

        This functions raises an IOError when the result already exists or
        a FileNotFoundError when the configuration file was not found in the
        specified result directory. Lastly a KeyError is raised when a computation
        with the same name is already active, meaning it is not possible to validate
        an active experiment computation until it is done.

        Args:
            events: the external events to dispatch during the experiment.
            result_dir: path to an existing experiment result directory.
            num_runs: the number of runs to validate the experiment.
            num_threads: the max number of threads the experiment can use.
            verbose: whether the internal events should give verbose output.

        Returns:
            whether the experiment successfully started.
        """
        result_dir = os.path.join(self.result_dir, result_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        config_path = os.path.join(result_dir, 'config')
        try:
            parser = Parser(verbose)
            config = parser.parse_experiment_config_from_yml(config_path,
                                                             self.data_registry,
                                                             self.experiment_factory)
            if config is None:
                return False
        except FileNotFoundError as err:
            raise FileNotFoundError(errno.ENOENT, 'Config file not found', config_path) from err

        self.thread_processor.start(ThreadExperiment(
            config.name,
            events,
            verbose,
            pipeline_config=ExperimentPipelineConfig(
                result_dir,
                self.data_registry,
                self.experiment_factory,
                config,
                resolve_experiment_start_run(result_dir),
                num_runs,
                num_threads
            )
        ))

        return True

    def get_active_computations(self) -> List[str]:
        """Get the names of any active computations.

        Returns:
            a list of computations names that are currently running.
        """
        return self.thread_processor.get_active_threads()

    def get_available_algorithms(self, model_type: str = None):
        """Get the available algorithms of the recommender system.

        Args:
            model_type: type of model to query for availability, accepted values are
                TYPE_PREDICTION, TYPE_RECOMMENDATION or None.

        Returns:
            a dictionary with the availability of algorithms categorized by API.
        """
        return self.experiment_factory.get_sub_availability(
            KEY_MODELS,
            sub_type=model_type
        )

    def get_available_datasets(self) -> Dict[str, Any]:
        """Get the available datasets of the recommender system.

        Returns:
            a dictionary where the key corresponds to the dataset name and
                the value corresponds to the matrix information dictionary.
        """
        return self.data_registry.get_info()

    def get_available_data_filters(self) -> Dict[str, Any]:
        """Get the available data filters of the recommender system.

        Returns:
            a dictionary with the availability of data filters.
        """
        # TODO the data filter factory does not exist
        return self.experiment_factory.get_sub_availability(
            KEY_DATA,
            sub_type=KEY_DATA_FILTERS
        )

    def get_available_metrics(self, eval_type: str = None) -> Dict[str, Any]:
        """Get the available metrics of the recommender system.

        Args:
            eval_type(str): type of evaluation to query for availability, accepted values are
                TYPE_PREDICTION, TYPE_RECOMMENDATION or None.

        Returns:
            a dictionary with the availability of metrics categorized by evaluation type.
        """
        return self.experiment_factory.get_sub_availability(
            KEY_EVALUATION,
            sub_type=eval_type
        )

    def get_available_rating_converters(self) -> Dict[str, Any]:
        """Get the available data rating converters of the recommender system.

        Returns:
            a dictionary with the availability of rating converters.
        """
        return self.experiment_factory.get_sub_availability(
            KEY_DATA,
            sub_type=KEY_RATING_CONVERTER
        )

    def get_available_splitters(self) -> Dict[str, Any]:
        """Get the available data splitters of the recommender system.

        Returns:
            a dictionary with the availability of data splitters.
        """
        return self.experiment_factory.get_sub_availability(
            KEY_DATA,
            sub_type=KEY_SPLITTING
        )
