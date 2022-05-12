"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
import errno
import os

from .core.event_io import get_io_events
from .core.threading.thread_processor import ThreadProcessor
from .data.data_factory import KEY_DATASETS
from .data.pipeline.data_event import get_data_events
from .data.set.dataset_registry import DataRegistry
from .data.ratings.rating_modifier_factory import KEY_RATING_MODIFIER
from .data.split.split_factory import KEY_SPLITTING
from .evaluation.evaluation_factory import KEY_EVALUATION
from .evaluation.pipeline.evaluation_event import get_evaluation_events
from .experiment.experiment_event import get_experiment_events
from .experiment.experiment_config import ExperimentConfig
from .experiment.experiment_config import experiment_config_to_dict
from .experiment.experiment_config_parsing import Parser
from .experiment.experiment_factory import create_experiment_factory
from .experiment.experiment_run import resolve_experiment_start_run
from .experiment.experiment_thread import ThreadExperiment
from .model.model_factory import KEY_MODELS
from .model.pipeline.model_event import get_model_events


class RecommenderSystem:
    """
    Top level API intended for use by applications
    """

    def __init__(self, data_dir, result_dir):
        self.data_registry = DataRegistry(data_dir)
        self.experiment_factory = create_experiment_factory()
        self.thread_processor = ThreadProcessor()
        self.result_dir = result_dir

    def abort_computation(self, computation_name):
        """Attempts to abort a running computation thread.

        Args:
            computation_name(str): name of the active computation thread.
        """
        if not self.thread_processor.is_active_thread(computation_name):
            return

        self.thread_processor.stop(computation_name)

    def run_experiment(self, events, config, num_threads=0, verbose=True, validate_config=True, ):
        """Runs an experiment with the specified configuration.

        Args:
            events(dict): the external events to dispatch during the experiment.
            verbose(bool): whether the internal events should give verbose output.
            config(ExperimentConfig): the configuration of the experiment.
            num_threads(int): the max number of threads the experiment can use.
            validate_config(bool): whether to validate the configuration beforehand.
        """

        result_dir = os.path.join(self.result_dir, config.name)
        if os.path.isdir(result_dir):
            raise IOError('Result already exists: ' + result_dir)

        if not isinstance(config, ExperimentConfig):
            raise TypeError('Invalid experiment configuration type.')

        if validate_config:
            parser = Parser(verbose)
            config = parser.parse_experiment_config(experiment_config_to_dict(config),
                                                    self.data_registry,
                                                    self.experiment_factory)
            if config is None:
                return

        self.start_thread_experiment(events, result_dir, config, num_threads, verbose)

    def run_experiment_from_yml(self, events, file_path, verbose=True, num_threads=0):
        """Runs an experiment from a yml file.

        Args:
            events(dict): the external events to dispatch during the experiment.
            verbose(bool): whether the internal events should give verbose output.
            file_path(str): path to the yml file without extension.
            num_threads(int): the max number of threads the experiment can use.
        """
        try:
            parser = Parser(verbose)
            config = parser.parse_experiment_config_from_yml(file_path,
                                                             self.data_registry,
                                                             self.experiment_factory)
            if config is None:
                return
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, 'Config file not found', file_path)
            return

        self.run_experiment(events, config, num_threads=num_threads, verbose=verbose, validate_config=False)

    def validate_experiment(self, events, result_dir, num_runs, num_threads=0, verbose=True):
        """Validates an experiment for an additional number of runs.

        Args:
            events(list(tuple)):the external events to dispatch during the experiment.
            result_dir(str): path to an existing experiment result directory.
            num_runs(int): the number of runs to validate the experiment.
            num_threads(int): the max number of threads the experiment can use.
        """
        result_dir = os.path.join(self.result_dir, result_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        config_path = os.path.join(result_dir, 'config')
        try:
            events = {event_id: func_on_event for (event_id, func_on_event) in RecommenderSystem.get_events()}
            parser = Parser(verbose)
            config = parser.parse_experiment_config_from_yml(config_path,
                                                             self.data_registry,
                                                             self.experiment_factory)
            if config is None:
                return
        except FileNotFoundError:
            return

        start_run = resolve_experiment_start_run(result_dir)
        self.start_thread_experiment(events, result_dir, config, num_threads, verbose, start_run, num_runs)

    def start_thread_experiment(self, events, result_dir, config, num_threads, verbose, start_run=0, num_runs=1):
        # Add external events.
        for (event_id, func_on_event) in RecommenderSystem.get_events():
            external_func = None
            if event_id in events:
                external_func = events[event_id]
            events[event_id] = (func_on_event, external_func)

        # Start thread with thread experiment.
        self.thread_processor.start(ThreadExperiment(
            config.name,
            events,
            verbose=verbose,
            registry=self.data_registry,
            factory=self.experiment_factory,
            output_dir=result_dir,
            config=config,
            start_run=start_run, num_runs=num_runs,
            num_threads=num_threads,
        ))

    def get_available_algorithms(self, model_type=None):
        """Gets the available algorithms of the recommender system.

        Args:
            model_type(str): type of model to query for availability, accepted values are
                TYPE_PREDICTION, TYPE_RECOMMENDATION or None.

        Returns:
            (dict) with available algorithms divided into types when model_type is not specified.
                For each type it contains a dict, with key-value pair describing an API:
                    key(str): name of the algorithm API,
                    value(array like): list of dict entries with algorithm name and params.
        """
        return self.__get_factory_sub_availability(KEY_MODELS, model_type)


    def get_available_datasets(self):
        """Gets the available datasets of the recommender system."""
        return self.data_registry.get_info()

    def get_available_metrics(self, eval_type=None):
        """Gets the available metrics of the recommender system.

        Args:
            eval_type(str): type of evaluation to query for availability, accepted values are
                TYPE_PREDICTION, TYPE_RECOMMENDATION or None.

        Returns:
            (dict) with available metrics divided into types when eval_type is not specified.
                For each type it contains a dict, with key-value pair describing a metric category:
                    key(str): name of the metric category,
                    value(array like): list of dict entries with metric name and params.
        """
        return self.__get_factory_sub_availability(KEY_EVALUATION, eval_type)

    def get_available_rating_modifiers(self):
        return self.__get_factory_sub_availability(KEY_DATASETS, sub_type=KEY_RATING_MODIFIER)

    def get_available_splitters(self):
        """Gets the available data splitters of the recommender system.

        Returns:
            (dict) with all available models.
                Each key-value pair describes an API:
                    key(str): name of the model API,
                    value(array like): list of dict entries with predictor name and params.
        """
        return self.__get_factory_sub_availability(KEY_DATASETS, sub_type=KEY_SPLITTING)

    def __get_factory_sub_availability(self, factory_name, sub_type=None):
        factory = self.experiment_factory.get_factory(factory_name)
        if sub_type is None:
            return factory.get_available()

        type_factory = factory.get_factory(sub_type)
        if type_factory is None:
            return {}

        return type_factory.get_available()

    @staticmethod
    def get_events():
        """Gets all recommender system events.

        Returns:
            (array like) list of pairs in the format (event_id, func_on_event)
        """
        events = []
        events += get_experiment_events()
        events += get_io_events()
        events += get_data_events()
        events += get_model_events()
        events += get_evaluation_events()

        return events
