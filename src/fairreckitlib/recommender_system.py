"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .data.set.dataset_registry import DataRegistry
from .data.split.split_factory import create_split_factory
from .events.data_event import get_data_events
from .events.evaluation_event import get_evaluation_events
from .events.experiment_event import get_experiment_events
from .events.io_event import get_io_events
from .events.model_event import get_model_events
from .experiment.config import ExperimentConfig
from .experiment.config import experiment_config_to_dict
from .experiment.parsing.run import Parser
from .experiment.run import ExperimentFactories
from .experiment.run import resolve_experiment_start_run
from src.fairreckitlib.evaluation.metrics.factory import create_metric_factory
from .model.model_factory import create_model_factory
from .threading.thread_experiment import ThreadExperiment
from .threading.thread_processor import ThreadProcessor


class RecommenderSystem:
    """
    Top level API intended for use by applications
    """

    def __init__(self, data_dir, result_dir):
        self.data_registry = DataRegistry(data_dir)
        self.split_factory = create_split_factory()
        self.metric_factory = create_metric_factory()
        self.model_factory = create_model_factory()

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

    def evaluate_experiment(self, experiment_dir, config):
        """TODO"""
        result_dir = os.path.join(self.result_dir, experiment_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        # TODO evaluate additional metrics
        raise NotImplementedError()

    def run_experiment(self, events, config, num_threads=0, verbose=True, validate_config=True, ):
        """Runs an experiment with the specified configuration.

        Args:
            events(list(tuple)): the external events to dispatch during the experiment.
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
                                                    self.split_factory,
                                                    self.predictor_factory,
                                                    self.recommender_factory,
                                                    self.metric_factory)
            if config is None:
                return

        self.start_thread_experiment(events, result_dir, config, num_threads, verbose)

    def run_experiment_from_yml(self, file_path, verbose=True, num_threads=0):
        """Runs an experiment from a yml file.

        Args:
            file_path(str): path to the yml file without extension.
            num_threads(int): the max number of threads the experiment can use.
        """
        try:
            parser = Parser(verbose)
            config = parser.parse_experiment_config_from_yml(file_path,
                                                             self.data_registry,
                                                             self.split_factory,
                                                             self.predictor_factory,
                                                             self.recommender_factory,
                                                             self.metric_factory)
            if config is None:
                return
        except FileNotFoundError:
            return

        self.run_experiment(config, num_threads=num_threads, validate_config=False)

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
                                                             self.split_factory,
                                                             self.predictor_factory,
                                                             self.recommender_factory,
                                                             self.metric_factory)
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
            factories=ExperimentFactories(
                self.data_registry,
                self.split_factory,
                self.model_factory,
                self.metric_factory
            ),
            output_dir=result_dir,
            config=config,
            start_run=start_run, num_runs=num_runs,
            num_threads=num_threads,
        ))

    def get_available_datasets(self):
        """Gets the available datasets of the recommender system."""
        return self.data_registry.get_info()

    def get_available_metrics(self):
        # TODO refactor
        from src.fairreckitlib.evaluation.metrics.evaluator_lenskit import EvaluatorLenskit
        from src.fairreckitlib.evaluation.metrics.evaluator_rexmex import EvaluatorRexmex

        metrics_dict = {}
        all_metrics = list(EvaluatorLenskit.metric_dict.keys()) + list(EvaluatorRexmex.metric_dict.keys())
        from src.fairreckitlib.evaluation.metrics.common import metric_category_dict
        for category in metric_category_dict:
            # Only return the metrics that are available in the used libraries
            intersection = [metric.value for metric in metric_category_dict[category] if metric in all_metrics]
            metrics_dict[category.value] = intersection
        return metrics_dict

    def get_available_predictors(self):
        """Gets the available predictors of the recommender system.

        Returns:
            (dict) with all available predictors.
                Each key-value pair describes an API:
                    key(str): name of the API,
                    value(array like): dict entries with predictor name and params.
        """
        return self.model_factory.get_available_algorithms()

    def get_available_recommenders(self):
        """Gets the available recommenders of the recommender system.

        Returns:
            (dict) with all available predictors.
                Each key-value pair describes an API:
                    key(str): name of the API,
                    value(array like): dict entries with recommender name and params.
        """
        return self.model_factory.get_available_algorithms()

    def get_available_splitters(self):
        """Gets the available splitters of the recommender system."""
        return self.split_factory.get_available_split_names()

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
