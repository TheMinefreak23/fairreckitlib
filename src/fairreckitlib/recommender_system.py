"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .data.registry import DataRegistry
from .data.split.factory import create_split_factory
from .events import config_event
from .events import io_event
from .events.data_event import get_data_events
from .events.dispatcher import EventDispatcher
from .events.evaluation_event import get_evaluation_events
from .events.io_event import get_io_events
from .events.model_event import get_model_events
from .experiment.constants import EXP_TYPE_PREDICTION
from .experiment.constants import EXP_TYPE_RECOMMENDATION
from .experiment.config import ExperimentConfig
from .experiment.config import experiment_config_to_dict
from .experiment.config import save_config_to_yml
from .experiment.parsing.run import parse_experiment_config_from_yml
from .experiment.parsing.run import parse_experiment_config
from .experiment.run import ExperimentFactories
from .experiment.run import resolve_experiment_start_run
from .metrics.factory import create_metric_factory
from .pipelines.model.factory import create_predictor_model_factory
from .pipelines.model.factory import create_recommender_model_factory
from .threading.thread_experiment import ThreadExperiment
from .threading.thread_processor import ThreadProcessor


class RecommenderSystem:
    """
    Top level API intended for use by applications
    """

    def __init__(self, data_dir, result_dir, verbose=True):
        self.data_registry = DataRegistry(data_dir)
        self.split_factory = create_split_factory()
        self.metric_factory = create_metric_factory()
        self.predictor_factory = create_predictor_model_factory()
        self.recommender_factory = create_recommender_model_factory()

        self.verbose = verbose
        self.event_dispatcher = EventDispatcher()
        for _, (event_id, func_on_event) in enumerate(RecommenderSystem.get_events()):
            self.event_dispatcher.add_listener(event_id, self, func_on_event)

        self.thread_processor = ThreadProcessor()

        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
            self.event_dispatcher.dispatch(
                io_event.ON_MAKE_DIR,
                dir=self.result_dir
            )

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

    def run_experiment(self, config, num_threads=0, validate_config=True):
        """Runs an experiment with the specified configuration.

        Args:
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
            config = parse_experiment_config(experiment_config_to_dict(config), self)
            if config is None:
                return

        os.mkdir(result_dir)
        self.event_dispatcher.dispatch(
            io_event.ON_MAKE_DIR,
            dir=result_dir
        )

        save_config_to_yml(os.path.join(result_dir, 'config'), config)

        self.thread_processor.start(ThreadExperiment(
            config.name,
            self.event_dispatcher,
            factories=ExperimentFactories(
                self.data_registry,
                self.split_factory,
                self.__get_model_factory(config.type),
                self.metric_factory
            ),
            output_dir=result_dir,
            config=config,
            start_run=0, num_runs=1,
            num_threads=num_threads
        ))

    def run_experiment_from_yml(self, file_path, num_threads=0):
        """Runs an experiment from a yml file.

        Args:
            file_path(str): path to the yml file without extension.
            num_threads(int): the max number of threads the experiment can use.
        """
        try:
            config = parse_experiment_config_from_yml(file_path, self)
            if config is None:
                return
        except FileNotFoundError:
            return

        self.run_experiment(config, num_threads=num_threads, validate_config=False)

    def validate_experiment(self, result_dir, num_runs, num_threads=0):
        """Validates an experiment for an additional number of runs.

        Args:
            result_dir(str): path to an existing experiment result directory.
            num_runs(int): the number of runs to validate the experiment.
            num_threads(int): the max number of threads the experiment can use.
        """
        result_dir = os.path.join(self.result_dir, result_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        config_path = os.path.join(result_dir, 'config')
        try:
            config = parse_experiment_config_from_yml(config_path, self)
            if config is None:
                return
        except FileNotFoundError:
            return

        start_run = resolve_experiment_start_run(result_dir)

        self.thread_processor.start(ThreadExperiment(
            config.name,
            self.event_dispatcher,
            factories=ExperimentFactories(
                self.data_registry,
                self.split_factory,
                self.__get_model_factory(config.type),
                self.metric_factory
            ),
            output_dir=result_dir,
            config=config,
            start_run=start_run, num_runs=num_runs,
            num_threads=num_threads
        ))

    def get_available_datasets(self):
        """Gets the available datasets of the recommender system."""
        return self.data_registry.get_info()

    def get_available_metrics(self):
        # TODO refactor
        from .metrics.evaluator_lenskit import EvaluatorLenskit
        from .metrics.evaluator_rexmex import EvaluatorRexmex

        metrics_dict = {}
        all_metrics = list(EvaluatorLenskit.metric_dict.keys()) + list(EvaluatorRexmex.metric_dict.keys())
        from .metrics.common import metric_category_dict
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
        return self.predictor_factory.get_available_algorithms()

    def get_available_recommenders(self):
        """Gets the available recommenders of the recommender system.

        Returns:
            (dict) with all available predictors.
                Each key-value pair describes an API:
                    key(str): name of the API,
                    value(array like): dict entries with recommender name and params.
        """
        return self.recommender_factory.get_available_algorithms()

    def get_available_splitters(self):
        """Gets the available splitters of the recommender system."""
        return self.split_factory.get_available_split_names()

    @staticmethod
    def get_events():
        """Gets all recommender system events.

        Returns:
            (array like) list of pairs in the format (event_id, func_on_event)
        """
        events = [(config_event.ON_PARSE, config_event.on_parse)]

        events += get_io_events()
        events += get_data_events()
        events += get_model_events()
        events += get_evaluation_events()

        return events

    def __get_model_factory(self, experiment_type):
        if experiment_type == EXP_TYPE_PREDICTION:
            return self.predictor_factory
        if experiment_type == EXP_TYPE_RECOMMENDATION:
            return self.recommender_factory

        return None
