"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import os

from .data.registry import DataRegistry
from .data.split.factory import get_split_factory
from .events import config_event
from .events import io_event
from .events.dispatcher import EventDispatcher
from .experiment.constants import EXP_TYPE_PREDICTION
from .experiment.constants import EXP_TYPE_RECOMMENDATION
from .experiment.config import ExperimentConfig
from .experiment.parsing.run import parse_experiment_config_from_yml
from .experiment.run import ExperimentFactories
from .experiment.run import run_experiment
from .pipelines.model.factory import create_predictor_model_factory
from .pipelines.model.factory import create_recommender_model_factory


class RecommenderSystem:
    """
    Top level API intended for use by applications
    """

    def __init__(self, data_dir, result_dir, verbose=True):
        self.data_registry = DataRegistry(data_dir)
        self.split_factory = get_split_factory()
        self.predictor_factory = create_predictor_model_factory()
        self.recommender_factory = create_recommender_model_factory()

        self.verbose = verbose
        self.event_dispatcher = EventDispatcher()

        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
            self.event_dispatcher.dispatch(
                io_event.ON_MAKE_DIR,
                dir=self.result_dir
            )

    def abort_experiment(self):
        """TODO"""
        # TODO cancel an active experiment computation
        raise NotImplementedError()

    def evaluate_experiment(self, experiment_dir, config):
        """TODO"""
        result_dir = os.path.join(self.result_dir, experiment_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        # TODO evaluate additional metrics
        raise NotImplementedError()

    def run_experiment(self, config, num_threads=0, validate_config=False):
        """Runs an experiment with the specified configuration.

        Args:
            config(ExperimentConfig): the configuration of the experiment.
            num_threads(int): the max number of threads the model pipeline can use.
            validate_config(bool): whether to validate the configuration beforehand.
        """
        result_dir = os.path.join(self.result_dir, config.name)
        if os.path.isdir(result_dir):
            raise IOError('Result already exists: ' + result_dir)

        if not isinstance(config, ExperimentConfig):
            raise ValueError('Invalid experiment configuration')

        if validate_config:
            # TODO and set argument default to True
            raise NotImplementedError()

        self.event_dispatcher.add_listener(io_event.ON_MAKE_DIR, self, io_event.on_make_dir)

        os.mkdir(result_dir)
        self.event_dispatcher.dispatch(
            io_event.ON_MAKE_DIR,
            dir=result_dir
        )

        run_0_dir = os.path.join(result_dir, 'run_0')
        os.mkdir(run_0_dir)
        self.event_dispatcher.dispatch(
            io_event.ON_MAKE_DIR,
            dir=run_0_dir
        )

        self.event_dispatcher.remove_listener(io_event.ON_MAKE_DIR, self, io_event.on_make_dir)

        result_overview = run_experiment(
            run_0_dir,
            ExperimentFactories(
                self.data_registry,
                self.split_factory,
                self.__get_model_factory(config.type)
            ),
            config,
            self.event_dispatcher,
            num_threads=num_threads,
            verbose=self.verbose
        )

        self.write_storage_file(run_0_dir, result_overview)

    def run_experiment_from_yml(self, file_path, num_threads=0):
        """Runs an experiment from a yml file.

        Args:
            file_path(str): path to the yml file without extension.
            num_threads(int): the max number of threads the model pipeline can use.
        """
        self.event_dispatcher.add_listener(config_event.ON_PARSE, self, config_event.on_parse)
        config = parse_experiment_config_from_yml(file_path, self)
        self.event_dispatcher.remove_listener(config_event.ON_PARSE, self, config_event.on_parse)

        # parsing failed
        if config is None:
            return

        self.run_experiment(config, num_threads=num_threads, validate_config=False)

    def validate_experiment(self, experiment_dir, num_runs):
        """TODO"""
        result_dir = os.path.join(self.result_dir, experiment_dir)
        if not os.path.isdir(result_dir):
            raise IOError('Result does not exist: ' + result_dir)

        # TODO run the same experiment again for 'num_runs'
        raise NotImplementedError()

    def write_storage_file(self, run_0_dir, results):
        """Write a JSON file with overview of the results file paths"""
        import json

        formatted_results = map(lambda result: {
                'name': result['dataset'] + '_' + result['model'],
                'evaluation_path': result['dir'] + '\\evaluation.tsv',
                'ratings_path': result['dir'] + '\\ratings.tsv',
                'ratings_settings_path': result['dir'] + '\\settings.tsv'
            }, results)

        with open(run_0_dir+'/overview.json', 'w') as file:
            json.dump({'overview': list(formatted_results)}, file, indent=4)

    def get_available_datasets(self):
        """Gets the available datasets of the recommender system."""
        return self.data_registry.get_info()

    def get_available_metrics(self):
        # TODO refactor
        from metrics.evaluator_lenskit import EvaluatorLenskit
        from metrics.evaluator_rexmex import EvaluatorRexmex
        EvaluatorLenskit.metricDict.keys() + EvaluatorRexmex.metricDict.keys()
        # raise NotImplementedError()

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
        raise NotImplementedError()

    def __get_model_factory(self, experiment_type):
        if experiment_type == EXP_TYPE_PREDICTION:
            return self.predictor_factory
        if experiment_type == EXP_TYPE_RECOMMENDATION:
            return self.recommender_factory

        return None
